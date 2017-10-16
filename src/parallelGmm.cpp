#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "barrier.h"
#include "gmm.h"
#include "seqGmm.h"
#include "parallelGmm.h"
#include "util.h"

void checkStopCriteria(void* untypedArgs) {
	SharedThreadStartArgs* sargs = (SharedThreadStartArgs*) untypedArgs;
	assert(sargs != NULL);

	sargs->prevLogL = sargs->currentLogL;
	sargs->currentLogL = 0;
	for(size_t i = 0; i < sargs->numProcesses; ++i) {
		sargs->currentLogL += sargs->logLK[i];
	}

	sargs->shouldContinue = shouldContinue(
		sargs->prevLogL, sargs->currentLogL,
		sargs->tolerance
	);
}

void computeGammaSum(void* untypedArgs) {
	SharedThreadStartArgs* sargs = (SharedThreadStartArgs*) untypedArgs;
	assert(sargs != NULL);

	sargs->logGammaSum = calcLogGammaSum( 
		sargs->logpi, 
		sargs->gmm->numComponents, 
		sargs->logGamma
	);
}

void* parallelFitStart(void* untypedArgs) {
	ThreadStartArgs* args = (ThreadStartArgs*) untypedArgs;
	assert(args != NULL);

	SharedThreadStartArgs* sargs = args->shared;

	const size_t numComponents = sargs->gmm->numComponents;

	const float* X = sargs->X;
	const size_t numPoints = sargs->numPoints;
	const size_t pointDim = sargs->pointDim;

	float* logpi = sargs->logpi;
	float* loggamma = sargs->loggamma;
	float* logGamma = sargs->logGamma;

	float* xm = (float*)checkedCalloc(pointDim, sizeof(float));
	float* outerProduct = (float*)checkedCalloc(pointDim * pointDim, sizeof(float));

	size_t iteration = 0;

	do {
		// --- E-Step ---

		// Compute gamma (parallel across components)
		calcLogMvNorm(
			sargs->gmm->components, numComponents, 
			args->componentStart, args->componentEnd, 
			X, numPoints, pointDim,
			loggamma
		);

		arriveAt(sargs->barrier, NULL, NULL);

		// parallel across points
		logLikelihoodAndGammaNK(
			logpi, numComponents,
			loggamma, numPoints,
			args->pointStart, args->pointEnd,
			& sargs->logLK[args->id]
		);

		arriveAt(sargs->barrier, sargs, checkStopCriteria);

		if(!sargs->shouldContinue) {
			break;
		}

		arriveAt(sargs->barrier, NULL, NULL);

		// Afterwards, everybody does
		calcLogGammaK(
			loggamma, numPoints, 
			args->componentStart, args->componentEnd, 
			logGamma, numComponents
		);

		arriveAt(sargs->barrier, sargs, computeGammaSum);

		// --- M-Step ---
		performMStep(
			sargs->gmm->components, numComponents,
			args->componentStart, args->componentEnd,
			logpi, loggamma, logGamma, sargs->logGammaSum,
			X, numPoints, pointDim,
			outerProduct, xm
		);

	} while (++iteration < sargs->maxIterations);

	free(xm);
	free(outerProduct);

	return NULL;
}

GMM* parallelFit(
	const float* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents,
	const size_t maxIterations
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);
	assert(maxIterations > 0);

	if(numComponents == 1) {
		return fit(X, numPoints, pointDim, numComponents, maxIterations);
	}

	GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	size_t numProcessors = 8;
	if(numComponents < numProcessors) {
		numProcessors = numComponents;
	}

	Barrier barrier;
	initBarrier(&barrier, numProcessors);

	SharedThreadStartArgs stsa;
	stsa.X = X;
	stsa.numPoints = numPoints;
	stsa.pointDim = pointDim;

	stsa.gmm = gmm;

	stsa.tolerance = 1e-8;
	stsa.maxIterations = maxIterations;
	stsa.prevLogL = -INFINITY;
	stsa.currentLogL = -INFINITY;
	stsa.logpi = (float*)checkedCalloc(numComponents, sizeof(float));
	stsa.loggamma = (float*)checkedCalloc(numComponents * numPoints, sizeof(float));
	stsa.logGamma = (float*)checkedCalloc(numComponents, sizeof(float));
	stsa.logGammaSum = 0.0;
	stsa.logLK = (float*)checkedCalloc(numProcessors, sizeof(float));
	stsa.numProcesses = numProcessors;
	stsa.barrier = &barrier;

	for(size_t k = 0; k < numComponents; ++k) {
		const float pik = gmm->components[k].pi;
		assert(pik >= 0);
		stsa.logpi[k] = logf(pik);
	}

	size_t pointResidual = numPoints % numProcessors;
	size_t pointsPerProcessor = (numPoints - pointResidual) / numProcessors;

	size_t componentResidual = numComponents % numProcessors;
	size_t componentsPerProcessor = (numComponents - componentResidual) / numProcessors;

	ThreadStartArgs args[numProcessors];
	for(size_t i = 0; i < numProcessors; ++i) {
		args[i].id = i;	
		args[i].shared = &stsa;

		if(i == 0) {
			args[i].pointStart = 0;
			args[i].componentStart = 0;
		} else {
			args[i].pointStart = args[i - 1].pointEnd;
			args[i].componentStart = args[i - 1].componentEnd;
		}

		args[i].pointEnd = args[i].pointStart + pointsPerProcessor;
		if(pointResidual > 0) {
			--pointResidual;
			++args[i].pointEnd;
		}

		assert(args[i].pointEnd <= numPoints);

		args[i].componentEnd = args[i].componentStart + componentsPerProcessor;
		if(componentResidual > 0) {
			--componentResidual;
			++args[i].componentEnd;
		}
	}

	size_t numThreads = numProcessors - 1;
	pthread_t threads[numThreads];

	for(size_t i = 0; i < numThreads; ++i) {
		int result = pthread_create(&threads[i], NULL, parallelFitStart, &args[i]);
		if(result != 0) {
			break;
		}
	}

	parallelFitStart(&args[numThreads]);

	for(size_t i = 0; i < numThreads; ++i) {
		int result = pthread_join(threads[i], NULL);
		if(result != 0) {
			break;
		}
	}

	destroyBarrier(&barrier);

	free(stsa.logpi);
	free(stsa.loggamma);
	free(stsa.logGamma);
	free(stsa.logLK);

	return gmm;
}
