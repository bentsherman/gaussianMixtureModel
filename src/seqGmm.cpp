#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdexcept>

#include "gmm.h"
#include "seqGmm.h"
#include "util.h"

GMM* fit(
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

	GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	const float tolerance = 1e-8;
	size_t iteration = 0;
	float prevLogL = -INFINITY;
	float currentLogL = -INFINITY;

	float* logpi = (float*)checkedCalloc(numComponents, sizeof(float));
	float* loggamma = (float*)checkedCalloc(numPoints * numComponents, sizeof(float));
	float* logGamma = (float*)checkedCalloc(numComponents, sizeof(float));

	float* xm = (float*)checkedCalloc(pointDim, sizeof(float));
	float* outerProduct = (float*)checkedCalloc(pointDim * pointDim, sizeof(float));

	for(size_t k = 0; k < numComponents; ++k) {
		const float pik = gmm->components[k].pi;
		assert(pik >= 0);
		logpi[k] = logf(pik);
	}

	try {
		do {
			// --- E-Step ---

			// Compute gamma
			calcLogMvNorm(
				gmm->components, numComponents,
				0, numComponents,
				X, numPoints, pointDim,
				loggamma
			);

			// 2015-09-20 GEL Eliminated redundant mvNorm clac in logLikelihood by
			// passing in precomputed gamma values. Also moved loop termination here
			// since likelihood determines termination. Result: 1.3x improvement in
			// execution time.  (~8 ms to ~6 ms on oldFaithful.dat)
			// 2017-04-14 GEL Decided to fuse logLikelihood and Gamma NK calculation
			// since they both rely on the log p(x) calculation, and it would be
			// wasteful to compute and store p(x), since log L and gamma NK are only
			// consumers of that data.
			prevLogL = currentLogL;
			logLikelihoodAndGammaNK(
				logpi, numComponents,
				loggamma, numPoints,
				0, numPoints,
				& currentLogL
			);

			if(!shouldContinue(prevLogL, currentLogL, tolerance)) {
				break;
			}

			// Let Gamma[component] = \Sum_point gamma[component, point]
			calcLogGammaK(
				loggamma, numPoints,
				0, numComponents,
				logGamma, numComponents
			);

			float logGammaSum = calcLogGammaSum(logpi, numComponents, logGamma);

			// --- M-Step ---
			performMStep(
				gmm->components, numComponents,
				0, numComponents,
				logpi, loggamma, logGamma, logGammaSum,
				X, numPoints, pointDim,
				outerProduct, xm
			);

		} while (++iteration < maxIterations);

		// save outputs
		gmm->failed = false;
		gmm->y_pred = calcLabels(loggamma, numPoints, numComponents);
		gmm->logL = currentLogL;
	}
	catch ( std::runtime_error& e ) {
		fprintf(stderr, "warning: model failed\n");
		gmm->failed = true;
	}

	free(logpi);
	free(loggamma);
	free(logGamma);

	free(xm);
	free(outerProduct);

	return gmm;
}
