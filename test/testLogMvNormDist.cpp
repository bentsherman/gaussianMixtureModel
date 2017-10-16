#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#include "component.h"
#include "cudaWrappers.h"

typedef void (*test1DStandardNormalWrapper)(const size_t, const size_t, const float*, const float*, const float*, const float, float*);

void test1DStandardNormal( test1DStandardNormalWrapper target ) {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;

	float sigmaL[pointDim * pointDim];
	memset(sigmaL, 0, pointDim * pointDim * sizeof(float));
	for(size_t i = 0; i < pointDim; ++i) {
		sigmaL[i * pointDim + i] = 1;
	}

	float det = 1;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i] * sigmaL[i * pointDim + i];
	}

	float logNormalizer = -0.5 * pointDim * logf(2.0 * M_PI) - 0.5 * logf(det);

	float mu[pointDim];
	memset(mu, 0, pointDim * sizeof(float));

	float X[pointDim * numPoints];
	memset(X, 0, pointDim * numPoints * sizeof(float));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i * pointDim + 0] = 3.0 * ( ( (float)i - (float)numPoints/2 ) / (float)(numPoints/2.0) );
	}

	float logP[numPoints];
	memset(logP, 0, numPoints * sizeof(float));

	target(
		numPoints, pointDim,
		X, mu, sigmaL, logNormalizer,
		logP
	);

	float normalizer = -0.5 * logf(2.0 * M_PI);
	for(size_t i = 0; i < numPoints; ++i) {
		float x = X[i];
		float expected = -0.5 * x * x + normalizer;
		float actual = logP[i];
		assert(actual != -INFINITY);
		assert(actual != INFINITY);
		assert(actual == actual);

		float absDiff = fabsf(expected - actual);
		if(absDiff >= FLT_EPSILON) {
			printf("f(%.16f) = %.16f, but should equal = %.16f; absDiff = %.16f\n",
				x, actual, expected, absDiff);
		}

		assert(absDiff < FLT_EPSILON);
	}
}

void cpuLogMvNormDistWrapper(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* mu, const float* sigmaL, const float logNormalizer,
	float* logP
) {
	Component phi;
	phi.mu = (float *)(mu);
	phi.sigmaL = (float *)(sigmaL);
	phi.normalizer = logNormalizer;
	logMvNormDist(&phi, pointDim, X, numPoints, logP);
}

void gpuLogMvNormDistWrapper(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* mu, const float* sigmaL, const float logNormalizer,
	float* logP
) {
	gpuLogMVNormDist(
		numPoints, pointDim,
		X, mu, sigmaL,
		logP
	);
}

void test1DStandardNormalParallelRun() {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;

	float sigmaL[pointDim * pointDim];
	memset(sigmaL, 0, pointDim * pointDim * sizeof(float));
	for(size_t i = 0; i < pointDim; ++i) {
		sigmaL[i * pointDim + i] = 1;
	}

	float det = 1;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i] * sigmaL[i * pointDim + i];
	}

	float logNormalizer = -0.5 * pointDim * logf(2.0 * M_PI) - 0.5 * logf(det);

	float mu[pointDim];
	memset(mu, 0, pointDim * sizeof(float));

	float X[pointDim * numPoints];
	memset(X, 0, pointDim * numPoints * sizeof(float));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i * pointDim + 0] = 3.0 * ( ( (float)i - (float)numPoints/2 ) / (float)(numPoints/2.0) );
	}

	float seqLogP[numPoints];
	memset(seqLogP, 0, numPoints * sizeof(float));
	cpuLogMvNormDistWrapper(numPoints, pointDim, X, mu, sigmaL, logNormalizer, seqLogP);

	float cudaLogP[numPoints];
	memset(cudaLogP, 0, numPoints * sizeof(float));
	gpuLogMvNormDistWrapper(numPoints, pointDim, X, mu, sigmaL, logNormalizer, cudaLogP);

	for(size_t i = 0; i < numPoints; ++i) {
		float x = X[i];
		float seqValue = seqLogP[i];
		float cudaValue = cudaLogP[i];

		float absDiff = fabsf(seqValue - cudaValue);
		if(absDiff >= FLT_EPSILON) {
			printf("Seq. f(%.16f) = %.16f, but Cuda f(%.16f) = %.16f; absDiff = %.16f\n",
				x, x, seqValue, cudaValue, absDiff);
		}

		assert(absDiff < FLT_EPSILON);
	}
}


int main(int argc, char** argv) {
	test1DStandardNormal(cpuLogMvNormDistWrapper);
	test1DStandardNormal(gpuLogMvNormDistWrapper);
	test1DStandardNormalParallelRun();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
