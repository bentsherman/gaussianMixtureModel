#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#include "component.h"
#include "gmm.h"
#include "cudaWrappers.h"

typedef float (*GmmLogLikelihoodWrapper)(const size_t, const size_t, const float*, float*);

void test1DStandardNormalLogLikelihood(GmmLogLikelihoodWrapper target) {
	const size_t numPoints = 1024 + 512;
	const size_t numComponents = 2;
	const size_t pointDim = 1;

	float sigma = 1;
	float det = 1;


	float logNormalizer = -0.5 * pointDim * logf(2.0 * M_PI) - 0.5 * logf(det);

	float mu0 = -1.5;
	float mu1 = +1.5;


	float X[numPoints];
	memset(X, 0, numPoints * sizeof(float));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i] = 3.0 * ( ( (float)i - (float)numPoints/2 ) / (float)(numPoints/2.0) );
	}

	float logP[numComponents * numPoints];
	memset(logP, 0, numComponents * numPoints * sizeof(float));

	struct Component phi;
	phi.sigmaL = &sigma;
	phi.normalizer = logNormalizer;

	phi.mu = &mu0;
	logMvNormDist(&phi, 1, X, numPoints, logP);

	phi.mu = &mu1;
	logMvNormDist(&phi, 1, X, numPoints, &logP[numPoints]);

	float logPi[numComponents];
	logPi[0] = logf(0.25);
	logPi[1] = logf(0.75);

	float actual = target(numPoints, numComponents, logPi, logP);
	assert(actual != -INFINITY);
	assert(actual != INFINITY);
	assert(actual == actual);

	// gpu impl overwrites logP with loggamma (logP - log p(x)), just run again
	phi.mu = &mu0;
	logMvNormDist(&phi, 1, X, numPoints, logP);

	phi.mu = &mu1;
	logMvNormDist(&phi, 1, X, numPoints, &logP[numPoints]);

	float expected = 0;
	for(size_t i = 0; i < numPoints; ++i) {
		float maxValue = -INFINITY;
		for(size_t k = 0; k < numComponents; ++k) {
			const float value = logPi[k] + logP[k * numPoints + i];
			if(maxValue < value) {
				maxValue = value;
			}
		}

		float sum = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			const float value = logPi[k] + logP[k * numPoints + i];
			sum += expf(value - maxValue);
		}

		expected += maxValue + logf(sum);
	}

	float absDiff = fabsf(expected - actual);
	if(absDiff >= DBL_EPSILON) {
		printf("log L = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
			actual, expected, absDiff);
	}

	assert(absDiff < DBL_EPSILON);
}

float cpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
) {
	float logL = 0;
	logLikelihood(logPi, numComponents, logP, numPoints, 0, numPoints, &logL);
	return logL;
}

float gpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
) {
	return gpuGmmLogLikelihood(numPoints, numComponents, logPi, logP);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogLikelihood(cpuGmmLogLikelihoodWrapper);
	test1DStandardNormalLogLikelihood(gpuGmmLogLikelihoodWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
