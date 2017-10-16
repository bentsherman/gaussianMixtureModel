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

	const float sigma = 1;
	const float det = 1;
	const float logNormalizer = -0.5 * pointDim * logf(2.0 * M_PI) - 0.5 * logf(det);

	const float logPi[] = { logf(0.25), logf(0.75) };
	const float mu[] = { -1.5, 1.5 };

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
	for(size_t k = 0; k < numComponents; ++k) {
		phi.mu = &mu[k];
		logMvNormDist(&phi, 1, X, numPoints, & logP[k * numPoints]);
	}

	float actualLogL = target(numPoints, numComponents, logPi, logP);

	// Verify the logL portion
	{
		assert(actualLogL != -INFINITY);
		assert(actualLogL != INFINITY);
		assert(actualLogL == actualLogL);

		float expectedLogL = 0;
		for(size_t i = 0; i < numPoints; ++i) {
			float sum = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				sum += expf(logPi[k] + logNormalizer - 0.5 * powf( X[i] - mu[k], 2.0 )); 
			}

			expectedLogL += logf(sum);
		}

		float absDiff = fabsf(expectedLogL - actualLogL);
		if(absDiff >= FLT_EPSILON) {
			printf("log L = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
				actualLogL, expectedLogL, absDiff);
		}

		assert(absDiff < FLT_EPSILON);
	}

	// Verify the gammaNK portion
	{
		for(size_t i = 0; i < numPoints; ++i) { 
			float sum = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				sum += expf(logPi[k] + logNormalizer - 0.5 * powf( X[i] - mu[k], 2.0 )); 
			}
			float logPx = logf(sum);

			for(size_t k = 0; k < numComponents; ++k) {
				float expectedGammaNK = logNormalizer - 0.5 * powf(X[i] - mu[k], 2.0) - logPx;
				float actualGammaNK = logP[k * numPoints + i];

				float absDiff = fabsf(expectedGammaNK - actualGammaNK);
				if(absDiff >= 10.0 * FLT_EPSILON) {
					printf("gamma_{n = %zu, k = %zu} = %.16f, but should equal = %.16f; absDiff = %.16f, epsilon = %.16f\n", 
						i, k, actualGammaNK, expectedGammaNK, absDiff, 10.0 * FLT_EPSILON);
				}

				assert(absDiff < 10.0 * FLT_EPSILON);
			}
		}
	}
}

float cpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
) {
	float logL = 0;
	logLikelihoodAndGammaNK(logPi, numComponents, logP, numPoints, 0, numPoints, &logL);
	return logL;
}

float gpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
) {
	// does both loglikelihood and gamma nk
	return gpuGmmLogLikelihood(numPoints, numComponents, logPi, logP);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogLikelihood(cpuGmmLogLikelihoodWrapper);
	test1DStandardNormalLogLikelihood(gpuGmmLogLikelihoodWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
