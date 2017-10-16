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

typedef void (*CalcLogGammaNKWrapper)(const size_t, const size_t, const float*, float*);

void test1DStandardNormalLogGammaNK(int gnkIncPik, CalcLogGammaNKWrapper target) {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;
	const size_t numComponents = 1;

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

	struct Component phi;
	phi.mu = mu;
	phi.sigmaL = sigmaL;
	phi.normalizer = logNormalizer;
	logMvNormDist(&phi, pointDim, X, numPoints, logP);
	
	float logPi[numComponents];
	float uniformPi = 1.0 / (float)numComponents;
	for(size_t k = 0; k < numComponents; ++k) {
		logPi[k] = logf(uniformPi);
	}

	float loggamma[numPoints];
	memcpy(loggamma, logP, numPoints * sizeof(float));

	target(numPoints, numComponents, logPi, loggamma);

	for(size_t i = 0; i < numPoints; ++i) {
		float sum = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			sum += logPi[k] + logP[k * numPoints + i];
		}

		for(size_t k = 0; k < numComponents; ++k) {
			float actual = loggamma[k * numPoints + i];
			assert(actual != -INFINITY);
			assert(actual != INFINITY);
			assert(actual == actual);

			float expected = logP[k * numPoints + i] - sum;
			if(gnkIncPik) {
				expected += logPi[k];
			}

			float absDiff = fabsf(expected - actual);
			if(absDiff >= FLT_EPSILON) {
				printf("gamma_{n = %zu, k = %zu} = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
					i, k, actual, expected, absDiff);
			}

			assert(absDiff < FLT_EPSILON);
		}
	}
}

void cpuCalcLogGammaNKWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
) {
	calcLogGammaNK(logPi, numComponents, 0, numPoints, logP, numPoints);
}

void gpuCalcLogGammaNKWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
) {
	gpuCalcLogGammaNK(numPoints, numComponents, logPi, logP);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogGammaNK(1, cpuCalcLogGammaNKWrapper);
	test1DStandardNormalLogGammaNK(1, gpuCalcLogGammaNKWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
