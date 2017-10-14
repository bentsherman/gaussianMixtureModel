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

typedef void (*CalcLogGammaKWrapper)(const size_t, const size_t, const float*, float*);

void test1DStandardNormalLogGammaK(CalcLogGammaKWrapper target) {
	const size_t numPoints = 1024;
	const size_t numComponents = 5;

	float loggamma[numPoints * numComponents];
	for(size_t k = 0; k < numComponents; ++k) {
		float* loggammak = & loggamma[k * numPoints];
		for(size_t i = 0; i < numPoints; ++i) {
			loggammak[i] = -logf((float)numPoints);
		}
	}

	float logGamma[numComponents];
	memset(logGamma, 0, numComponents * sizeof(float));

	target(numPoints, numComponents, loggamma, logGamma);

	float expected = 0.0;
	for(size_t k = 0; k < numComponents; ++k) {
		float actual = logGamma[k];
		float absDiff = fabsf(expected - actual);
		if(absDiff >= DBL_EPSILON) {
			printf("Gamma_{k = %zu} = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
				k, actual, expected, absDiff);
		}

		assert(absDiff < DBL_EPSILON);
	}
}

void cpuCalcLogGammaKWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* loggamma, float* logGamma
) {
	calcLogGammaK(loggamma, numPoints, 0, numComponents, logGamma, numComponents);
}

void gpuCalcLogGammaKWrapper(
	const size_t numPoints, const size_t numComponents,
	const float* loggamma, float* logGamma
) {
	gpuCalcLogGammaK(numPoints, numComponents, loggamma, logGamma);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogGammaK(cpuCalcLogGammaKWrapper);
	test1DStandardNormalLogGammaK(gpuCalcLogGammaKWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
