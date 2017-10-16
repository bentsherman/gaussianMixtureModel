#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <stdlib.h>

void gpuGmmFit(
	const float* X,
	const size_t numPoints,
	const size_t pointDim,
	const size_t numComponents,
	float* pi,
	float* Mu,
	float* Sigma,
	float* SigmaL,
	float* normalizers,
	const size_t maxIterations
);

// Wrappers for unit testing
void gpuSum(
	size_t numPoints, size_t pointDim,
	float* host_a, float* host_sum
);

float gpuMax(
	const size_t N, float* a
);

void gpuLogMVNormDist(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* mu, const float* sigmaL,
	float* logP
);

float gpuGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const float* logPi, float* logP
);

void gpuCalcLogGammaNK(
	const size_t numPoints, const size_t numComponents,
	const float* logpi, float* loggamma
);

void gpuCalcLogGammaK(
	const size_t numPoints, const size_t numComponents,
	const float* loggamma, float* logGamma
);

#endif
