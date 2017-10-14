#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include <stdlib.h>

void choleskyDecomposition(
	const float* A, const size_t pointDim, 
	float* L
);

void solvePositiveDefinite(
	const float* L, const float* B, 
	float* X, 
	const size_t pointDim, const size_t numPoints
);

void lowerDiagByVector(
	const float* L,
	const float* x,
	float* b,
	const size_t n
);

void vectorAdd(
	const float* a,
	const float* b,
	float* c,
	const size_t n
); 

void vecAddInPlace(
	float* a, 
	const float* b, 
	const size_t D
);

void vecDivByScalar(
	float* a, 
	const float b, 
	const size_t D
);

float vecDiffNorm(
	const float* a, 
	const float* b, 
	const size_t D
);

#endif
