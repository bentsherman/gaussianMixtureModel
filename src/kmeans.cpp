#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kmeans.h"
#include "linearAlgebra.h"

void kmeans(const float* X, const size_t numPoints, const size_t pointDim, float* M, const size_t numComponents) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(M != NULL);
	assert(numComponents > 0);

	const float tolerance = 1e-3;
	float diff = 0;

	const size_t maxIterations = 20;

	float MP[numComponents * pointDim];
	size_t counts[numComponents];

	for(size_t iteration = 0; iteration < maxIterations && diff > tolerance; ++iteration) {
		memset(MP, 0, numComponents * pointDim * sizeof(float));	
		memset(counts, 0, numComponents * sizeof(size_t));	

		for(size_t i = 0; i < numPoints; ++i) {
			const float* Xi = & X[i * pointDim];

			// arg min
			float minD = INFINITY;
			size_t minDk = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				const float* Mk = & M[k * pointDim];
				float dist = vecDiffNorm(Xi, Mk, pointDim);
				if(minD > dist) {
					minD = dist;
					minDk = k;
				}	
			}

			vecAddInPlace(&M[minDk * pointDim], Xi, pointDim);
			++counts[minDk];
		}

		for(size_t k = 0; k < numComponents; ++k) {
			vecDivByScalar(&MP[k * pointDim], counts[k], pointDim);
		}

		diff = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			diff += vecDiffNorm(&MP[k * pointDim], &M[k * pointDim], pointDim);
		}
		diff /= (float) numComponents;

		memcpy(M, MP, numComponents * pointDim * sizeof(float));
	}
}

