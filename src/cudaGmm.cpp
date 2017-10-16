#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gmm.h"
#include "cudaGmm.h"
#include "util.h"

#include "cudaWrappers.h"

GMM* cudaFit(
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
	
	GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	float* pi = (float*) malloc(numComponents * sizeof(float));
	float* Mu = (float*) malloc(numComponents * pointDim * sizeof(float));
	float* Sigma = (float*) malloc(numComponents * pointDim * pointDim * sizeof(float));	
	float* SigmaL = (float*) malloc(numComponents * pointDim * pointDim * sizeof(float));
	float* normalizers = (float*) malloc(numComponents * sizeof(float));

	for(size_t k = 0; k < numComponents; ++k) {
		Component* c = & gmm->components[k];

		pi[k] = c->pi;
		memcpy(&Mu[k * pointDim], c->mu, pointDim * sizeof(float));
		memcpy(&Sigma[k * pointDim * pointDim], c->sigma, pointDim * pointDim * sizeof(float));
		memcpy(&SigmaL[k * pointDim * pointDim], c->sigmaL, pointDim * pointDim * sizeof(float));
		normalizers[k] = c->normalizer;
	}

	gpuGmmFit(
		X,
		numPoints, pointDim, numComponents,
		pi, Mu, Sigma,
		SigmaL, normalizers,
		maxIterations
	);

	for(size_t k = 0; k < numComponents; ++k) {
		Component* c = & gmm->components[k];

		c->pi = pi[k];
		memcpy(c->mu, &Mu[k * pointDim], pointDim * sizeof(float));
		memcpy(c->sigma, &Sigma[k * pointDim * pointDim], pointDim * pointDim * sizeof(float));
		memcpy(c->sigmaL, &SigmaL[k * pointDim * pointDim], pointDim * pointDim * sizeof(float));
		c->normalizer = normalizers[k];
	}

	free(normalizers);
	free(SigmaL);
	free(Sigma);
	free(Mu);
	free(pi);

	return gmm;
}
