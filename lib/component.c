#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "component.h"
#include "linearAlgebra.h"
#include "util.h"

#ifndef PI
#define PI 3.141592653589793238
#endif

void printToConsole(const struct Component* component, const size_t pointDim) {
	if(component == NULL) {
		fprintf(stdout, "NULL\n");
		return;
	}

	fprintf(stdout, "{\n");

	fprintf(stdout, "\"pi\" : %.15f,\n", component->pi);

	fprintf(stdout, "\"mu\" : [ ");
	for (size_t dim = 0; dim < pointDim; ++dim) {
		fprintf(stdout, "%.15f ", component->mu[dim]);
		if(dim + 1 < pointDim) {
			fprintf(stdout, ", ");
		}
	}
	fprintf(stdout, "],\n");

	fprintf(stdout, "\"sigma\" : [ ");
	for (size_t dim = 0; dim < pointDim * pointDim; ++dim) {
		fprintf(stdout, "%.15f ", component->sigma[dim]);
		if(dim + 1 < pointDim * pointDim) {
			fprintf(stdout, ", ");
		}
	}
	fprintf(stdout, "]\n");
	fprintf(stdout, "}");
}

void prepareCovariance(struct Component* component, const size_t pointDim) {
	assert(component != NULL);

	// Perform cholesky factorization once each iteration instead of 
	// repeadily for each normDist execution.
	choleskyDecomposition(
		component->sigma, 
		pointDim, 
		component->sigmaL
	);

	// log det(Sigma) = log det(L L^T) = log det(L)^2 = 2 log prod L_{i,i} 
	//		  = 2 sum log L_{i,i}
	float logDet = 1.0;
	for (size_t i = 0; i < pointDim; ++i) {
		float diag = component->sigmaL[i * pointDim + i];
		assert(diag > 0);
		logDet += logf(diag);
	}

	logDet *= 2;

	component->normalizer = - 0.5 * (pointDim * logf(2.0 * PI) + logDet);
}

void logMvNormDist(
	const struct Component* component, const size_t pointDim,
	const float* X, const size_t numPoints, 
	float* P
) {
	// 2015-09-23 GEL Through profiling (Sleepy CS), found that program was 
	// spending most of its time in this method. Decided to change from 
	// processing single point at a time to processing set of points at a time. 
	// Found this gave 1.36x improvement (~6ms -> ~4ms) on the Old Faithful 
	// dataset. Total 1.77x improvement.

	// Here we are computing the probability density function of the  multivariate
	// normal distribution conditioned on a single component for the set of points 
	// given by X.
	//
	// P(x|component) = exp{ -0.5 * (x - mu)^T Sigma^{-} (x - mu) } / sqrt{ (2pi)^k det(Sigma) }
	//
	// Where Sigma and Mu are really Sigma_{component} Mu_{component}

	assert(component != NULL);
	assert(pointDim > 0);
	assert(X != NULL);
	assert(numPoints > 0);
	assert(P != NULL);

	float* XM = (float*)malloc(numPoints * pointDim * sizeof(float));
	float* SXM = (float*)malloc(numPoints * pointDim * sizeof(float));
	float* innerProduct = (float*)malloc(numPoints * sizeof(float));

	// Let XM = (x - m)
	for (size_t point = 0; point < numPoints; ++point) {
		for (size_t dim = 0; dim < pointDim; ++dim) {
			const size_t i = point * pointDim + dim;
			XM[i] = X[i] - component->mu[dim];
		}
	}

	// Sigma SXM = XM => Sigma^{-} XM = SXM
	solvePositiveDefinite(
		component->sigmaL, 
		XM, 
		SXM,
		pointDim, 
		numPoints 
		);

	// XM^T SXM
	memset(innerProduct, 0, numPoints * sizeof(float));
	for (size_t point = 0; point < numPoints; ++point) {
		for (size_t dim = 0; dim < pointDim; ++dim) {
			innerProduct[point] += XM[point * pointDim + dim] * SXM[point * pointDim + dim];
		}
	}

	// Compute P expf( -0.5 innerProduct ) / normalizer
	for (size_t point = 0; point < numPoints; ++point) {
		// Normalizer already has negative sign on it.
		P[point] = -0.5 * innerProduct[point] + component->normalizer;
		assert(P[point] == P[point]);
	}

	free(XM);
	free(SXM);
	free(innerProduct);
}

float sampleStandardNormal() {
	// Ratio-of-uniforms method
	// Computer Generation of Random Variables using the Ratio of Uniform Deviates
	// Kinderman, Monahan 1977

	static const float d = +0.857763884960707; // sqrtf(2.0 * expf(-1));

	float u1 = 0, v2 = 0, u2 = 0, x = 0;	
	do {
		u1 = rand() / (float)RAND_MAX;
		v2 = rand() / (float)RAND_MAX;
		u2 = (2.0 * v2 - 1.0) * d;
		x = u2 / u1;
	} while ( u1 * u1 > expf(-0.5 * x * x) );

	return x;
}

float* sampleWishart(const size_t dimension, const size_t degreeFreedom) {
	// Section 3, Wishart Distributions and Inverse-Wishart Sampling S. Sawyer

	// A numerical procedure to generate a sample covariance matrix
	// Odell, Feiveson, 1966
	assert(dimension > 0);
	assert(degreeFreedom > dimension + 1);

	float V[dimension];
	for(size_t i = 0; i < dimension; ++i) {
		// V_{i} is sampled from a chi-square distribution with n - i + 1 degrees of 
		// freedom.
		V[i] = 0;
		for(size_t j = 0; j <= degreeFreedom - i; ++j) {
			const float x = sampleStandardNormal();
			V[i] += x * x;
		}
	}

	float N[dimension * dimension];
	for(size_t i = 0; i < dimension; ++i) {
		const size_t ii = i * dimension + i;
		N[ii] = 0;

		for(size_t j = 0; j < i; ++j) {
			// N_{i,j} sampled from a standard normal distribution
			const size_t ij = i * dimension + j;
			const size_t ji = j * dimension + i;
			N[ij] = sampleStandardNormal();
			N[ji] = N[ij];
		}
	}

	float* W = (float*)checkedCalloc(dimension*dimension, sizeof(float));
	W[0] = V[0];
	for(size_t i = 1; i < dimension; ++i) {
		const size_t ii = i * dimension + i;
		for(size_t r = 0; r <= i - 1; ++r) {
			const size_t ri = r * dimension + i;
			const float nri = N[ri];
			W[ii] += nri * nri;
		}
	}

	for(size_t i = 0; i < dimension; ++i) {
		assert(V[i] > 0);
		V[i] = sqrtf(V[i]);
	}

	for(size_t i = 1; i < dimension; ++i) {
		for(size_t j = 0; j < i; ++j) {
			const size_t ij = i * dimension + j;
			const size_t ji = j * dimension + i;

			W[ij] = N[ij] * V[i];

			// TODO: Figure out how Odell/Feiveson factored this out to ensure O(d^2).
			if(i > 0) {
				for(size_t r = 0; r <= i - 1; ++r) {
					const size_t ri = r * dimension + i;
					const size_t rj = r * dimension + j;
					W[ij] += N[ri] * N[rj];
				}
			}

			W[ji] = W[ij];
		}
	}

	return W;
}

float* sampleWishartCholesky(const size_t dimension, const size_t degreeFreedom) {
	// Eqn (3.2) Wishart Distributions and Inverse-Wishart Sampling S. Sawyer

	assert(dimension > 0);
	assert(degreeFreedom > dimension + 1);

	float V[dimension];
	for(size_t i = 0; i < dimension; ++i) {
		// V_{i} is sampled from a chi-square distribution with n - i + 1 degrees of 
		// freedom.
		V[i] = 0;
		for(size_t j = 0; j <= degreeFreedom - i; ++j) {
			const float x = sampleStandardNormal();
			V[i] += x * x;
		}
	}

	float* L = (float*)checkedCalloc(dimension * dimension, sizeof(float));
	for(size_t i = 0; i < dimension; ++i) {
		const size_t ii = i * dimension + i;
		L[ii] = sqrtf(V[i]);

		for(size_t j = 0; j < i; ++j) {
			// N_{i,j} sampled from a standard normal distribution
			const size_t ij = i * dimension + j;
			L[ij] = sampleStandardNormal();
		}
	}

	return L;
}
