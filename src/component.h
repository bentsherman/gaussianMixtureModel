#ifndef COMPONENT_H
#define COMPONENT_H

#include <stdlib.h>

typedef struct {
	// Parameters, weight, mean, covariance
	float pi;
	float* mu;
	float* sigma;

	// Lower triangular covariance matrix
	float* sigmaL;

	// Probability density normalizer
	float normalizer;
} Component;

void printToConsole(
	const Component* component,
	const size_t pointDim
);

void prepareCovariance(
	Component* component,
	const size_t pointDim
);

void logMvNormDist(
	const Component* component, const size_t pointDim,
	const float* X, const size_t numPoints,
	float* logProb
);

float sampleStandardNormal();

float* sampleWishart(
	const size_t dimension, const size_t degreeFreedom
);

float* sampleWishartCholesky(
	const size_t dimension, const size_t degreeFreedom
);

#endif
