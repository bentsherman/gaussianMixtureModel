#ifndef COMPONENT_H
#define COMPONENT_H

#include <stdlib.h>

struct Component {
	// Parameters, weight, mean, covariance
	float pi;
	float* mu;
	float* sigma;

	// Lower triangular covariance matrix
	float* sigmaL;

	// Probability density normalizer
	float normalizer;
};

void printToConsole(
	const struct Component* component,
	const size_t pointDim
);

void prepareCovariance(
	struct Component* component, 
	const size_t pointDim
); 

void logMvNormDist(
	const struct Component* component, const size_t pointDim, 
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
