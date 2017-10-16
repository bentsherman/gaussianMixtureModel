#ifndef GMM_H
#define GMM_H

#include <stdlib.h>

#include "component.h"

typedef struct {
	// Dimension of the data (each data point is a vector \in R^{pointDim})
	size_t pointDim;

	// The individual components that constitute the model
	size_t numComponents;
	Component* components;

	// The final log-likelihood of the model
	float logL;
} GMM;

GMM* initGMM(
	const float* X,
	const size_t numPoints,
	const size_t pointDim,
	const size_t numComponents
);

void freeGMM(GMM* gmm);

void calcLogMvNorm(
	const Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	const float* X, const size_t numPoints, const size_t pointDim,
	float* logProb
);

void logLikelihood(
	const float* logpi, const size_t numComponents,
	const float* logProb, const size_t numPoints,
	const size_t pointStart, const size_t pointEnd,
	float* logL
);

int shouldContinue(
	const float prevLogL, const float currentLogL,
	const float tolerance
);

void calcLogGammaNK(
	const float* logpi, const size_t numComponents,
	const size_t pointStart, const size_t pointEnd,
	float* loggamma, const size_t numPoints
);

void logLikelihoodAndGammaNK(
	const float* logpi, const size_t numComponents,
	float* logProb, const size_t numPoints,
	const size_t pointStart, const size_t pointEnd,
	float* logL
);

void calcLogGammaK(
	const float* loggamma, const size_t numPoints,
	const size_t componentStart, const size_t componentEnd,
	float* logGamma, const size_t numComponents
);

float calcLogGammaSum(
	const float* logpi, const size_t numComponents,
	const float* logGamma
);

void performMStep(
	Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	float* logpi, float* loggamma, float* logGamma, const float logGammaSum,
	const float* X, const size_t numPoints, const size_t pointDim,
	float* outerProduct, float* xm
);

float* generateGmmData(
	const size_t numPoints, const size_t pointDim, const size_t numComponents
);

void printGmmToConsole(
	GMM* gmm
);

#endif
