#ifndef PARALLELGMM_H
#define PARALLELGMM_H

#include <stdlib.h>

#include "barrier.h"
#include "gmm.h"

// These are resources accessed by all threads.
typedef struct {
	const float* X;
	size_t numPoints;
	size_t pointDim;

	GMM* gmm;

	float tolerance;
	size_t maxIterations;
	float prevLogL;
	float currentLogL;

	int shouldContinue;

	float* logpi;
	float* loggamma;
	float* logGamma;
	float logGammaSum;

	float* logLK;
	size_t numProcesses;

	Barrier* barrier;
} SharedThreadStartArgs;

// These are resources assigned to a single thread.
typedef struct {
	size_t id;
	SharedThreadStartArgs* shared;

	// These limits apply only to computing gamma
	size_t pointStart;
	size_t pointEnd; // i = pointStart; i < pointEnd; ++i

	// These limits apply for all components
	size_t componentStart;
	size_t componentEnd; // i = componentStart; i < componentEnd; ++i
} ThreadStartArgs;

// Barrier critical section callbacks
void checkStopCriteria(
	void* untypedArgs
);

void computeGammaSum(
	void* untypedArgs
);

// Thread start
void* parallelFitStart(
	void* untypedArgs
);

GMM* parallelFit(
	const float* X,
	const size_t numPoints,
	const size_t pointDim,
	const size_t numComponents,
	const size_t maxIterations
);

#endif
