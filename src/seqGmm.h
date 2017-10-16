#ifndef SEQGMM_H
#define SEQGMM_H

#include <stdlib.h>

#include "gmm.h"

GMM* fit(
	const float* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents,
	const size_t maxIterations
);

#endif 
