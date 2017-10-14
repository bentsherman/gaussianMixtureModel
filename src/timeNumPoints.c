#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gmm.h"
#include "seqGmm.h"
#include "parallelGmm.h"
#include "util.h"
#include "cudaGmm.h"

int main(int argc, char** argv) {
	srand(time(NULL));

	const size_t minNumPoints = 128;
	const size_t maxNumPoints = 512 * 1024;

	const size_t numComponents = 8;
	const size_t pointDim = 2;

	const size_t numSamples = 1;

	struct timeval start, end;

	const size_t maxIterations = 20;

	fprintf(stdout, "#numPoints numComponents pointDim seqElapsedSec parallelElapsedSec cudaElapsedSec\n");
	for(size_t numPoints = minNumPoints; numPoints <= maxNumPoints; numPoints *= 2) {
		for(size_t sample = 0; sample < numSamples; ++sample) {	
			float* X = generateGmmData(numPoints, pointDim, numComponents);

			gettimeofday(&start, NULL);
			freeGMM(fit(X, numPoints, pointDim, numComponents, maxIterations));
			gettimeofday(&end, NULL);
			float seqElapsedSec = calcElapsedSec(&start, &end);

			gettimeofday(&start, NULL);
			freeGMM(parallelFit(X, numPoints, pointDim, numComponents, maxIterations));
			gettimeofday(&end, NULL);
			float parallelElapsedSec = calcElapsedSec(&start, &end);

			gettimeofday(&start, NULL);
			freeGMM(cudaFit(X, numPoints, pointDim, numComponents, maxIterations));
			gettimeofday(&end, NULL);
			float gpuElapsedSec = calcElapsedSec(&start, &end);

			fprintf(stdout, "%zu %zu %zu %.7f %.7f %.7f\n", 
				numPoints, numComponents, pointDim, seqElapsedSec, parallelElapsedSec, gpuElapsedSec);

			free(X);
		}
	}

	return EXIT_SUCCESS;
}
