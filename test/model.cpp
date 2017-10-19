#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "datFile.h"
#include "clusteringmodel.h"
#include "util.h"

void usage(const char* programName) {
	assert(programName != NULL);
	fprintf(stdout, "%s <train.dat> <min-k> <max-k>\n", programName);
}

void print_vector(int* v, int n)
{
	for ( int i = 0; i < n; i++ ) {
		fprintf(stdout, "%d", v[i]);
	}
	fprintf(stdout, "\n");
}

int main(int argc, char** argv) {
	if(argc != 4) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t min_k = atoi(argv[2]);
	size_t max_k = atoi(argv[3]);

	size_t numPoints = 0, pointDim = 0;
	float* data = parseDatFile(argv[1], &numPoints, &pointDim);
	if(data == NULL) {
		return EXIT_FAILURE;
	}

	if(numPoints < max_k) {
		fprintf(stdout, "Number of components should be less than or equal to number of points.\n");
		free(data);
		return EXIT_FAILURE;
	}

	std::vector<int> components;
	for ( size_t k = min_k; k <= max_k; k++ ) {
		components.push_back(k);
	}

	ClusteringModel model(components);

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	GMM *gmm = model.run(data, numPoints, pointDim);

	gettimeofday(&stop, NULL);
	float elapsedSec = calcElapsedSec(&start, &stop);

	printGmmToConsole(gmm);

	fprintf(stdout, "\n");
	print_vector(gmm->y_pred, numPoints);
	fprintf(stdout, "\n");

	fprintf(stdout, "{\n");
	fprintf(stdout, "\"file\": \"%s\",\n", argv[1]);
	fprintf(stdout, "\"elapsedSec\": %.6f,\n", elapsedSec);
	fprintf(stdout, "}\n");

	free(data);
	freeGMM(gmm);

	return EXIT_SUCCESS;
}
