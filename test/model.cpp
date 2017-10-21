#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "datFile.h"
#include "clusteringmodel.h"
#include "util.h"

void usage(const char* programName) {
	assert(programName != NULL);
	fprintf(stderr, "usage: %s --data [datfile] --min-k [N] --max-k [N] [--gpu]\n", programName);
}

void print_vector(int* v, int n)
{
	for ( int i = 0; i < n; i++ ) {
		fprintf(stdout, "%d", v[i]);
	}
	fprintf(stdout, "\n");
}

int main(int argc, char** argv) {
	const char *filename = nullptr;
	size_t min_k = 1;
	size_t max_k = 5;
	bool gpu = false;

	for ( int i = 1; i < argc; i++ ) {
		if ( strcmp(argv[i], "--data") == 0 ) {
			filename = argv[i + 1];
			i++;
		}
		else if ( strcmp(argv[i], "--min-k") == 0 ) {
			min_k = atoi(argv[i + 1]);
			i++;
		}
		else if ( strcmp(argv[i], "--max-k") == 0 ) {
			max_k = atoi(argv[i + 1]);
			i++;
		}
		else if ( strcmp(argv[i], "--gpu") == 0 ) {
			gpu = true;
		}
		else {
			fprintf(stderr, "error: unknown option \"%s\"\n", argv[i]);
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	if ( filename == nullptr ) {
		fprintf(stderr, "error: data file is required\n");
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numPoints = 0, pointDim = 0;
	float* data = parseDatFile(filename, &numPoints, &pointDim);
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

	GMM *gmm = model.run(data, numPoints, pointDim, gpu);

	gettimeofday(&stop, NULL);
	float elapsedSec = calcElapsedSec(&start, &stop);

	printGmmToConsole(gmm);

	fprintf(stdout, "\n");
	print_vector(gmm->y_pred, numPoints);
	fprintf(stdout, "\n");

	fprintf(stdout, "{\n");
	fprintf(stdout, "\"file\": \"%s\",\n", filename);
	fprintf(stdout, "\"elapsedSec\": %.6f,\n", elapsedSec);
	fprintf(stdout, "}\n");

	free(data);
	freeGMM(gmm);

	return EXIT_SUCCESS;
}
