#include <cmath>
#include <cstdio>
#include "clusteringmodel.h"
#include "seqGmm.h"

ClusteringModel::ClusteringModel(const std::vector<int>& components)
{
	this->_components = components;
}

float BIC(GMM *gmm, int n, int d)
{
	int k = gmm->numComponents;
	int p = k * (1 + d + d * d);

	return logf(n) * p - 2 * gmm->logL;
}

GMM * ClusteringModel::run(const float *X, int n, int d)
{
	// run each clustering model
	std::vector<GMM *> models;

	for ( int k : _components ) {
		GMM *gmm = fit(X, n, d, k, 100);

		models.push_back(gmm);
	}

	// find the model with the lowest BIC value
	float min_criterion = 0;
	GMM *min_model = nullptr;

	for ( GMM *gmm : models ) {
		float criterion = BIC(gmm, n, d);

		printGmmToConsole(gmm);
		fprintf(stdout, "BIC: %f\n", criterion);


		if ( min_model == nullptr || criterion < min_criterion ) {
			min_criterion = criterion;
			min_model = gmm;
		}
	}

	// cleanup unused models
	for ( GMM *gmm : models ) {
		if ( gmm != min_model ) {
			freeGMM(gmm);
		}
	}

	return min_model;
}
