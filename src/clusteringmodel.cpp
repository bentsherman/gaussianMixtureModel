#include <cmath>
#include <cstdio>
#include "clusteringmodel.h"
#include "seqGmm.h"

ClusteringModel::ClusteringModel(const std::vector<int>& components)
{
	this->_components = components;
}

float BIC(GMM *model, int n, int d)
{
	int k = model->numComponents;
	int p = k * (1 + d + d * d);

	return logf(n) * p - 2 * model->logL;
}

GMM * ClusteringModel::run(const float *X, int n, int d)
{
	// run each clustering model
	std::vector<GMM *> models;

	for ( int k : _components ) {
		GMM *model = fit(X, n, d, k, 100);

		models.push_back(model);
	}

	// evaluate each model with BIC
	float min_criterion = 0;
	GMM *min_model = nullptr;

	for ( GMM *model : models ) {
		float criterion = BIC(model, n, d);

		printGmmToConsole(model);
		fprintf(stdout, "BIC: %f\n", criterion);

		if ( min_model == nullptr || criterion < min_criterion ) {
			min_criterion = criterion;
			min_model = model;
		}
	}

	return min_model;
}
