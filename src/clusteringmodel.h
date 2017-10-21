#ifndef CLUSTERINGMODEL_H
#define CLUSTERINGMODEL_H

#include <vector>
#include "gmm.h"

class ClusteringModel {
private:
	std::vector<int> _components;

public:
	ClusteringModel(const std::vector<int>& components);
	~ClusteringModel() {};

	GMM * run(const float *X, int n, int d, bool gpu);
};

#endif
