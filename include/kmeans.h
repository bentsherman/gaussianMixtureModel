#ifndef KMEANS_H
#define KMEANS_H

void kmeans(
	const float* X, const size_t numPoints, const size_t pointDim, 
	float* M, const size_t numComponents
); 

#endif
