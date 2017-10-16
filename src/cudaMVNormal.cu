#include <assert.h>

#include "cudaMVNormal.hu"

/*
 * Computes \sum_{i}^{N} x_i y_i for x, y \in \mathbb{R}^{N}.
 */
__device__ float devVecDot(const size_t N, const float* x, const float* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);
	// x == y allowed

	float sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

/*
 * Computes z_{i} \gets x_{i} - y_{i} for x, y \in \mathbb{R}^N.
 */
__device__ void devVecMinus(const size_t N, float* z, const float* x, const float* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);
	// x == y allowed

	for(size_t i = 0; i < N; ++i) {
		z[i] = x[i] - y[i];
	}
}

/*
 * Solves the lower triangular system L^T x = b for x, b \in \mathbb{R}^{N}, 
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTri(const size_t N, const float* L, float* x, const float* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

	for(size_t i = 0; i < N; ++i) {
		float sum = 0.0;
		if(i > 0) {
			for(size_t j = 0; j <= i - 1; ++j) {
				sum += L[i * N + j] * x[j];
			}
		}

		x[i] = (b[i] - sum) / L[i * N + i];
	}
}

/*
 * Solves the upper triangular system L^T x = b for x, b \in \mathbb{R}^{N}, 
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTriT(const size_t N, const float* L, float* x, const float* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

	// treat L as an upper triangular matrix U
	for(size_t i = 0; i < N; i++) {
		size_t ip = N - 1 - i;
		float sum = 0;
		for(size_t j = ip + 1; j < N; ++j) {
			sum += L[j * N + ip] * x[j];
		}

		x[ip] = (b[ip] - sum) / L[ip * N + ip];
	}
}


/*
 *
 */
__device__ float devLogMVNormNormalizer(
	const size_t pointDim,
	const float* sigmaL
) {
	float det = 1.0;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i];
	}
	det *= det;

	return -0.5 * logf( 2.0 * M_PI ) * pointDim - 0.5 * logf(det);
}

/*
 * Computes logf( p(x | mu, Sigma ) ) for multivariate normal distribution with 
 * parameters mu (mean), and Sigma (covariance).
 */
__device__ float devLogMVNormDist(
	const size_t pointDim,
	const float* x, const float* mu, const float* sigmaL,
	float* u, float* v
) {
	devVecMinus(pointDim, v, x, mu); // v <- x - mu
	devSolveLowerTri(pointDim, sigmaL, u, v); // u <- u s.t. L u = (x - mu)
	devSolveLowerTriT(pointDim, sigmaL, u, u); // u <- v s.t. L^T v = u
	return devLogMVNormNormalizer(pointDim, sigmaL) - 0.5 * devVecDot(pointDim, u, v);
}

__global__ void kernLogMVNormDist(
	const size_t numPoints, const size_t pointDim, 
	const float* X, float* mu, float* sigmaL,
	float* logProb
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	float u[1024];
	float v[1024];

	logProb[i] = devLogMVNormDist(
		pointDim, 
		& X[i * pointDim], mu, sigmaL,
		u, v
	);
}

