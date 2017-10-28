#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCommon.hu"
#include "cudaFolds.hu"
#include "cudaGmm.hu"

__global__ void kernCalcLogLikelihoodAndGammaNK(
	const size_t numPoints, const size_t numComponents,
	const float* logpi, float* logPx, float* loggamma
) {
	// loggamma[k * numPoints + i] =
	// On Entry: log p(x_i | mu_k, Sigma_k)
	// On exit: [log pi_k] + [log p(x_i | mu_k, sigma_k)] - [log p(x_i)]

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	float maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const float logProbK = logpi[k] + loggamma[k * numPoints + i];
		if(logProbK > maxArg) {
			maxArg = logProbK;
		}
	}

	float sum = 0.0;
	for (size_t k = 0; k < numComponents; ++k) {
		const float logProbK = logpi[k] + loggamma[k * numPoints + i];
		sum += expf(logProbK - maxArg);
	}

	assert(sum >= 0);
	const float logpx = maxArg + logf(sum);

	for(size_t k = 0; k < numComponents; ++k) {
		loggamma[k * numPoints + i] += -logpx;
	}

	logPx[i] = logpx;
}

__host__ float cudaGmmLogLikelihoodAndGammaNK(
	cudaDeviceProp* deviceProp,
	const size_t numPoints, const size_t numComponents,
	const float* logpi, float* logP,
	const float* device_logpi, float* device_logP
) {
	// logpi: 1 x numComponents
	// logP: numComponents x numPoints

	dim3 grid, block;
	calcDim(numPoints, deviceProp, &block, &grid);

	float logL = 0;
	float* device_logPx = mallocOnGpu(numPoints);

	kernCalcLogLikelihoodAndGammaNK<<<grid, block>>>(
		numPoints, numComponents,
		device_logpi, device_logPx, device_logP
	);

	cudaArraySum(
		deviceProp,
		numPoints, 1,
		device_logPx
	);

	check(cudaMemcpy(&logL, device_logPx, sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(device_logPx);

	// Copy back the full numPoints * numComponents values
	check(cudaMemcpy(logP, device_logP,
		numPoints * numComponents * sizeof(float), cudaMemcpyDeviceToHost));

	return logL;
}

__global__ void kernExp(float* A, float* bias) {
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	A[i] = expf(A[i] - *bias);
}

__global__ void kernBiasAndLog(float* sumexp, float* bias) {
	*sumexp = *bias + logf(*sumexp);
}

__host__ void cudaLogSumExp(
	cudaDeviceProp* deviceProp, dim3 grid, dim3 block,
	const size_t numPoints,
	float* device_src, float* device_dest,
	float* device_working,
	cudaStream_t stream
) {
	// dest <- src
	check(cudaMemcpyAsync(
		device_dest, device_src,
		numPoints * sizeof(float),
		cudaMemcpyDeviceToDevice,
		stream
	));

	// working <- src
	check(cudaMemcpyAsync(
		device_working, device_src,
		numPoints * sizeof(float),
		cudaMemcpyDeviceToDevice,
		stream
	));

	// working <- max { src }
	cudaArrayMax(deviceProp, numPoints, device_working, stream);

	// dest <- expf(src - max)
	kernExp<<<grid, block, 0, stream>>>(
		device_dest,
		device_working
	);

	// dest <- sum expf(src - max)
	cudaArraySum(deviceProp, numPoints, 1, device_dest, stream);

	// dest <- max + log sum expf(src - max)
	kernBiasAndLog<<<1, 1, 0, stream>>>(
		device_dest, device_working
	);
}

__global__ void kernCalcMu(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* loggamma, const float* GammaK,
	float* dest
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	const float a = expf(loggamma[i]) / expf(*GammaK);
	const float* x = & X[i * pointDim];
	float* y = & dest[i * pointDim];

	for(size_t i = 0; i < pointDim; ++i) {
		y[i] = a * x[i];
	}
}

__global__ void kernCalcSigma(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* mu, const float* loggamma, const float* GammaK,
	float* dest
) {
	assert(pointDim < 1024);

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	// gamma_{n,k} / Gamma_{k} (x - mu) (x - mu)^T

	const float a = expf(loggamma[i]) / expf(*GammaK);
	const float* x = & X[i * pointDim];
	float* y = & dest[i * pointDim * pointDim];

	float u[1024];
	for(size_t i = 0; i < pointDim; ++i) {
		u[i] = x[i] - mu[i];
	}

	for(size_t i = 0; i < pointDim; ++i) {
		float* yp = &y[i * pointDim];
		for(size_t j = 0; j < pointDim; ++j) {
			yp[j] = a * u[i] * u[j];
		}
	}
}

__global__ void kernUpdatePi(
	const size_t numPoints, const size_t numComponents,
	float* logpi, float* Gamma
) {
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int comp = b * blockDim.x + threadIdx.x;
	if(comp > numComponents) {
		return;
	}

	__shared__ float A[1024];
	A[comp] = logpi[comp] + logf(Gamma[comp * numPoints]);
	__syncthreads();

	float sum = 0;
	for(size_t k = 0; k < numComponents; ++k) {
		sum += expf(A[k]);
	}

	logpi[comp] = A[comp] - logf(sum);
}

__global__ void kernPrepareCovariances(
	const size_t numComponents, const size_t pointDim,
	float* Sigma, float* SigmaL,
	float* normalizers,
	int *error
) {
	// Parallel in the number of components

	// Sigma: numComponents x pointDim * pointDim
	// SigmaL: numComponents x pointDim * pointDim
	// normalizers: 1 x numComponents

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int comp = b * blockDim.x + threadIdx.x;
	if(comp > numComponents) {
		return;
	}

	// L is the resulting lower diagonal portion of A = LL^T
	const size_t ALen = pointDim * pointDim;
	float* A = & Sigma[comp * ALen];
	float* L = & SigmaL[comp * ALen];
	for(size_t i = 0; i < ALen; ++i) {
		L[i] = 0;
	}

	for (size_t k = 0; k < pointDim; ++k) {
		float sum = 0;
		for (int s = 0; s < k; ++s) {
			const float l = L[k * pointDim + s];
			const float ll = l * l;
			sum += ll;
		}

		assert(sum >= 0);

		sum = A[k * pointDim + k] - sum;
		if (sum <= FLT_EPSILON) {
			*error = 1;
			return;
		}

		L[k * pointDim + k] = sqrtf(sum);
		for (int i = k + 1; i < pointDim; ++i) {
			float subsum = 0;
			for (int s = 0; s < k; ++s)
				subsum += L[i * pointDim + s] * L[k * pointDim + s];

			L[i * pointDim + k] = (A[i * pointDim + k] - subsum) / L[k * pointDim + k];
		}
	}

	float logDet = 1.0;
	for (size_t i = 0; i < pointDim; ++i) {
		float diag = L[i * pointDim + i];
		assert(diag > 0);
		logDet += logf(diag);
	}

	logDet *= 2.0;

	normalizers[comp] = - 0.5 * (pointDim * logf(2.0 * M_PI) + logDet);
}
