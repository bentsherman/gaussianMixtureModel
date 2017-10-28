#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

// Intentionally not including header since it is meant for gcc consumption.
// #include "cudaWrappers.h"

#include "cudaCommon.hu"
#include "cudaFolds.hu"
#include "cudaGmm.hu"
#include "cudaMVNormal.hu"
#include "gmm.h"

void gpuSum(size_t numPoints, const size_t pointDim, float* host_a, float* host_sum) {
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(host_a != NULL);
	assert(host_sum != NULL);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	// cudaArraySum is meant for powers of two
	size_t M = largestPowTwoLessThanEq(numPoints);

	float cpuSum[pointDim];
	memset(cpuSum, 0, pointDim * sizeof(float));
	for(size_t i = M; i < numPoints; ++i) {
		for(size_t j = 0; j < pointDim; ++j) {
			cpuSum[j] += host_a[i * pointDim + j];
		}
	}

	numPoints = M;

	float *device_a = sendToGpu(numPoints * pointDim, host_a);

	// cudaArraySum is synchronous
	cudaArraySum(
		&deviceProp, numPoints, pointDim, device_a
		);

	check(cudaMemcpy(host_sum, device_a, pointDim * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(device_a);

	for(size_t i = 0; i < pointDim; ++i) {
		host_sum[i] += cpuSum[i];
	}
}

float gpuMax(size_t N, float* host_a) {
	assert(host_a != NULL);
	assert(N > 0);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	float *device_a = sendToGpu(N, host_a);

	cudaArrayMax(
		&deviceProp, N, device_a
		);

	float gpuMax = 0;
	check(cudaMemcpy(&gpuMax, device_a, sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(device_a);

	return gpuMax;
}

void gpuLogMVNormDist(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* mu, const float* sigmaL,
	float* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	float* device_X = sendToGpu(numPoints * pointDim, X);
	float* device_mu = sendToGpu(pointDim, mu);
	float* device_sigmaL = sendToGpu(pointDim * pointDim, sigmaL);
	float* device_logP = mallocOnGpu(numPoints);

	dim3 grid, block;
	calcDim(numPoints, &deviceProp, &block, &grid);
	kernLogMVNormDist<<<grid, block>>>(
		numPoints, pointDim,
		device_X, device_mu, device_sigmaL,
		device_logP
		);

	check(cudaMemcpy(logP, device_logP, numPoints * sizeof(float), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	cudaFree(device_X);
	cudaFree(device_mu);
	cudaFree(device_sigmaL);
	cudaFree(device_logP);
}

float gpuGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const float* logpi, float* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	float* device_logpi = sendToGpu(numComponents, logpi);

	// Sending all data because logP is an array organized by:
	// [ <- numPoints -> ]_0 [ <- numPoints -> ]_... [ <- numPoints -> ]_{k-1}
	// So even though we are only using M of those points on the GPU,
	// we need all numPoints to ensure indexing by numPoints * k + i works
	// correctly to access prob(x_i|mu_k,Sigma_k).
	float* device_logP = sendToGpu(numComponents * numPoints, logP);

	float logL = cudaGmmLogLikelihoodAndGammaNK(
		& deviceProp,
		numPoints, numComponents,
		logpi, logP,
		device_logpi, device_logP
	);

	cudaFree(device_logpi);
	cudaFree(device_logP);

	return logL;
}

void gpuCalcLogGammaNK(
	const size_t numPoints, const size_t numComponents,
	const float* logpi, float* loggamma
) {
	gpuGmmLogLikelihood(
		numPoints, numComponents,
		logpi, loggamma
	);
}

void gpuCalcLogGammaK(
	const size_t numPoints, const size_t numComponents,
	const float* loggamma, float* logGamma
) {
	// Gamma[k] = max + log sum expf(loggamma - max)

	float* working = (float*)malloc(numPoints * sizeof(float));
	for(size_t k = 0; k < numComponents; ++k) {
		// TODO: refactor to have a generic z = a + log sum expf(x - a)
		memcpy(working, & loggamma[k * numPoints], numPoints * sizeof(float));
		float maxValue = gpuMax(numPoints, working);

		memcpy(working, & loggamma[k * numPoints], numPoints * sizeof(float));
		for(size_t i = 0; i < numPoints; ++i) {
			working[i] = expf(working[i] - maxValue);
		}

		float sum = 0;
		gpuSum(numPoints, 1, working, & sum);
 		logGamma[k] = maxValue + logf(sum );
	}
	free(working);
}


void gpuGmmFit(
	const float* X,
	const size_t numPoints,
	const size_t pointDim,
	const size_t numComponents,
	float* pi,
	float* Mu,
	float* Sigma,
	float* SigmaL,
	float* normalizers,
	const size_t maxIterations,
	GMM* gmm
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0 && pointDim <= 1024);
	assert(numComponents > 0 && numComponents <= 1024);

	assert(pi != NULL);
	assert(Mu != NULL);
	assert(Sigma != NULL);
	assert(SigmaL != NULL);
	assert(normalizers != NULL);

	assert(maxIterations >= 1);

	int deviceId;

	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	// printf("name: %s\n", deviceProp.name);
	// printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
	// printf("concurrentKernels: %d\n", deviceProp.concurrentKernels);

	float* device_X = pinHostAndSendDevice(numPoints * pointDim, (float*) X);

	for(size_t i = 0; i < numComponents; ++i) {
		assert(pi[i] > 0);
		pi[i] = logf(pi[i]);
	}

	float* device_logpi = pinHostAndSendDevice(numComponents, pi);
	float* device_Mu = pinHostAndSendDevice(numComponents * pointDim, Mu);
	float* device_Sigma = pinHostAndSendDevice(numComponents * pointDim * pointDim, Sigma);

	float* device_SigmaL = pinHostAndSendDevice(numComponents * pointDim * pointDim, SigmaL);
	float* device_normalizers = pinHostAndSendDevice(numComponents, normalizers);

	int error = 0;
	int* device_error = (int*) pinHostAndSendDevice(1, (float*) &error);

	float* device_loggamma = mallocOnGpu(numPoints * numComponents);
	float* device_logGamma = mallocOnGpu(numPoints * numComponents);

	float previousLogL = -INFINITY;

	float* pinnedCurrentLogL;
	cudaMallocHost(&pinnedCurrentLogL, sizeof(float));
	*pinnedCurrentLogL = -INFINITY;

	// logPx, mu, sigma reductions
	// This means for mu and sigma can only do one component at a time otherwise
	// the memory foot print will limit how much data we can actually work with.
	float* device_working = mallocOnGpu(numComponents * numPoints * pointDim * pointDim);

	dim3 grid, block;
	calcDim(numPoints, &deviceProp, &block, &grid);

	size_t iteration = 0;
	const float tolerance = 1e-8;

	cudaStream_t streams[numComponents];
	for(size_t k = 0; k < numComponents; ++k) {
		cudaStreamCreate(&streams[k]);
	}

	cudaEvent_t kernelEvent[numComponents + 1];
	for(size_t k = 0; k <= numComponents; ++k) {
		cudaEventCreateWithFlags(& kernelEvent[k], cudaEventDisableTiming);
	}

	try {
		do {
			// --------------------------------------------------------------------------
			// E-Step
			// --------------------------------------------------------------------------

			// loggamma[k * numPoints + i] = p(x_i | mu_k, Sigma_k )
			for(size_t k = 0; k < numComponents; ++k) {
				// Fill in numPoint many probabilities
				kernLogMVNormDist<<<grid, block, 0, streams[k]>>>(
					numPoints, pointDim,
					device_X,
					& device_Mu[k * pointDim],
					& device_SigmaL[k * pointDim * pointDim],
					& device_loggamma[k * numPoints]
				);

				cudaEventRecord(kernelEvent[k], streams[k]);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				// streams[numComponents - 1] needs to wait for everyone else to finish
				cudaStreamWaitEvent(streams[numComponents-1], kernelEvent[k], 0);
			}

			// loggamma[k * numPoints + i] = p(x_i | mu_k, Sigma_k) / p(x_i)
			// working[i] = p(x_i)
			kernCalcLogLikelihoodAndGammaNK<<<grid, block, 0, streams[numComponents - 1]>>>(
				numPoints, numComponents,
				device_logpi, device_working, device_loggamma
			);

			// working[0] = sum_{i} p(x_i)
			cudaArraySum(&deviceProp, numPoints, 1, device_working, streams[numComponents - 1]);

			previousLogL = *pinnedCurrentLogL;
			check(cudaMemcpyAsync(
				pinnedCurrentLogL, device_working,
				sizeof(float),
				cudaMemcpyDeviceToHost,
				streams[numComponents - 1]
			));

			for(size_t k = 0; k < numComponents; ++k) {
				// synchronize everybody with the host
				cudaStreamSynchronize(streams[k]);
			}

			if(fabsf(*pinnedCurrentLogL - previousLogL) < tolerance || *pinnedCurrentLogL < previousLogL) {
				break;
			}

			// --------------------------------------------------------------------------
			// M-Step
			// --------------------------------------------------------------------------

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				cudaLogSumExp(
					& deviceProp, grid, block,
					numPoints,
					& device_loggamma[k * numPoints], & device_logGamma[k * numPoints],
					device_workingK,
					streams[k]
				);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				// working[i * pointDim + j] = gamma_ik / Gamma K * x_j
				kernCalcMu<<<grid, block, 0, streams[k]>>>(
					numPoints, pointDim,
					device_X,
					& device_loggamma[k * numPoints],
					& device_logGamma[k * numPoints],
					device_workingK
				);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				// working[0 + j] = sum gamma_ik / Gamma K * x_j
				cudaArraySum(
					&deviceProp, numPoints, pointDim,
					device_workingK,
					streams[k]
				);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				check(cudaMemcpyAsync(
					& device_Mu[k * pointDim],
					device_workingK,
					pointDim * sizeof(float),
					cudaMemcpyDeviceToDevice,
					streams[k]
				));
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				check(cudaMemcpyAsync(
					& device_Sigma[k * pointDim * pointDim],
					device_workingK,
					pointDim * pointDim * sizeof(float),
					cudaMemcpyDeviceToDevice,
					streams[k]
				));
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				kernCalcSigma<<<grid, block, 0, streams[k]>>>(
					numPoints, pointDim,
					device_X,
					& device_Mu[k * pointDim],
					& device_loggamma[k * numPoints],
					& device_logGamma[k * numPoints],
					device_workingK
				);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				// working[0 + j] = sum gamma_ik / Gamma K * [...]_j
				cudaArraySum(
					&deviceProp, numPoints, pointDim * pointDim,
					device_workingK,
					streams[k]
				);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				float* device_workingK = & device_working[k * numPoints * pointDim * pointDim];
				check(cudaMemcpyAsync(
					& device_Sigma[k * pointDim * pointDim],
					device_workingK,
					pointDim * pointDim * sizeof(float),
					cudaMemcpyDeviceToDevice,
					streams[k]
				));

				cudaEventRecord(kernelEvent[k], streams[k]);
			}

			for(size_t k = 0; k < numComponents; ++k) {
				// streams[numComponents - 1] needs to wait for everyone else to finish
				cudaStreamWaitEvent(streams[numComponents-1], kernelEvent[k], 0);
			}

			// pi_k^(t+1) = pi_k Gamma_k / sum_{i}^{K} pi_i * Gamma_i
			// Use thread sync to compute denom to avoid data race
			kernUpdatePi<<<1, numComponents, 0, streams[numComponents - 1]>>>(
				numPoints, numComponents,
				device_logpi, device_logGamma
			);

			// recompute sigmaL and normalizer
			kernPrepareCovariances<<<1, numComponents, 0, streams[numComponents - 1]>>>(
				numComponents, pointDim,
				device_Sigma, device_SigmaL,
				device_normalizers,
				device_error
			);

			cudaEventRecord(kernelEvent[numComponents], streams[numComponents - 1]);

			for(size_t k = 0; k < numComponents; ++k) {
				// Everyone needs to wait for the work on streams[numComponents - 1] to finish.
				cudaStreamWaitEvent(streams[k], kernelEvent[numComponents], 0);
			}

			// check error to see if inverse failed
			check(cudaMemcpy(&error, device_error, sizeof(int), cudaMemcpyDeviceToHost));

			if ( error ) {
				throw std::runtime_error("Failed to compute inverse");
			}

		} while(++iteration < maxIterations);

		// copy loggamma to host to compute output labels
		float* loggamma = (float *)malloc(numPoints * numComponents * sizeof(float));

		check(cudaMemcpy(
			loggamma,
			device_loggamma,
			numPoints * numComponents * sizeof(float),
			cudaMemcpyDeviceToHost
		));

		gmm->failed = false;
		gmm->y_pred = calcLabels(loggamma, numPoints, numComponents);
		gmm->logL = *pinnedCurrentLogL;
	}
	catch ( std::runtime_error& e ) {
		fprintf(stderr, "warning: model failed\n");
		gmm->failed = true;
	}

	for(size_t k = 0; k <= numComponents; ++k) {
		cudaEventDestroy(kernelEvent[k]);
	}

	for(size_t k = 0; k < numComponents; ++k) {
		cudaStreamDestroy(streams[k]);
	}

	cudaFreeHost(pinnedCurrentLogL);
	cudaFree(device_working);
	cudaFree(device_logGamma);
	cudaFree(device_loggamma);

	unpinHost(device_error, &error);
	recvDeviceUnpinHost(device_normalizers, normalizers, numComponents);
	recvDeviceUnpinHost(device_SigmaL, SigmaL, numComponents * pointDim * pointDim);
	recvDeviceUnpinHost(device_Sigma, Sigma, numComponents * pointDim * pointDim);
	recvDeviceUnpinHost(device_Mu, Mu, numComponents * pointDim);
	recvDeviceUnpinHost(device_logpi, pi, numComponents);

	for(size_t i = 0; i < numComponents; ++i) {
		pi[i] = expf(pi[i]);
	}

	unpinHost(device_X, (float*) X);
}
