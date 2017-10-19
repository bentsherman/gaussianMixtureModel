#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "component.h"
#include "gmm.h"
#include "kmeans.h"
#include "linearAlgebra.h"
#include "util.h"

GMM* initGMM(
	const float* X,
	const size_t numPoints,
	const size_t pointDim,
	const size_t numComponents
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);

	// X is an numPoints x pointDim set of training data

	GMM* gmm = (GMM*)checkedCalloc(1, sizeof(GMM));
	gmm->pointDim = pointDim;
	gmm->numComponents = numComponents;
	gmm->components = (Component*) checkedCalloc(numComponents, sizeof(Component));

	// Seed with kmeans (seeding with random points can lead to degeneracy)
	float M[pointDim * numComponents];
	for(size_t k = 0; k < numComponents; ++k) {
		// Use a random point for mean of each component
		size_t j = rand() % numPoints;
		for(size_t dim = 0; dim < gmm->pointDim; dim++) {
			M[k * pointDim + dim] = X[j * gmm->pointDim + dim];
		}
	}

	kmeans(X, numPoints, pointDim, M, numComponents);

	float uniformTau = 1.0 / numComponents;
	for(size_t k = 0; k < gmm->numComponents; ++k) {
		Component* component = & gmm->components[k];

		// Assume every component has uniform weight
		component->pi = uniformTau;

		component->mu = (float*)checkedCalloc(pointDim, sizeof(float));
		memcpy(component->mu, &M[k * pointDim], pointDim * sizeof(float));

		// Use identity covariance- assume dimensions are independent
		component->sigma = (float*)checkedCalloc(pointDim * pointDim, sizeof(float));
		for (size_t dim = 0; dim < pointDim; ++dim)
			component->sigma[dim * pointDim + dim] = 1;

		// Initialize zero artifacts
		component->sigmaL = (float*)checkedCalloc(pointDim * pointDim, sizeof(float));
		component->normalizer = 0;

		prepareCovariance(component, pointDim);
	}


	return gmm;
}

void freeGMM(GMM* gmm) {
	assert(gmm != NULL);

	for(size_t k = 0; k < gmm->numComponents; ++k) {
		Component* component = & gmm->components[k];
		free(component->mu);
		free(component->sigma);
		free(component->sigmaL);
	}

	free(gmm->components);
	free(gmm->y_pred);
	free(gmm);
}

void calcLogMvNorm(
	const Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	const float* X, const size_t numPoints, const size_t pointDim,
	float* logProb
) {
	assert(components != NULL);
	assert(numComponents > 0);
	assert(componentStart < componentEnd);
	assert(componentEnd > 0);
	assert(X != NULL);
	assert(numComponents > 0);
	assert(pointDim > 0);
	assert(logProb != NULL);

	for (size_t k = componentStart; k < componentEnd; ++k) {
		logMvNormDist(
			& components[k], pointDim,
			X, numPoints,
			& logProb[k * numPoints]
		);
	}
}

void logLikelihood(
	const float* logpi, const size_t numComponents,
	const float* logProb, const size_t numPoints,
	const size_t pointStart, const size_t pointEnd,
	float* logL
) {
	assert(logpi != NULL);
	assert(numComponents > 0);
	assert(logProb != NULL);
	assert(numPoints > 0);
	assert(pointStart < pointEnd);
	assert(pointEnd <= numPoints);
	assert(logL != NULL);

	*logL = 0.0;
	for (size_t point = pointStart; point < pointEnd; ++point) {
		float maxArg = -INFINITY;
		for(size_t k = 0; k < numComponents; ++k) {
			const float logProbK = logpi[k] + logProb[k * numPoints + point];
			if(logProbK > maxArg) {
				maxArg = logProbK;
			}
		}

		float sum = 0.0;
		for (size_t k = 0; k < numComponents; ++k) {
			const float logProbK = logpi[k] + logProb[k * numPoints + point];
			sum += expf(logProbK - maxArg);
		}

		assert(sum >= 0);
		*logL += maxArg + logf(sum);
	}
}

int shouldContinue(
	const float prevLogL, const float currentLogL,
	const float tolerance
) {
	// In principle this shouldn't happen, but if it does going to assume we're in
	// an odd state and that we should terminate.
	if(currentLogL < prevLogL) {
		return 0;
	}

	if(fabsf(currentLogL - prevLogL) < tolerance ) {
		return 0;
	}

	return 1;
}

void calcLogGammaNK(
	const float* logpi, const size_t numComponents,
	const size_t pointStart, const size_t pointEnd,
	float* loggamma, const size_t numPoints
) {
	assert(logpi != NULL);
	assert(numComponents > 0);

	assert(numPoints > 0);
	assert(pointStart < pointEnd);
	assert(pointEnd <= numPoints);

	assert(loggamma != NULL);

	for (size_t point = pointStart; point < pointEnd; ++point) {
		float maxArg = -INFINITY;
		for (size_t k = 0; k < numComponents; ++k) {
			const float arg = logpi[k] + loggamma[k * numPoints + point];
			if(arg > maxArg) {
				maxArg = arg;
			}
		}

		// compute log p(x)
		float sum = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			const float arg = logpi[k] + loggamma[k * numPoints + point];
			sum += expf(arg - maxArg);
		}
		assert(sum >= 0);

		const float logpx = maxArg + logf(sum);
		for(size_t k = 0; k < numComponents; ++k) {
			loggamma[k * numPoints + point] += -logpx;
		}
	}
}

void logLikelihoodAndGammaNK(
	const float* logpi, const size_t numComponents,
	float* logProb, const size_t numPoints,
	const size_t pointStart, const size_t pointEnd,
	float* logL
) {
	assert(logpi != NULL);
	assert(numComponents > 0);
	assert(logProb != NULL);
	assert(numPoints > 0);
	assert(pointStart < pointEnd);
	assert(pointEnd <= numPoints);
	assert(logL != NULL);

	*logL = 0.0;
	for (size_t point = pointStart; point < pointEnd; ++point) {
		float maxArg = -INFINITY;
		for(size_t k = 0; k < numComponents; ++k) {
			const float logProbK = logpi[k] + logProb[k * numPoints + point];
			if(logProbK > maxArg) {
				maxArg = logProbK;
			}
		}

		float sum = 0.0;
		for (size_t k = 0; k < numComponents; ++k) {
			const float logProbK = logpi[k] + logProb[k * numPoints + point];
			sum += expf(logProbK - maxArg);
		}

		assert(sum >= 0);
		const float logpx = maxArg + logf(sum);
		*logL += logpx;
		for(size_t k = 0; k < numComponents; ++k) {
			logProb[k * numPoints + point] += -logpx;
		}
	}
}

void calcLogGammaK(
	const float* loggamma, const size_t numPoints,
	const size_t componentStart, const size_t componentEnd,
	float* logGamma, const size_t numComponents
) {
	assert(loggamma != NULL);
	assert(numPoints > 0);

	assert(componentStart < componentEnd);
	assert(componentEnd <= numComponents);
	assert(numComponents > 0);

	assert(logGamma != NULL);

	memset(&logGamma[componentStart], 0, (componentEnd - componentStart) * sizeof(float));
	for(size_t k = componentStart; k < componentEnd; ++k) {
		const float* loggammak = & loggamma[k * numPoints];

		float maxArg = -INFINITY;
		for(size_t point = 0; point < numPoints; ++point) {
			const float loggammank = loggammak[point];
			if(loggammank > maxArg) {
				maxArg = loggammank;
			}
		}

		float sum = 0;
		for(size_t point = 0; point < numPoints; ++point) {
			const float loggammank = loggammak[point];
			sum += expf(loggammank - maxArg);
		}
		assert(sum >= 0);

		logGamma[k] = maxArg + logf(sum);
	}
}


float calcLogGammaSum(
	const float* logpi, const size_t numComponents,
	const float* logGamma
) {
	float maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const float arg = logpi[k] + logGamma[k];
		if(arg > maxArg) {
			maxArg = arg;
		}
	}

	float sum = 0;
	for (size_t k = 0; k < numComponents; ++k) {
		const float arg = logpi[k] + logGamma[k];
		sum += expf(arg - maxArg);
	}
	assert(sum >= 0);

	return maxArg + logf(sum);
}

void performMStep(
	Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	float* logpi, float* loggamma, float* logGamma, const float logGammaSum,
	const float* X, const size_t numPoints, const size_t pointDim,
	float* outerProduct, float* xm
) {
	assert(components != NULL);
	assert(numComponents > 0);
	assert(componentStart < componentEnd);
	assert(componentEnd <= numComponents);
	assert(logpi != NULL);
	assert(loggamma != NULL);
	assert(logGamma != NULL);
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(outerProduct != NULL);
	assert(xm != NULL);


	// update pi
	for(size_t k = componentStart; k < componentEnd; ++k) {
		Component* component = & components[k];
		logpi[k] += logGamma[k] - logGammaSum;
		component->pi = expf(logpi[k]);
		assert(0 <= component->pi && component->pi <= 1);
	}

	// Convert loggamma and logGamma over to gamma and logGamma to avoid duplicate,
	//  and costly, expf(x) calls.
	for(size_t k = componentStart; k < componentEnd; ++k) {
		for(size_t n = 0; n < numPoints; ++n) {
			const size_t i = k * numPoints + n;
			loggamma[i] = expf(loggamma[i]);
		}
	}

	for(size_t k = componentStart; k < componentEnd; ++k) {
		logGamma[k] = expf(logGamma[k]);
	}

	// Update mu
	for(size_t k = componentStart; k < componentEnd; ++k) {
		Component* component = & components[k];

		memset(component->mu, 0, pointDim * sizeof(float));
		for (size_t point = 0; point < numPoints; ++point) {
			for (size_t dim = 0; dim < pointDim; ++dim) {
				component->mu[dim] += loggamma[k * numPoints + point] * X[point * pointDim + dim];
			}
		}

		for (size_t i = 0; i < pointDim; ++i) {
			component->mu[i] /= logGamma[k];
		}
	}

	// Update sigma
	for(size_t k = componentStart; k < componentEnd; ++k) {
		Component* component = & components[k];
		memset(component->sigma, 0, pointDim * pointDim * sizeof(float));
		for (size_t point = 0; point < numPoints; ++point) {
			// (x - m)
			for (size_t dim = 0; dim < pointDim; ++dim) {
				xm[dim] = X[point * pointDim + dim] - component->mu[dim];
			}

			// (x - m) (x - m)^T
			for (size_t row = 0; row < pointDim; ++row) {
				for (size_t column = 0; column < pointDim; ++column) {
					outerProduct[row * pointDim + column] = xm[row] * xm[column];
				}
			}

			for (size_t i = 0; i < pointDim * pointDim; ++i) {
				component->sigma[i] += loggamma[k * numPoints + point] * outerProduct[i];
			}
		}

		for (size_t i = 0; i < pointDim * pointDim; ++i) {
			component->sigma[i] /= logGamma[k];
		}

		prepareCovariance(component, pointDim);
	}
}

int* calcLabels(float* loggamma, size_t numPoints, size_t numComponents)
{
	int* y = (int*)checkedCalloc(numPoints, sizeof(int));

	for ( size_t i = 0; i < numPoints; i++ ) {
		int max_j = -1;
		float max_gamma;

		for ( size_t j = 0; j < numComponents; j++ ) {
			if ( max_j == -1 || max_gamma < loggamma[i * numComponents + j] ) {
				max_j = j;
				max_gamma = loggamma[i * numComponents + j];
			}
		}

		y[i] = max_j;
	}

	return y;
}

float* generateGmmData(
	const size_t numPoints, const size_t pointDim, const size_t numComponents
) {
	float* X = (float*)checkedCalloc(numPoints * pointDim, sizeof(float));

	// Select mixture coefficients (could sample this from Dirichlet, but this is
	// computationally more efficient.)
	float pi[numComponents];
	float piSum = 0;

	float limit = 2.0 * numComponents / (float) numPoints;

	for(size_t i = 0; i < numComponents; ++i) {
		do {
			pi[i] = rand() / (float) RAND_MAX;
		} while (pi[i] < limit);
		piSum += pi[i];
	}

	for(size_t i = 0; i < numComponents; ++i) {
		pi[i] /= piSum;
	}

	size_t pointsPerComponent[numComponents];
	for(size_t i = 0; i < numComponents; ++i) {
		pointsPerComponent[i] = (size_t)round(pi[i]*numPoints);
	}

	float covLx[pointDim];
	float mean[pointDim];

	size_t xi = 0;
	for(size_t k = 0; k < numComponents && xi < numPoints; ++k) {
		// Select component mean
		for(size_t i = 0; i < pointDim; ++i) {
			mean[i] = numComponents * sampleStandardNormal();
		}

		// Select component covariance (dof is just a heuristic)
		const size_t dof = pointDim + 1 + numComponents;
		float* covL = sampleWishartCholesky(pointDim, dof);

		// Sample points from component proportional to component mixture coefficient
		for(size_t i = 0; i < pointsPerComponent[k] && xi < numPoints; ++i) {
			float* x = & X[xi * pointDim];
			for(size_t d = 0; d < pointDim; ++d) {
				x[d] = sampleStandardNormal();
			}

			lowerDiagByVector(covL, x, covLx, pointDim);
			vectorAdd(mean, covLx, x, pointDim);
			++xi;
		}

		free(covL);
	}

	// fisher yates shuffle
	for(size_t i = numPoints - 1; i > 0; --i) {
		size_t j = rand() % i;
		for(size_t d = 0; d < pointDim; ++d) {
			float t = X[i * pointDim + d];
			X[i * pointDim + d] = X[j * pointDim + d];
			X[j * pointDim + d] = t;
		}
	}

	// alternative to above is to sample k ~ pi each iteration.

	return X;
}

void printGmmToConsole(GMM* gmm) {
	assert(gmm != NULL);

	fprintf(stdout, "{\n");
	fprintf(stdout, "\"pointDim\" : %zu,\n", gmm->pointDim);
	fprintf(stdout, "\"numComponents\" : %zu,\n", gmm->numComponents);
	fprintf(stdout, "\"mixtures\" : [\n");
	for (size_t k = 0; k < gmm->numComponents; ++k) {
		printToConsole(& gmm->components[k], gmm->pointDim);
		if(k + 1 != gmm->numComponents) {
			fprintf(stdout, ", ");
		}
	}
	fprintf(stdout, "]\n");
	fprintf(stdout, "}\n");
}
