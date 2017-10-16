#ifndef BARRIER_H
#define BARRIER_H

#include <stdlib.h>

#include <sys/types.h>
#include <pthread.h>

typedef struct {
	pthread_cond_t cv;
	pthread_mutex_t mx;
	size_t totalProcessors;
	size_t waitingProcessors;
} Barrier;

void initBarrier(
	Barrier* barrier,
	size_t headCount
);

void destroyBarrier(
	Barrier* barrier
);

void arriveAt(
	Barrier* barrier,
	void* arg,
	void (*callback)(void*)
);

#endif
