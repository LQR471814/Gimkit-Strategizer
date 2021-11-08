#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void oof(int *progress)
{
	volatile int x = 0;
	while (x < 10000) {
		x += 17/23 + 2;
	};

	atomicAdd((int*)progress, 1);
}

int * createHostProgress() {
	int *progress;
	cudaMallocHost(&progress, sizeof(int));
	cudaHostRegister(progress, sizeof(int), 0);
	*progress = 0;

	return progress;
}

int * createPinnedProgress(int *hostPtr) {
	int *pinnedPtr;
	cudaHostGetDevicePointer(&pinnedPtr, hostPtr, 0);
	return pinnedPtr;
}

int main()
{
	int threads = 256;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *progress = createHostProgress();
	int *pinnedPtr = createPinnedProgress(progress);

	cudaEventRecord(start);
	oof<<<1, threads>>>(pinnedPtr);
	cudaEventRecord(stop);

	int hostProgress = 0;

	do {
		cudaEventQuery(stop);
		int n = *progress;
		if (n - hostProgress >= threads * 0.1) {
			hostProgress = n;
			printf("Progress %d / %d\n", hostProgress, threads);
		};
	} while (hostProgress < threads);

	cudaEventSynchronize(stop);

	printf("Done!\n");
	return 0;
}