#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "types.h"

struct ScamResult {
	int someNumber;
	float *result;
	curandState *randGen;
};

__forceinline__ __device__ int process(int *yo, int ack, int value) {
	ack++;
	yo[ack] = value;
	return ack;
}

__global__ void thread(int *data, ScamResult *result) {
	curand_init(1234, threadIdx.x, 0, result[threadIdx.x].randGen);
	*result[threadIdx.x].result = curand_uniform(result[threadIdx.x].randGen);

	int *yo = new int[10];
	yo[0] = 1234;
	printf("%d\n", yo[0]);

	int ok = process(yo, 0, 4321);
	printf("Ok %d ACKK %d\n", ok, yo[ok]);
}

int main(int argc, char** argv) {
	printf("Arguments %d\n", argc);

	for (int i = 1; i < argc; i++) {
		printf("Argument Index %d Value %s\n", argc, argv[i]);
	};

	int threads = 32;

	int *data;
	cudaMallocManaged(&data, sizeof(int));
	*data = 3;

	ScamResult *res;
	cudaMallocManaged(&res, sizeof(ScamResult) * threads);
	for (int i = 0; i < threads; i++) {
		float *result;
		cudaMallocManaged(&result, sizeof(int));

		curandState *gen;
		cudaMallocManaged(&gen, sizeof(curandState));

		res[i] = ScamResult{
			0, result, gen
		};
	};

	thread<<<1, threads>>>(data, res);
	cudaDeviceSynchronize();

	for (int i = 0; i < threads; i++) {
		printf("result %f\n", *res[i].result);
	};

	return 0;
}