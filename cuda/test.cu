#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "types.h"

struct Scam {
	int someNumber;
	int result;
};

__device__ struct Scam sub(int *data, int a, int b) {
	return Scam{
		0, (*data) * (b - a)
	};
}

__global__ void thread(int *data, Scam *result) {
	result[threadIdx.x] = sub(data, 3, 2);
}

int main() {
	int threads = 32;

	int *data;
	cudaMallocManaged(&data, sizeof(int));
	*data = 3;

	Scam *res;
	cudaMallocManaged(&res, sizeof(Scam) * threads);

	thread<<<1, threads>>>(data, res);
	cudaDeviceSynchronize();

	for (int i = 0; i < threads; i++) {
		printf("result %d\n", res[i].result);
	};

	return 0;
}