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

__global__ void thread(int *data) {
	Scam val = sub(data, 3, 2);
	printf("%d\n", val.result);
}

int main() {
	int *data;
	cudaMallocManaged(&data, sizeof(int));
	*data = 3;

	thread<<<1, 32>>>(data);

	return 0;
}