#include <cuda_runtime.h>
#include <iostream>

void printGPUInfo() {
	cudaDeviceProp *props;
	cudaMallocManaged(&props, sizeof(cudaDeviceProp));
	cudaError_t error = cudaGetDeviceProperties(props, 0);
	if (error != cudaSuccess) {
		printf("GPU version error %s\n", cudaGetErrorString(error));
	}

	printf("GPU Compute Capability %d.%d\n\n", props->major, props->minor);
}
