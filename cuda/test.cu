#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>

// Kernel function to add the elements of two arrays
__global__
void multiply(int n, float *r)
{
	curandGenerator_t *g;
	cudaMallocManaged(&g, sizeof(curandGenerator_t));

	curandCreateGenerator(g, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateUniform(*g, r, n);
}

int main(void)
{
	// int threads = 256;
	int N = 1<<20;
	float *r;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&r, N*sizeof(float));

	// Run kernel on 1M elements on the GPU
	multiply<<<1, 1>>>(N, r);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	for (int i = 0; i < N; i++)
		std::cout << printf("%f", r[i]) << std::endl;

	// Free memory
	cudaFree(r);

	return 0;
}
