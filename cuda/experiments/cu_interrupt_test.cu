#include <functional>
#include <iostream>
#include <csignal>

__global__ void longRunning(int *run) {
	printf("Waiting...\n");
	while (*run != 1) {}
	printf("Run value %d\n", *run);
}

int main(){
	int *run;
	cudaMallocHost(&run, sizeof(int));
	cudaHostRegister(run, sizeof(int), 0);
	*run = true;

	int *d_run;
	cudaHostGetDevicePointer(&d_run, run, 0);
	*d_run = true;
	longRunning<<<1, 1>>>(d_run);

	printf("Type 'quit' to stop the gpu\n");

	std::string inp = "";
	while (inp != "quit") {
		std::cin >> inp;
	}
	*run = 1;

	printf("Input done\n");
	cudaDeviceSynchronize();
	printf("SUCCESS\n");

	return EXIT_SUCCESS;
}
