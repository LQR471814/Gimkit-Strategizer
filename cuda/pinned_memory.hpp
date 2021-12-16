template <typename T>
T* createHostPointer(T defaultValue) {
	T* host;
	cudaMallocHost(&host, sizeof(T));
	cudaHostRegister(host, sizeof(T), cudaHostRegisterDefault);
	*host = defaultValue;
	return host;
}

template <typename T>
T* createPinnedPointer(T *hostPointer) {
	T* pinned;
	cudaHostGetDevicePointer(&pinned, hostPointer, 0);
	return pinned;
}
