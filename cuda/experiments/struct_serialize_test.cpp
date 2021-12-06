#include <stdio.h>
#include <iostream>

template <typename T>
class SerializableArray {
	public:
		T* ptr;
		int size;
};

struct FancyStruct {
	uint8_t a1;
	float a2;
	uint32_t* a3;
	size_t a3Size;
};

template <typename T>
void writeToPointer(char** ptr, T value) {
	T* start = (T*)(*ptr);
	*start = value;
	*ptr = (char*)(start + 1);
}

template <typename T>
T readFromPointer(char** ptr) {
	T* value = (T*)(*ptr);
	*ptr = (char*)(value + 1);
	return *value;
}

char* serializeFancyStruct(FancyStruct s) {
	char *ptr = (char*)malloc(sizeof(FancyStruct) - sizeof(uint32_t*));
	char *start = ptr;

	writeToPointer<uint8_t>(&ptr, s.a1);
	writeToPointer<float>(&ptr, s.a2);
	writeToPointer<size_t>(&ptr, s.a3Size);

	return start;
}

FancyStruct deserializeFancyStruct(char* ptr) {
	FancyStruct result = {};

	result.a1 = readFromPointer<uint8_t>(&ptr);
	result.a2 = readFromPointer<float>(&ptr);
	result.a3Size = readFromPointer<size_t>(&ptr);

	return result;
}

template <typename T>
void serialize(T *array, int size) {
	FILE *pFile;
	pFile = fopen("serialized.bin", "wb");
	fwrite(array, sizeof(T), size, pFile);
	fclose(pFile);
}

template <typename T>
void unserialize() {
	FILE *pFile;
	pFile = fopen("serialized.bin", "rb");
	if (pFile == NULL) {
		fputs("File error", stderr);
		exit(1);
	}

	fseek(pFile, 0, SEEK_END);
	long lSize = ftell(pFile);
	rewind(pFile);

	// allocate memory to contain the whole file:
	FancyStruct* unserializedArray = (T*) malloc(lSize);
	if (unserializedArray == NULL) {
		fputs ("Memory error",stderr);
		exit (2);
	}

	// copy the file into the buffer:
	size_t n = fread(unserializedArray, 1, lSize, pFile);
	if (n != lSize) {
		fputs ("Reading error", stderr);
		exit (3);
	}

	printf("A1 %d A2 %f\n", unserializedArray[0].a1, unserializedArray[0].a2);

	free(unserializedArray);
	fclose(pFile);
}

int main() {
	uint32_t* a = (uint32_t*)malloc(sizeof(uint32_t) * 10);
	for (int i = 0; i < 10; i++) {
		a[i] = i;
	}

	FancyStruct s = {(uint8_t)6, (float)9, a, (size_t)10};
	char *ptr = serializeFancyStruct(s);
	FancyStruct afterDeserialization = deserializeFancyStruct(ptr);
	printf(
		"A1 %d A2 %f Size %d\n",
		afterDeserialization.a1,
		afterDeserialization.a2,
		afterDeserialization.a3Size
	);

	// serialize<FancyStruct>(array, 10);
	// unserialize<FancyStruct>();

	return 0;
}