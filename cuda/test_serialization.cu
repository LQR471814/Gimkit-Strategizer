#include <iostream>
#include "types.hpp"

void testPlayState() {
	PlayState state = {{1, 1, 1, 1}, 0, 100};

	char *data = (char*)malloc(sizeof(PlayState));
	char *start = data;
	serializePlayState(state, &data);

	PlayState result = unserializePlayState(&start);
	printPlayState(result);
}

void testComputeState() {
	int stackSize = 12;
	int sequenceSize = 20;
	int stateCount = 2;

	ComputeState state = {};
	state.depth = (Depth*)malloc(sizeof(Depth));
	*state.depth = 0;
	state.init = {{1, 1, 1, 1}, 0, 100};

	state.sequence = (UpgradeId*)malloc(sizeof(UpgradeId) * sequenceSize);
	for (int i = 0; i < sequenceSize; i++) {
		state.sequence[i] = -1;
	}

	state.stack = (PlayStackFrame*)malloc(sizeof(PlayStackFrame) * stackSize);
	for (int i = 0; i < stackSize; i++) {
		state.stack[i] = {{{state.init}, 123, 7}, 2, 42, 3};
	}

	size_t stateSize = computeStateSize(stackSize, sequenceSize);

	char *data = (char*)malloc(stateSize);
	for (int i = 0; i < stateCount; i++) {
		char* buf = serializeComputeState(state, stackSize, sequenceSize);
		for (int x = 0; x < stateSize; x++) {
			data[x + i*stateSize] = buf[x];
		}
	}

	FILE *pFile;
	pFile = fopen("test.bin", "wb");
	for (int i = 0; i < 2; i++) {
		fwrite(
			data, sizeof(char),
			stateSize, pFile
		);
	}
	fclose(pFile);

	pFile = fopen("serialized.bin", "rb");

	char* readBytes = (char*)malloc(stateSize * 2);
	for (int i = 0; i < 2; i++) {
		fread(readBytes, sizeof(char), stateSize, pFile);
	}

	ComputeState result = unserializeComputeState(readBytes, stackSize, sequenceSize);
	result.problems = 0;

	printComputeState(result, stackSize, sequenceSize);
}

int main() {
	testPlayState();
	testComputeState();

	return 0;
}