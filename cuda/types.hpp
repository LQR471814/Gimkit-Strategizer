#include <vector>

typedef double Money;

typedef uint64_t Goal;
typedef int64_t Minimum;

typedef uint32_t ProblemCount;

typedef int8_t UpgradeId;
typedef uint8_t MaxUpgradeLevel;
typedef uint8_t Depth;

const uint16_t BLOCK_SIZE = 256;

const UpgradeId MONEY_PER_QUESTION = 0;
const UpgradeId STREAK_BONUS = 1;
const UpgradeId MULTIPLIER = 2;
const UpgradeId INSURANCE = 3;

const uint8_t UPGRADE_COUNT = 4;
const MaxUpgradeLevel MAX_LEVEL = 10;

struct UpgradeLevel
{
	float value;
	uint32_t cost;
};

std::vector<UpgradeLevel> moneyPerQuestionLevels = {
	{1, 0},
	{5, 10},
	{50, 100},
	{100, 1000},
	{500, 10000},
	{2000, 75000},
	{5000, 300000},
	{10000, 1000000},
	{250000, 10000000},
	{1000000, 100000000},
};

std::vector<UpgradeLevel> streakBonusLevels = {
	{1, 0},
	{3, 20},
	{10, 200},
	{50, 2000},
	{250, 20000},
	{1200, 200000},
	{6500, 2000000},
	{35000, 20000000},
	{175000, 200000000},
	{1000000, 2000000000},
};

std::vector<UpgradeLevel> multiplierLevels = {
	{1, 0},
	{1.5, 50},
	{2, 300},
	{3, 2000},
	{5, 12000},
	{8, 85000},
	{12, 700000},
	{18, 6500000},
	{30, 65000000},
	{100, 1000000000},
};

std::vector<UpgradeLevel> insuranceLevels = {
	{0, 0},
	{10, 10},
	{25, 250},
	{40, 1000},
	{50, 25000},
	{70, 100000},
	{80, 1000000},
	{90, 5000000},
	{95, 25000000},
	{99, 500000000},
};

//? This doesn't use a pointer because that's kinda hard to implement
struct UpgradeStats
{
	MaxUpgradeLevel moneyPerQuestion = 0;
	MaxUpgradeLevel streakBonus = 0;
	MaxUpgradeLevel multiplier = 0;
	MaxUpgradeLevel insurance = 0;
};

struct GoalResult
{
	ProblemCount problems = 0;
	Money newMoney = -1;
};

struct PlayState
{
	struct UpgradeStats stats = {};
	float setbackChance = 0;
	Money money = 0;
};

struct Permutation
{
	ProblemCount problems;
	std::vector<UpgradeId> sequence;
	struct PlayState play;
};

struct PermuteContext
{
	UpgradeLevel **data;
	std::vector<UpgradeId> upgrades;
	Depth max;
};

struct PermuteState
{
	struct PlayState play;
	std::vector<UpgradeId> sequence;
};

struct ComputeContext
{
	UpgradeLevel **data;
	Depth max;
	Money moneyGoal;

	UpgradeId *upgrades;
	MaxUpgradeLevel upgradesSize;

	Minimum *currentMinimum;
	uint8_t *cancel; //0 - running | 1 - stop
};

struct PlayStackParameters
{
	PlayState state = {};
	ProblemCount problems = 0;
	Minimum upperMinimum = -1;
};

struct PlayStackFrame
{
	PlayStackParameters params = {};
	UpgradeId branch = 0;
	Minimum currentMin = -1;
	UpgradeId minTarget = -1;
};

struct ComputeState
{
	struct PlayState init;
	ProblemCount problems;
	Depth *depth;
	PlayStackFrame *stack;
	UpgradeId *sequence;
};

struct ComputeOptions
{
	Depth syncDepth;
	Depth maxDepth;

	uint64_t timeout;
	std::string saveFilename;
	std::string recoverFrom;
	float loggingFidelity;
};

ComputeState copyComputeState(
	ComputeState original,
	int stackSize,
	int sequenceSize
) {
	ComputeState copied = {original.init, original.problems};

	cudaMallocManaged(&copied.depth, sizeof(Depth));
	*copied.depth = *original.depth;

	cudaMallocManaged(&copied.stack, sizeof(PlayStackFrame) * stackSize);
	for (int i = 0; i < stackSize; i++) {
		copied.stack[i] = original.stack[i];
	}

	cudaMallocManaged(&copied.sequence, sizeof(UpgradeId) * sequenceSize);
	for (int i = 0; i < sequenceSize; i++) {
		copied.sequence[i] = original.sequence[i];
	}

	return copied;
}

__host__ __device__ void printPlayState(PlayState p)
{
	printf(
		"$%f Stats %d %d %d %d\n",
		p.money,
		p.stats.moneyPerQuestion,
		p.stats.streakBonus,
		p.stats.multiplier,
		p.stats.insurance);
}

__host__ __device__ void printPlayStack(PlayStackFrame *stack, Depth depth)
{
	printf("Depth %d Branch %d\n", depth, stack[depth].branch);
	printf(" Params\n");
	printf(" -> Problems %d\n", stack[depth].params.problems);
	printf(" -> ");
	printPlayState(stack[depth].params.state);
	printf(" -> Current Min %ld Target %d\n", stack[depth].currentMin, stack[depth].minTarget);
}

__host__ __device__ void printComputeState(ComputeState state, int stackSize, int sequenceSize) {
	printPlayState(state.init);

	printf("Starting depth: %u\n", *state.depth);
	printPlayState(state.init);

	for (int i = 0; i < sequenceSize; i++) {
		printf("%d ", state.sequence[i]);
	}

	printf("\n");

	for (int i = 0; i < stackSize; i++) {
		printPlayStack(state.stack, i);
	}
}

__forceinline__ __host__ __device__ UpgradeStats incrementStat(UpgradeStats s, int id)
{
	switch (id)
	{
	case MONEY_PER_QUESTION:
		s.moneyPerQuestion++;
		break;
	case STREAK_BONUS:
		s.streakBonus++;
		break;
	case MULTIPLIER:
		s.multiplier++;
		break;
	case INSURANCE:
		s.insurance++;
		break;
	}

	return s;
}

__forceinline__ __host__ __device__ int getStat(UpgradeStats s, int id)
{
	switch (id)
	{
	case MONEY_PER_QUESTION:
		return s.moneyPerQuestion;
	case STREAK_BONUS:
		return s.streakBonus;
	case MULTIPLIER:
		return s.multiplier;
	case INSURANCE:
		return s.insurance;
	}

	return -1;
}

template <typename T>
void writeToPointer(char **ptr, T value)
{
	T *start = (T *)(*ptr);
	*start = value;
	*ptr = (char *)(start + 1);
}

template <typename T>
T readFromPointer(char **ptr)
{
	T *value = (T *)(*ptr);
	*ptr = (char *)(value + 1);
	return *value;
}

// Note: A custom serialization function is only required
//  for structs that contain a pointer of any kind

__forceinline__ size_t playStateSize()
{
	return sizeof(PlayState);
}

__forceinline__ size_t computeStateSize(int stackSize, int sequenceSize)
{
	return sizeof(PlayState) +
		   sizeof(Depth) +
		   sizeof(PlayStackFrame) * stackSize +
		   sizeof(UpgradeId) * sequenceSize;
}

void serializePlayState(PlayState state, char **ptr)
{
	writeToPointer<UpgradeStats>(ptr, state.stats);
	writeToPointer<float>(ptr, state.setbackChance);
	writeToPointer<Money>(ptr, state.money);
}

char *serializeComputeState(ComputeState s, int stackSize, int sequenceSize)
{
	char *ptr = (char *)malloc(computeStateSize(stackSize, sequenceSize));
	char *start = ptr;

	serializePlayState(s.init, &ptr);
	writeToPointer<Depth>(&ptr, *s.depth);

	for (int i = 0; i < stackSize; i++)
	{
		writeToPointer<PlayStackFrame>(&ptr, s.stack[i]);
	}

	for (int i = 0; i < sequenceSize; i++)
	{
		writeToPointer<UpgradeId>(&ptr, s.sequence[i]);
	}

	return start;
}

PlayState unserializePlayState(char** ptr)
{
	PlayState state = {};

	state.stats = readFromPointer<UpgradeStats>(ptr);
	state.setbackChance = readFromPointer<float>(ptr);
	state.money = readFromPointer<Money>(ptr);

	return state;
}

ComputeState unserializeComputeState(
	char* ptr, int stackSize, int sequenceSize
) {
	ComputeState state = {};

	cudaMallocManaged(&state.depth, sizeof(Depth));
	cudaMallocManaged(&state.stack, sizeof(PlayStackFrame) * stackSize);
	cudaMallocManaged(&state.sequence, sizeof(UpgradeId) * sequenceSize);

	state.init = unserializePlayState(&ptr);
	*state.depth = readFromPointer<Depth>(&ptr);

	for (int i = 0; i < stackSize; i++)
	{
		state.stack[i] = readFromPointer<PlayStackFrame>(&ptr);
	}

	for (int i = 0; i < sequenceSize; i++)
	{
		state.sequence[i] = readFromPointer<UpgradeId>(&ptr);
	}

	return state;
}
