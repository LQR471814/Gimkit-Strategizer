#include <curand.h>
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
	curandState *randState = NULL;
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
	uint8_t *running; //? 0 - running : 1 - stop
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

struct TComputeState
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

	float loggingFidelity;
};

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
	return sizeof(PlayState) - sizeof(curandState *);
}

__forceinline__ size_t computeStateSize(int stackSize, int sequenceSize)
{
	return sizeof(TComputeState) -
		   sizeof(PlayStackFrame *) -
		   sizeof(UpgradeId *) +
		   sizeof(PlayStackFrame) * stackSize +
		   sizeof(UpgradeId) * sequenceSize;
}

char *serializeComputeState(TComputeState s, int stackSize, int sequenceSize)
{
	char *ptr = (char *)malloc(computeStateSize(stackSize, sequenceSize));
	char *start = ptr;

	char *serializedState = serializePlayState(s.init);
	for (int i = 0; i < playStateSize(); i++)
	{
		writeToPointer(&ptr, serializedState[i]);
	}

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

char *serializePlayState(PlayState state)
{
	char *ptr = (char *)malloc(playStateSize());
	char *start = ptr;

	writeToPointer<UpgradeStats>(&ptr, state.stats);
	writeToPointer<float>(&ptr, state.setbackChance);
	writeToPointer<Money>(&ptr, state.money);

	return start;
}
