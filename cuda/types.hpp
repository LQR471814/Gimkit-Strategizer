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

struct TComputeStates
{
	struct PlayState init;
	PlayStackFrame *stack;
	ProblemCount problems;
	UpgradeId *sequence;
};

struct ComputeOptions {
	Depth syncDepth;
	Depth maxDepth;

	float loggingFidelity;
};

__host__ __device__ void printPlayState(PlayState p) {
	printf(
		"$%f Stats %d %d %d %d\n",
		p.money,
		p.stats.moneyPerQuestion,
		p.stats.streakBonus,
		p.stats.multiplier,
		p.stats.insurance
	);
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
