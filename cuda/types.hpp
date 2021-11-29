#include <curand.h>
#include <vector>

typedef double Money;

const int BLOCK_SIZE = 256;

const int MONEY_PER_QUESTION = 0;
const int STREAK_BONUS = 1;
const int MULTIPLIER = 2;
const int INSURANCE = 3;

const int UPGRADE_COUNT = 4;
const int MAX_LEVEL = 10;

struct UpgradeLevel
{
	float value;
	int cost;
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
	int moneyPerQuestion = 0;
	int streakBonus = 0;
	int multiplier = 0;
	int insurance = 0;
};

struct GoalResult
{
	int problems = -1;
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
	int problems;
	std::vector<int> sequence;
	struct PlayState play;
};

struct PermuteContext
{
	UpgradeLevel **data;
	std::vector<int> upgrades;
	int max;
};

struct PermuteState
{
	struct PlayState play;
	std::vector<int> sequence;
};

struct ComputeContext
{
	UpgradeLevel **data;
	int max;
	Money moneyGoal;

	int *upgrades;
	int upgradesSize;

	int *currentMinimum = NULL;
};

struct PlayStackParameters
{
	PlayState state = {};
	int problems = 0;
	int upperMinimum = -1;
};

struct PlayStackFrame
{
	PlayStackParameters params = {};
	int branch = 0;
	int currentMin = -1;
	int minTarget = -1;
};

struct TComputeStates
{
	struct PlayState init;
	PlayStackFrame *stack;
	int problems;
	int *sequence;
};

struct ComputeOptions {
	unsigned int syncDepth;
	unsigned int maxDepth;

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
