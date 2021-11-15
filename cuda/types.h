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

struct RecurseContext
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

struct TRecurseResult
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
	bool verboseLog;
};
