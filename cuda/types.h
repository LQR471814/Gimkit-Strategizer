#include <curand.h>
#include <vector>

struct GoalResult
{
	int problems;
	int newMoney;
};

struct UpgradeStats
{
	int moneyPerQuestion;
	int streakBonus;
	int multiplier;
	int insurance;
};

struct PlayState
{
	struct UpgradeStats stats;
	float setbackChance;
	int money;
	curandState *randState;
};

struct Permutation
{
	int problems;
	std::vector<int> sequence;
	PlayState play;
};

struct PermuteContext {
	struct UpgradeIndex *data;
	std::vector<int> upgrades;
	int max;
};

struct PermuteState
{
	struct PlayState play;
	std::vector<int> sequence;
};

struct RecurseContext {
	struct UpgradeIndex *data;
	int max;
	int moneyGoal;

	int *upgrades;
	int upgradesSize;
};

struct RecurseState
{
	struct PlayState play;
	int target;
};

struct UpgradeLevel
{
	float value;
	int cost;
};

enum Upgrades
{
	MONEY_PER_QUESTION,
	STREAK_BONUS,
	MULTIPLIER,
	INSURANCE,
};

const int MAX_LEVEL = 10;

struct UpgradeIndex
{
	struct UpgradeLevel moneyPerQuestion[10];
	struct UpgradeLevel streakBonus[10];
	struct UpgradeLevel multiplier[10];
	struct UpgradeLevel insurance[10];
};

struct UpgradeIndex index = {
	{
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
	},
	{
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
	},
	{
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
	},
	{
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
	},
};
