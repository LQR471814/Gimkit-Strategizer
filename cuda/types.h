#include <curand.h>
#include <vector>

struct GoalResult
{
	int problems;
	float newMoney;
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
	float money;
	curandState *randState;
};

struct Permutation
{
	int problems;
	std::vector<int> sequence;
	struct PlayState play;
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
	float moneyGoal;

	int *upgrades;
	int upgradesSize;
};

struct TRecurseResult {
	struct PlayState init;
	int problems;
	int *sequence;
};

struct UpgradeLevel
{
	float value;
	int cost;
};

const int MONEY_PER_QUESTION = 0;
const int STREAK_BONUS = 1;
const int MULTIPLIER = 2;
const int INSURANCE = 3;

struct UpgradeIndex
{
	int MAX_LEVEL;
	UpgradeLevel *moneyPerQuestion;
	UpgradeLevel *streakBonus;
	UpgradeLevel *multiplier;
	UpgradeLevel *insurance;
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

struct UpgradeIndex index = {10};
