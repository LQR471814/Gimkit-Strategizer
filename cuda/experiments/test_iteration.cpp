#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "types_cpp.h"

void printPlayState(PlayState p) {
	printf(
		"$%f Stats %d %d %d %d\n",
		(Money)p.money,
		p.stats.moneyPerQuestion,
		p.stats.streakBonus,
		p.stats.multiplier,
		p.stats.insurance
	);
}

UpgradeStats incrementStat(UpgradeStats s, int id)
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

int getStat(UpgradeStats s, int id)
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

struct UpgradeLevel getUpgrade(UpgradeIndex *data, int id, int level)
{
	switch (id)
	{
	case MONEY_PER_QUESTION:
		return (*data).moneyPerQuestion[level];
	case STREAK_BONUS:
		return (*data).streakBonus[level];
	case MULTIPLIER:
		return (*data).multiplier[level];
	case INSURANCE:
		return (*data).insurance[level];
	}

	return UpgradeLevel{};
}

struct GoalResult playGoal(UpgradeIndex *data, PlayState s, Money goal)
{
	float mq = (*data).moneyPerQuestion[s.stats.moneyPerQuestion].value;
	float sb = (*data).streakBonus[s.stats.streakBonus].value;
	float mu = (*data).multiplier[s.stats.multiplier].value;
	float in = (*data).insurance[s.stats.insurance].value;

	double a = mu*sb;
	double b = -mu*(sb - 2*mq);
	double c = 2*(s.money-goal);

	double problems = ceilf(
		(-b + sqrtf(pow(b, 2) - 4*a*c)) / (2*a)
	);

	Money money = s.money + (
		mu*problems * (
			2*mq + sb*(problems - 1)
		)
	) / 2;

	return GoalResult{int(problems), money};
}

struct GoalResult playUpgrade(UpgradeIndex *data, PlayState s, int target)
{
	int goal = getUpgrade(data, target, getStat(s.stats, target) + 1).cost;
	GoalResult result = playGoal(data, s, goal);
	result.newMoney -= goal;

	return result;
}

struct std::vector<Permutation> permuteRecursive(PermuteContext *c, PermuteState r, int depth)
{
	if (depth == (*c).max)
	{
		Permutation p = Permutation{0, r.sequence, r.play};
		return std::vector<Permutation>{p};
	};

	std::vector<Permutation> permutes;
	for (int u : (*c).upgrades)
	{
		PlayState lowerState = r.play;
		GoalResult res = playUpgrade((*c).data, r.play, u);
		lowerState.money = res.newMoney;
		lowerState.stats = incrementStat(r.play.stats, u);

		std::vector<int> lowerSequence;
		lowerSequence = r.sequence;
		lowerSequence.push_back(u);

		std::vector<Permutation> results = permuteRecursive(
			c, PermuteState{
				lowerState,
				lowerSequence,
			},
			depth + 1);

		for (Permutation p : results)
		{
			p.problems += res.problems;
			permutes.push_back(p);
		}
	}

	return permutes;
}

struct PlayStackParameters {
	PlayState state = {};
	int problems = 0;
};

struct PlayStackFrame {
	PlayStackParameters params = {};
	int branch = 0;
	int *results = NULL;
};

void printPlayStack(PlayStackFrame *stack, int depth, int resultLength) {
	printf("Depth %d Branch %d\n", depth, stack[depth].branch);
	printf(" Params\n");
	printf(" -> Problems %d\n", stack[depth].params.problems);
	printf(" -> ");
	printPlayState(stack[depth].params.state);
	printf(" -> Results (problems) ");
	for (int i = 0; i < resultLength; i++) {
		printf("%d ", stack[depth].results[i]);
	};
	printf("\n");
}

int iterativeCall(PlayStackFrame *stack, PlayStackParameters params, int depth)
{
	depth++;
	stack[depth].branch = 0;
	stack[depth].params = params;
	return depth;
}

int iterativeReturn(PlayStackFrame *stack, int depth, int value) {
	depth--;
	stack[depth].results[stack[depth].branch] = value;
	stack[depth].branch++;
	return depth;
}

int playIterative(ComputeContext *c, PlayState play, PlayStackFrame *stack, int *result, int startOffset)
{
	int depth = 0;
	stack[depth].params.state = play;

	while (true) {
		if (stack[depth].params.state.money >= (*c).moneyGoal) {
			depth = iterativeReturn(stack, depth, stack[depth].params.problems);
			continue;
		};

		if (stack[depth].branch == (*c).upgradesSize) {
			int min = -1, minTarget = -1;

			for (int i = 0; i < (*c).upgradesSize; i++) {
				if (stack[depth].results[i] < min || min < 0) {
					min = stack[depth].results[i];
					minTarget = i;
				};
			};

			result[startOffset + depth] = minTarget;
			min += stack[depth].params.problems;
			if (depth == 0) {
				return min;
			};

			depth = iterativeReturn(stack, depth, min);
			continue;
		};

		if (depth == (*c).max) {
			GoalResult res = playGoal((*c).data, stack[depth].params.state, (*c).moneyGoal);
			depth = iterativeReturn(stack, depth, stack[depth].params.problems + res.problems);
			continue;
		};

		if (getStat(
			stack[depth].params.state.stats,
			(*c).upgrades[stack[depth].branch]
		)+1 == (*c).data->MAX_LEVEL) {
			depth = iterativeCall(stack, stack[depth].params, depth);
			continue;
		}

		GoalResult res = playUpgrade(
			(*c).data, stack[depth].params.state,
			(*c).upgrades[stack[depth].branch]
		);

		PlayState lowerState = {
			incrementStat(
				stack[depth].params.state.stats,
				(*c).upgrades[stack[depth].branch]
			),
			stack[depth].params.state.setbackChance,
			res.newMoney,
			stack[depth].params.state.randState
		};

		depth = iterativeCall(stack, {
			lowerState,
			res.problems
		}, depth);
	};
}

int playRecursive(ComputeContext *c, PlayState play, int *result, int depth)
{
	if (play.money >= (*c).moneyGoal)
	{
		return 0;
	};

	if (depth == (*c).max)
	{
		GoalResult res = playGoal((*c).data, play, (*c).moneyGoal);
		return res.problems;
	};

	int min = -1, minTarget = -1;
	for (int i = 0; i < (*c).upgradesSize; i++)
	{
		GoalResult res = playUpgrade((*c).data, play, (*c).upgrades[i]);
		PlayState lowerState = {
			incrementStat(play.stats, (*c).upgrades[i]),
			play.setbackChance,
			res.newMoney,
			play.randState
		};

		int lowerProblems = res.problems + playRecursive(
			c, lowerState,
			result, depth + 1
		);

		if (min < 0 || lowerProblems < min)
		{
			min = lowerProblems;
			minTarget = i;
		};
	}

	result[depth] = minTarget;
	return min;
}

void assignVecToPointer(std::vector<int> vec, int *result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = vec[i];
	};
}

void allocUpgradeLevels(UpgradeLevel **results, std::vector<UpgradeLevel> levels) {
	*results = (UpgradeLevel*) malloc(sizeof(UpgradeLevel) * levels.size());
	for (int i = 0; i < levels.size(); i++) {
		(*results)[i] = levels[i];
	};
}

std::vector<Permutation> getRoots(UpgradeIndex *data, std::vector<int> upgrades, int syncDepth) {
	PermuteContext c = {data, upgrades, syncDepth};
	PermuteState r = {
		UpgradeStats{0, 0, 0, 0}, // play
		0, 0, NULL,      		  // play
		std::vector<int>{},		  // sequence
	};

	return permuteRecursive(&c, r, 0);
}

struct UpgradeIndex* initializeIndex() {
	UpgradeIndex *data = (UpgradeIndex*)malloc(sizeof(UpgradeIndex));
	data->MAX_LEVEL = upgradeIndex.MAX_LEVEL;

	allocUpgradeLevels(&(*data).moneyPerQuestion, moneyPerQuestionLevels);
	allocUpgradeLevels(&(*data).streakBonus, streakBonusLevels);
	allocUpgradeLevels(&(*data).multiplier, multiplierLevels);
	allocUpgradeLevels(&(*data).insurance, insuranceLevels);

	return data;
}

int* initializeSequence(std::vector<int> init, int targetSize) {
	int *sequence = new int[targetSize];

	for (int i = 0; i < targetSize; i++) {
		sequence[i] = -1;
		if (i < init.size()) {
			sequence[i] = init[i];
		};
	};

	return sequence;
}

int* initializeUpgrades(std::vector<int> init) {
	int *upgrades = new int[init.size()];
	for (int i = 0; i < init.size(); i++) {
		upgrades[i] = init[i];
	};

	return upgrades;
}

int computeSync(std::vector<int> upgrades, Money moneyGoal, int syncDepth, int maxDepth, int *result) {
	struct UpgradeIndex *data = initializeIndex();
	std::vector<Permutation> roots = getRoots(data, upgrades, syncDepth);
	int *recurseUpgrades = initializeUpgrades(upgrades);

	int lowerDepth = maxDepth - syncDepth;

	ComputeContext rc = {
		data,
		lowerDepth,
		moneyGoal,
		recurseUpgrades,
		static_cast<int>(upgrades.size()),
	};

	printf("Memory Allocation Succeeded\n");
	printf("Roots: %d\n", static_cast<int>(roots.size()));

	int min = -1;
	for (Permutation p : roots)
	{
		int *recurseResult = initializeSequence(p.sequence, maxDepth);
		PlayStackFrame *stack = new PlayStackFrame[lowerDepth + 1];
		for (int i = 0; i <= lowerDepth; i++) {
			int *results = new int[upgrades.size()];
			stack[i] = {};
			stack[i].results = results;
		};

		int problems = p.problems + playIterative(&rc, p.play, stack, recurseResult, syncDepth);
		delete[] stack;

		printf("Problems: %d |", problems);
		for (int i = 0; i < maxDepth; i++) {
			printf(" %d", recurseResult[i]);
		};
		printf("\n");

		if (min < 0 || problems < min) {
			min = problems;
			for (int i = 0; i < maxDepth; i++) {
				result[i] = recurseResult[i];
			};
		};

		delete[] recurseResult;
	};

	delete[] recurseUpgrades;

	return min;
}

int main()
{
	int syncDepth = 2;
	int maxDepth = 15;
	// Money moneyGoal = 1000000000000; //? First to a trillion
	Money moneyGoal = 1000000000;

	std::vector<int> upgrades = {
		MONEY_PER_QUESTION,
		STREAK_BONUS,
		MULTIPLIER,
	};

	int *result = new int[maxDepth];
	int min = computeSync(upgrades, moneyGoal, syncDepth, maxDepth, result);

	printf("========== RESULTS ==========\n");
	printf("Minimum Problems: %d\n", min);
	printf("Sequence Required: ");
	for (int i = 0; i < maxDepth; i++) {
		printf("%d ", result[i]);
	};

	return 0;
}

