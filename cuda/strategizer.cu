#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "types.h"

__host__ __device__ UpgradeStats incrementStat(UpgradeStats s, int id)
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

__host__ __device__ int getStat(UpgradeStats s, int id)
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

__host__ __device__ struct UpgradeLevel getUpgrade(UpgradeIndex *data, int id, int level)
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

__host__ __device__ struct GoalResult playGoal(UpgradeIndex *data, PlayState s, int goal)
{
	int streak = 0;
	int problems = 0;
	int money = s.money;

	UpgradeLevel mq = (*data).moneyPerQuestion[s.stats.moneyPerQuestion];
	UpgradeLevel sb = (*data).streakBonus[s.stats.streakBonus];
	UpgradeLevel mu = (*data).multiplier[s.stats.multiplier];
	UpgradeLevel in = (*data).insurance[s.stats.insurance];

	while (money < goal)
	{
	#ifdef __CUDA_ARCH__
		float i = curand_uniform(s.randState);
	#else
		float i = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	#endif
		if (i < s.setbackChance)
		{
			money -= (mq.value * mu.value) - (mq.value * mu.value) * in.value / 100;
		}
		else
		{
			money += mu.value * (mq.value + sb.value * streak);
			streak++;
		}

		problems++;
	};

	return GoalResult{problems, money};
}

__host__ __device__ struct GoalResult playUpgrade(UpgradeIndex *data, PlayState s, int target)
{
	if (target > MAX_LEVEL)
	{
		return GoalResult{0, s.money};
	};

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

__host__ __device__ int playRecursive(RecurseContext *c, RecurseState r, int *result, int depth)
{
	if (r.play.money >= (*c).moneyGoal)
	{
		return 0;
	};

	if (depth == (*c).max)
	{
		GoalResult res = playGoal((*c).data, r.play, (*c).moneyGoal);
		return res.problems;
	};

	int min = -1, minTarget = -1;
	for (int i = 0; i < (*c).upgradesSize; i++)
	{
		PlayState lowerStats;
		GoalResult res = playUpgrade((*c).data, r.play, (*c).upgrades[i]);

		lowerStats.money = res.newMoney;
		lowerStats.stats = incrementStat(r.play.stats, (*c).upgrades[i]);

		int lowerProblems = res.problems + playRecursive(
			c, RecurseState{lowerStats, i},
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

__global__ void computeStrategy(RecurseContext *c, TRecurseResult *results, int rootSize, int depth)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index > rootSize) {
		return;
	};

	results[index].problems = playRecursive(
		c, results[index].init, results[index].sequence, depth
	);
}

void assignVecToPointer(std::vector<int> vec, int *result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = vec[i];
	};
}

std::vector<Permutation> getRoots(std::vector<int> upgrades, int syncDepth) {
	PermuteContext c = {&index, upgrades, syncDepth};
	PermuteState r = {
		UpgradeStats{0, 0, 0, 0}, // play
		0, 0, NULL,			      // play
		std::vector<int>{},		  // sequence
	};

	return permuteRecursive(&c, r, 0);
}

int computeSync(std::vector<int> upgrades, int moneyGoal, int syncDepth, int maxDepth, int *result) {
	std::vector<Permutation> roots = getRoots(upgrades, syncDepth);
	int upgradeCount = upgrades.size();
	int *recurseUpgrades = new int[upgradeCount];
	assignVecToPointer(upgrades, recurseUpgrades, upgradeCount);

	RecurseContext rc = {
		&index,
		maxDepth,
		moneyGoal,
		recurseUpgrades,
		upgradeCount,
	};

	int min = -1;
	for (Permutation p : roots)
	{
		for (int u : upgrades)
		{
			RecurseState rs = {p.play, u};

			int *recurseResult = new int[maxDepth];
			for (int i = 0; i < p.sequence.size(); i++)
			{
				recurseResult[i] = p.sequence[i];
			};

			int problems = p.problems + playRecursive(&rc, rs, recurseResult, syncDepth);

			if (min < 0 || problems < min) {
				min = problems;
				std::copy(recurseResult, recurseResult + maxDepth, result);
			} else {
				delete[] recurseResult;
			};
		}
	};

	return min;
}

int computeThreaded(std::vector<int> upgrades, int moneyGoal, int syncDepth, int maxDepth, int *output) {
	std::vector<Permutation> roots = getRoots(upgrades, syncDepth);
	int upgradeCount = upgrades.size();
	int *recurseUpgrades = new int[upgradeCount];
	assignVecToPointer(upgrades, recurseUpgrades, upgradeCount);

	RecurseContext rc = {
		&index,
		maxDepth,
		moneyGoal,
		recurseUpgrades,
		upgradeCount,
	};

	int threads = 256;
	int threadBlocks = roots.size() / threads;
	if (threadBlocks < 1) {
		threadBlocks = 1;
	};

	TRecurseResult *results;
	cudaMallocManaged(&results, sizeof(TRecurseResult) * roots.size() * upgradeCount);

	for (int u = 0; u < upgradeCount; u++) {
		for (int i = 0; i < roots.size(); i++) {
			int index = sizeof(TRecurseResult) * (u * roots.size() + i);

			int *sequence = NULL;
			cudaMallocManaged(&sequence, maxDepth * sizeof(int));

			curandState *rState;
			cudaMalloc(&rState, sizeof(curandState));

			PlayState playCtx = roots[index].play;
			playCtx.randState = rState;

			results[index] = TRecurseResult{
				RecurseState{playCtx, u},
				roots[index].problems,
				sequence
			};
		};
	}

	computeStrategy<<<threadBlocks, threads>>>(
		&rc, results, roots.size(), syncDepth
	);

	cudaDeviceSynchronize();

	int min = -1;
	for (int i = 0; i < roots.size() * 3; i++) {
		if (min < 0 || results[i].problems < min) {
			min = results[i].problems;
			std::copy(results[i].sequence, results[i].sequence + maxDepth, output);
		};
	};

	cudaFree(results);
	return min;
}

int
main(void)
{
	int syncDepth = 2;
	int maxDepth = 5;

	std::vector<int> upgrades = {
		MONEY_PER_QUESTION,
		STREAK_BONUS,
		MULTIPLIER};

	int *result = new int[maxDepth];
	// int min = computeSync(upgrades, 1000, syncDepth, maxDepth, result);
	int min = computeThreaded(upgrades, 1000, syncDepth, maxDepth, result);

	printf("Minimum Problems: %d\n", min);
	printf("Sequence Required: ");
	for (int i = 0; i < maxDepth; i++) {
		printf("%d ", result[i]);
	};

	return 0;
}
