#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "types.h"

__host__ __device__ void printPlayState(PlayState p) {
	printf(
		"$%f Stats %d %d %d %d\n",
		(Money)p.money,
		p.stats.moneyPerQuestion,
		p.stats.streakBonus,
		p.stats.multiplier,
		p.stats.insurance
	);
}

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

__host__ __device__ struct GoalResult playGoal(UpgradeIndex *data, PlayState s, Money goal)
{
	int problems = 0;
	float streak = 0;
	Money money = s.money;

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
			money -= (Money) (mq.value * mu.value) - (mq.value * mu.value) * in.value / 100;
		}
		else
		{
			money += (Money) mu.value * (mq.value + sb.value * streak);
			streak++;
		}

		problems++;
	};

	return GoalResult{problems, money};
}

__host__ __device__ struct GoalResult playUpgrade(UpgradeIndex *data, PlayState s, int target)
{
	if (getStat(s.stats, target) > data->MAX_LEVEL)
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

__host__ __device__ int playRecursive(RecurseContext *c, PlayState play, int *result, int depth)
{
	// printf(" -----> Depth %d\n", depth);
	if (play.money >= (*c).moneyGoal)
	{
		// printf("Already done %f\n", play.money);
		return 0;
	};

	if (depth == (*c).max)
	{
		GoalResult res = playGoal((*c).data, play, (*c).moneyGoal);
		// printf("Maxed Depth %d Problems %d\n", depth, res.problems);
		// printPlayState(play);
		return res.problems;
	};

	int min = -1, minTarget = -1;
	for (int i = 0; i < (*c).upgradesSize; i++)
	{
		GoalResult res = playUpgrade((*c).data, play, (*c).upgrades[i]);
		// printf("Start Depth %d Problems %d\n", depth, res.problems);
		PlayState lowerState = {
			incrementStat(play.stats, (*c).upgrades[i]),
			play.setbackChance,
			res.newMoney,
			play.randState
		};

		// if (depth == 2) {
		// 	printf("Depth %d Upgrade Target %d ---------\n", depth, (*c).upgrades[i]);
		// 	printPlayState(play);
		// 	printf(" -> ");
		// 	printPlayState(lowerState);
		// };

		int lowerProblems = res.problems + playRecursive(
			c, lowerState,
			result, depth + 1
		);
		// printf("End Depth %d Problems %d\n", depth, lowerProblems);

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
	if (index >= rootSize) {
		return;
	};

	curand_init(1234, index, 0, results[index].init.randState);
	int problems = playRecursive(
		c, results[index].init, results[index].sequence, depth
	);

	results[index].problems = problems;
}

void assignVecToPointer(std::vector<int> vec, int *result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = vec[i];
	};
}

void allocUpgradeLevels(UpgradeLevel **results, std::vector<UpgradeLevel> levels) {
	cudaMallocManaged(results, sizeof(UpgradeLevel) * levels.size());
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
	UpgradeIndex *data;
	cudaMallocManaged(&data, sizeof(UpgradeIndex));
	data->MAX_LEVEL = index.MAX_LEVEL;

	allocUpgradeLevels(&(*data).moneyPerQuestion, moneyPerQuestionLevels);
	allocUpgradeLevels(&(*data).streakBonus, streakBonusLevels);
	allocUpgradeLevels(&(*data).multiplier, multiplierLevels);
	allocUpgradeLevels(&(*data).insurance, insuranceLevels);

	return data;
}

int* initializeSequence(std::vector<int> init, int targetSize) {
	int *sequence;
	cudaMallocManaged(&sequence, sizeof(int) * targetSize);

	for (int i = 0; i < targetSize; i++) {
		sequence[i] = -1;
		if (i < init.size()) {
			sequence[i] = init[i];
		};
	};

	return sequence;
}

int* initializeUpgrades(std::vector<int> init) {
	int *upgrades;
	cudaMallocManaged(&upgrades, sizeof(int) * init.size());
	for (int i = 0; i < init.size(); i++) {
		upgrades[i] = init[i];
	};

	return upgrades;
}

int computeSync(std::vector<int> upgrades, Money moneyGoal, int syncDepth, int maxDepth, int *result) {
	struct UpgradeIndex *data = initializeIndex();
	std::vector<Permutation> roots = getRoots(data, upgrades, syncDepth);
	int *recurseUpgrades = initializeUpgrades(upgrades);

	RecurseContext rc = {
		data,
		maxDepth,
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
		int problems = p.problems + playRecursive(&rc, p.play, recurseResult, syncDepth);

		// printf("Problems: %d |", problems);
		// for (int i = 0; i < maxDepth; i++) {
		// 	printf(" %d", recurseResult[i]);
		// };
		// printf("\n");

		if (min < 0 || problems < min) {
			min = problems;
			for (int i = 0; i < maxDepth; i++) {
				result[i] = recurseResult[i];
			};
		} else {
			cudaFree(recurseResult);
		};
	};

	return min;
}

int computeThreaded(std::vector<int> upgrades, Money moneyGoal, int syncDepth, int maxDepth, int *output) {
	struct UpgradeIndex *data = initializeIndex();
	std::vector<Permutation> roots = getRoots(data, upgrades, syncDepth);
	int *recurseUpgrades = initializeUpgrades(upgrades);

	RecurseContext c = {
		data,
		maxDepth,
		moneyGoal,
		recurseUpgrades,
		static_cast<int>(upgrades.size()),
	};

	RecurseContext *rc = NULL;
	cudaMallocManaged(&rc, sizeof(RecurseContext));
	*rc = c;

	TRecurseResult *results;
	cudaMallocManaged(&results, sizeof(TRecurseResult) * roots.size());

	for (int i = 0; i < roots.size(); i++) {
		int *sequence = initializeSequence(roots[i].sequence, maxDepth);
		curandState *gen = NULL;
		cudaMallocManaged(&gen, sizeof(curandState));

		roots[i].play.randState = gen;
		results[i] = TRecurseResult{
			roots[i].play,
			roots[i].problems,
			sequence
		};
	};

	printf("Memory Allocation Succeeded\n");

	int threads = 256;
	int threadBlocks = roots.size() / threads;
	if (threadBlocks < 1) {
		threadBlocks = 1;
	};

	printf("Roots %zd Blocks %d\n", roots.size(), threadBlocks);
	computeStrategy<<<threadBlocks, threads>>>(
		rc, results, roots.size(), syncDepth
	);

	cudaError_t err = cudaDeviceSynchronize();
	printf("Compute Status %s\n", cudaGetErrorName(err));

	int min = -1;
	for (int i = 0; i < roots.size(); i++) {
		int problems = roots[i].problems + results[i].problems;

		// printf("Problems: %d |", problems);
		// for (int x = 0; x < maxDepth; x++) {
		// 	printf(" %d", results[i].sequence[x]);
		// };
		// printf("\n");

		if (min < 0 || problems < min) {
			min = problems;
			for (int x = 0; x < maxDepth; x++) {
				output[x] = results[i].sequence[x];
			};
		};
	};

	cudaFree(results);
	return min;
}

int main()
{
	int syncDepth = 7;
	int maxDepth = 12;
	// Money moneyGoal = 1000000000000; //? First to a trillion
	Money moneyGoal = 1000;

	std::vector<int> upgrades = {
		MONEY_PER_QUESTION,
		STREAK_BONUS,
		MULTIPLIER};

	int *result = new int[maxDepth];
	// int min = computeSync(upgrades, moneyGoal, syncDepth, maxDepth, result);
	int min = computeThreaded(upgrades, moneyGoal, syncDepth, maxDepth, result);

	printf("========== RESULTS ==========\n");
	printf("Minimum Problems: %d\n", min);
	printf("Sequence Required: ");
	for (int i = 0; i < maxDepth; i++) {
		printf("%d ", result[i]);
	};

	return 0;
}
