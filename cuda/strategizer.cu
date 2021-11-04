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

__host__ __device__ void printPlayStack(PlayStackFrame *stack, int depth, int resultLength) {
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

__forceinline__ __host__ __device__ int iterativeCall(PlayStackFrame *stack, PlayStackParameters params, int depth)
{
	depth++;
	stack[depth].branch = 0;
	stack[depth].params = params;
	return depth;
}

__forceinline__ __host__ __device__ int iterativeReturn(PlayStackFrame *stack, int depth, int value) {
	depth--;
	stack[depth].results[stack[depth].branch] = value;
	stack[depth].branch++;
	return depth;
}

__host__ __device__ int playIterative(RecurseContext *c, PlayState play, PlayStackFrame *stack, int *result, int startOffset)
{
	int depth = 0;
	stack[depth].params.state = play;

	//? To prevent crashes when the initial moneyValue is already larger than the goal
	if (stack[depth].params.state.money >= (*c).moneyGoal) {
		return 0;
	};

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

void deallocateIndex(UpgradeIndex *index) {
	cudaFree((*index).moneyPerQuestion);
	cudaFree((*index).streakBonus);
	cudaFree((*index).multiplier);
	cudaFree((*index).insurance);
	cudaFree(index);
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

PlayStackFrame* initializeStack(int lowerDepth, int upgradesSize) {
	PlayStackFrame *stack;
	cudaMallocManaged(&stack, sizeof(PlayStackFrame) * (lowerDepth+1));
	for (int i = 0; i < lowerDepth+1; i++) {
		int *results;
		cudaMallocManaged(&results, sizeof(int) * upgradesSize);

		stack[i] = {};
		stack[i].results = results;
	};

	return stack;
}

void deallocateStack(PlayStackFrame* stack, int lowerDepth, int upgradesSize) {
	for (int i = 0; i < lowerDepth+1; i++) {
		cudaFree(stack[i].results);
	};
	cudaFree(stack);
}

__global__ void computeStrategy(RecurseContext *c, TRecurseResult *results, int rootSize, int depth)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= rootSize) {
		return;
	};

	curand_init(1234, index, 0, results[index].init.randState);
	int problems = playIterative(
		c, results[index].init,
		results[index].stack,
		results[index].sequence, depth
	);

	results[index].problems = problems;
}

int computeSync(std::vector<int> upgrades, Money moneyGoal, int syncDepth, int maxDepth, int *result) {
	struct UpgradeIndex *data = initializeIndex();

	std::vector<Permutation> roots = getRoots(data, upgrades, syncDepth);
	int *recurseUpgrades = initializeUpgrades(upgrades);
	int lowerDepth = maxDepth - syncDepth;

	RecurseContext rc = {
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
		PlayStackFrame *stack = initializeStack(lowerDepth, upgrades.size());

		int problems = p.problems + playIterative(&rc, p.play, stack, recurseResult, syncDepth);
		deallocateStack(stack, lowerDepth, upgrades.size());

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

		cudaFree(recurseResult);
	};

	cudaFree(recurseUpgrades);
	return min;
}

int computeThreaded(std::vector<int> upgrades, Money moneyGoal, int syncDepth, int maxDepth, int *output) {
	struct UpgradeIndex *data = initializeIndex();
	std::vector<Permutation> roots = getRoots(data, upgrades, syncDepth);
	int *recurseUpgrades = initializeUpgrades(upgrades);

	int lowerDepth = maxDepth - syncDepth;

	RecurseContext c = {
		data,
		lowerDepth,
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
		PlayStackFrame *stack = initializeStack(lowerDepth, upgrades.size());

		curandState *gen = NULL;
		cudaMallocManaged(&gen, sizeof(curandState));

		roots[i].play.randState = gen;
		results[i] = TRecurseResult{
			roots[i].play,
			stack,
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
	printf("Compute Status %s\n", cudaGetErrorString(err));

	int min = -1;
	for (int i = 0; i < roots.size(); i++) {
		int problems = roots[i].problems + results[i].problems;

		if (min < 0 || problems < min) {
			min = problems;
			for (int x = 0; x < maxDepth; x++) {
				output[x] = results[i].sequence[x];
			};
		};

		cudaFree(results[i].init.randState);
		cudaFree(results[i].sequence);
		deallocateStack(results[i].stack, lowerDepth, upgrades.size());
	};

	cudaFree(results);
	cudaFree(recurseUpgrades);
	cudaFree(rc);
	deallocateIndex(data);
	return min;
}

int main()
{
	int syncDepth = 7;
	int maxDepth = 10;
	// Money moneyGoal = 1000000000000; //? First to a trillion
	Money moneyGoal = 10000000;

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
