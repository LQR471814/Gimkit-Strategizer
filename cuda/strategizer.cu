#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "types.hpp"
#include "CLI11.hpp"

#include "misc.hpp"
#include "pinned_memory.hpp"

__host__ __device__ struct GoalResult playGoal(UpgradeLevel **data, PlayState s, Money goal, int giveup)
{
	if (s.money >= goal) {
		return GoalResult{0, s.money};
	}

	float mq = data[MONEY_PER_QUESTION][s.stats.moneyPerQuestion].value;
	float sb = data[STREAK_BONUS][s.stats.streakBonus].value;
	float mu = data[MULTIPLIER][s.stats.multiplier].value;
	// float in = data[INSURANCE][s.stats.insurance].value;

	float a = mu*sb;
	float b = -mu*(sb - 2*mq);
	float c = 2*(s.money-goal);

	float problems = ceilf(
		(-b + sqrtf(pow(b, 2) - 4*a*c)) / (2*a)
	);

	Money money = s.money + (
		mu*problems * (
			2*mq + sb*(problems - 1)
		)
	) / 2;

	return GoalResult{int(problems), money};
}

__forceinline__ __host__ __device__ struct GoalResult playUpgrade(UpgradeLevel **data, PlayState s, int target, int giveup)
{
	int goal = data[target][getStat(s.stats, target) + 1].cost;
	GoalResult result = playGoal(data, s, goal, giveup);
	result.newMoney -= goal;

	return result;
}

struct std::vector<Permutation> permuteRecursive(PermuteContext *c, PermuteState r, int depth)
{
	if (depth == (*c).max)
	{
		Permutation p = Permutation{0, r.sequence, r.play};
		return std::vector<Permutation>{p};
	}

	std::vector<Permutation> permutes;
	for (int u : (*c).upgrades)
	{
		PlayState lowerState = r.play;
		GoalResult res = playUpgrade((*c).data, r.play, u, -1);
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
	printf(" -> Current Min %d Target %d\n", stack[depth].currentMin, stack[depth].minTarget);
}

__forceinline__ __host__ __device__ int iterativeCall(PlayStackFrame *stack, PlayStackParameters params, int depth)
{
	depth++;
	stack[depth].branch = 0;
	stack[depth].params = params;
	return depth;
}

__forceinline__ __host__ __device__ int iterativeReturn(PlayStackFrame *stack, int depth, int value)
{
	depth--;
	if (value < stack[depth].currentMin || stack[depth].currentMin < 0) {
		stack[depth].currentMin = value;
		stack[depth].minTarget = stack[depth].branch;
	}

	stack[depth].branch++;
	return depth;
}

__host__ __device__ int playIterative(ComputeContext *c, PlayState play, PlayStackFrame *stack, int *result, int startOffset)
{
	int depth = 0;
	stack[depth].params.state = play;

	//? To prevent crashes when the initial moneyValue is already larger than the goal
	if (stack[depth].params.state.money >= (*c).moneyGoal) {
		return 0;
	}

	while (true) {
		if (
			(*c).currentMinimum &&
			stack[depth].params.problems >= *(*c).currentMinimum &&
			*(*c).currentMinimum > 0
		) {
			depth = iterativeReturn(
				stack, depth,
				stack[depth].params.problems + 9999
			);
			continue;
		}

		if (stack[depth].params.state.money >= (*c).moneyGoal) {
			depth = iterativeReturn(
				stack, depth,
				stack[depth].params.problems
			);
			continue;
		}

		if (stack[depth].branch == (*c).upgradesSize) {
			result[startOffset + depth] = stack[depth].minTarget;
			if (depth == 0) {
				return stack[depth].currentMin;
			}

			depth = iterativeReturn(
				stack, depth,
				stack[depth].currentMin
			);
			continue;
		}

		if (depth == (*c).max) {
			GoalResult res = playGoal(
				(*c).data,
				stack[depth].params.state,
				(*c).moneyGoal,
				stack[depth].params.upperMinimum
			);

			depth = iterativeReturn(
				stack, depth,
				stack[depth].params.problems + res.problems
			);
			continue;
		}

		if (getStat(
			stack[depth].params.state.stats,
			(*c).upgrades[stack[depth].branch]
		)+1 == MAX_LEVEL) {
			depth = iterativeCall(stack, stack[depth].params, depth);
			continue;
		}

		GoalResult res = playUpgrade(
			(*c).data, stack[depth].params.state,
			(*c).upgrades[stack[depth].branch],
			stack[depth].params.upperMinimum
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
			stack[depth].params.problems + res.problems,
			stack[depth].currentMin
		}, depth);
	}
}

void assignVecToPointer(std::vector<int> vec, int *result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = vec[i];
	}
}

UpgradeLevel* allocUpgradeLevels(std::vector<UpgradeLevel> levels) {
	UpgradeLevel *result;
	cudaMallocManaged(&result, sizeof(UpgradeLevel) * levels.size());
	for (int i = 0; i < levels.size(); i++) {
		result[i] = levels[i];
	}
	return result;
}

std::vector<Permutation> getRoots(UpgradeLevel **data, std::vector<int> upgrades, int syncDepth) {
	PermuteContext c = {data, upgrades, syncDepth};
	PermuteState r = {
		UpgradeStats{0, 0, 0, 0}, // play
		0, 0, NULL,      		  // play
		std::vector<int>{},		  // sequence
	};

	return permuteRecursive(&c, r, 0);
}

struct UpgradeLevel** initializeIndex() {
	UpgradeLevel **data;
	cudaMallocManaged(&data, sizeof(UpgradeLevel) * UPGRADE_COUNT);

	data[MONEY_PER_QUESTION] = allocUpgradeLevels(moneyPerQuestionLevels);
	data[STREAK_BONUS] = allocUpgradeLevels(streakBonusLevels);
	data[MULTIPLIER] = allocUpgradeLevels(multiplierLevels);
	data[INSURANCE] = allocUpgradeLevels(insuranceLevels);

	return data;
}

void deallocateIndex(UpgradeLevel **data) {
	for (int i = 0; i < UPGRADE_COUNT; i++)
		cudaFree(data[i]);
	cudaFree(data);
}

int* initializeSequence(std::vector<int> init, int targetSize) {
	int *sequence;
	cudaMallocManaged(&sequence, sizeof(int) * targetSize);

	for (int i = 0; i < targetSize; i++) {
		sequence[i] = -1;
		if (i < init.size()) {
			sequence[i] = init[i];
		}
	}

	return sequence;
}

int* initializeUpgrades(std::vector<int> init) {
	int *upgrades;
	cudaMallocManaged(&upgrades, sizeof(int) * init.size());
	for (int i = 0; i < init.size(); i++) {
		upgrades[i] = init[i];
	}

	return upgrades;
}

PlayStackFrame* initializeStack(int lowerDepth, int upgradesSize) {
	PlayStackFrame *stack;
	cudaMallocManaged(&stack, sizeof(PlayStackFrame) * (lowerDepth+1));
	for (int i = 0; i < lowerDepth+1; i++) {
		stack[i] = {};
	}

	return stack;
}

__global__ void computeStrategy(int *progress, ComputeContext *c, TComputeStates *results, int rootSize, int depth)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= rootSize) {
		return;
	}

	curand_init(1234, index, 0, results[index].init.randState);
	int problems = playIterative(
		c, results[index].init,
		results[index].stack,
		results[index].sequence, depth
	);

	results[index].problems += problems;

	if (progress)
		atomicAdd(progress, 1);
}

ComputeContext* initializeContext(std::vector<int> upgrades, Money moneygoal, ComputeOptions options) {
	struct UpgradeLevel **data = initializeIndex();
	int *recurseUpgrades = initializeUpgrades(upgrades);

	int lowerDepth = options.maxDepth - options.syncDepth;
	int *globalMin;
	cudaMallocManaged(&globalMin, sizeof(int));
	*globalMin = -1;

	ComputeContext c = {
		data,
		lowerDepth,
		moneygoal,
		recurseUpgrades,
		static_cast<int>(upgrades.size()),
		globalMin,
	};

	ComputeContext *rc = NULL;
	cudaMallocManaged(&rc, sizeof(ComputeContext));
	*rc = c;

	return rc;
}

void deallocateContext(ComputeContext *rc) {
	cudaFree(rc->upgrades);
	cudaFree(rc->currentMinimum);
	deallocateIndex(rc->data);
	cudaFree(rc);
}

TComputeStates* initializeThreadStates(std::vector<Permutation> roots, int upgrades, ComputeOptions opts) {
	TComputeStates *results;
	cudaMallocManaged(&results, sizeof(TComputeStates) * roots.size());

	for (int i = 0; i < roots.size(); i++) {
		int *sequence = initializeSequence(roots[i].sequence, opts.maxDepth);
		PlayStackFrame *stack = initializeStack(opts.maxDepth - opts.syncDepth, upgrades);

		curandState *gen = NULL;
		cudaMallocManaged(&gen, sizeof(curandState));

		roots[i].play.randState = gen;
		results[i] = TComputeStates{
			roots[i].play,
			stack,
			roots[i].problems,
			sequence
		};
	}

	return results;
}

int compute(std::vector<int> upgrades, Money moneyGoal, int *output, ComputeOptions opts) {
	// --> Initialize Roots / Compute States
	ComputeContext *rc = initializeContext(upgrades, moneyGoal, opts);
	std::vector<Permutation> roots = getRoots((*rc).data, upgrades, opts.syncDepth);
	TComputeStates* states = initializeThreadStates(roots, upgrades.size(), opts);
	printf("Memory Allocation Succeeded\n");

	int threadBlocks = ceil(float(roots.size()) / float(BLOCK_SIZE));
	printf("Blocksize %d\n", BLOCK_SIZE);
	printf("Roots %zd Blocks %d\n", roots.size(), threadBlocks);

	// --> Initialize progress and gpu compute control
	int *running = 0;
	int *progress = NULL, *d_progress = NULL;
	cudaEvent_t start, stop;

	printGPUInfo();
	progress = createHostPointer<int>(0);
	d_progress = createPinnedPointer<int>(progress);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// --> Run
	computeStrategy<<<threadBlocks, BLOCK_SIZE>>>(
		d_progress, rc, states, roots.size(), opts.syncDepth
	);

	// --> Report progress
	cudaEventRecord(stop);
	int bufProgress = 0;
	int trueProgress = 0;
	do {
		cudaEventQuery(stop);
		trueProgress = *progress;
		if (trueProgress - bufProgress >= roots.size() * opts.loggingFidelity) {
			printf("Progress %d / %zd\n", bufProgress, roots.size());
			bufProgress = trueProgress;
		}
	} while (trueProgress < roots.size());

	// --> Report run performance
	cudaEventSynchronize(stop);
	printf("\nCompute Status (ignore if there is no visible error) %s\n", cudaGetErrorString(cudaGetLastError()));

	float *elapsed = new float;
	cudaEventElapsedTime(elapsed, start, stop);
	printf("Completed in %fs\n", (*elapsed) / 1000);
	delete elapsed;

	// --> Sort results
	int min = -1;
	for (int i = 0; i < roots.size(); i++) {
		if (min < 0 || states[i].problems < min) {
			min = states[i].problems;
			for (int x = 0; x < opts.maxDepth; x++) {
				output[x] = states[i].sequence[x];
			}
		}

		cudaFree(states[i].init.randState);
		cudaFree(states[i].sequence);
		cudaFree(states[i].stack);
	}

	// --> Cleanup
	cudaFree(states);
	deallocateContext(rc);
	return min;
}

int main(int argc, char** argv)
{
	CLI::App app{"A program that simulates many, many gimkit games"};

	Money moneyGoal = 1000000;
	app.add_option<Money, double>(
		"-g,--goal",
		moneyGoal,
		"Amount of money to reach before stopping"
	);

	unsigned int syncDepth = 2;
	app.add_option<unsigned int>(
		"-r,--roots",
		syncDepth,
		"The depth to recurse synchronously to (threads spawned = <amount of upgrades>^depth) (overrides block count)"
	);

	unsigned int maxDepth = 5;
	app.add_option<unsigned int>(
		"-d,--depth",
		maxDepth,
		"The amount of upgrades to be purchased"
	);

	float loggingFidelity = 0.05;
	app.add_option<float>(
		"-l,--logging-fidelity",
		loggingFidelity,
		"The fidelity in which progress is reported (smaller makes progress update more frequently)"
	);

	CLI11_PARSE(app, argc, argv);

	std::vector<int> upgrades = {
		MONEY_PER_QUESTION,
		STREAK_BONUS,
		MULTIPLIER,
	};

	ComputeOptions opts = {
		syncDepth,
		maxDepth,
		loggingFidelity,
	};

	int min = 0;
	int *result = new int[maxDepth];
	min = compute(upgrades, moneyGoal, result, opts);

	printf("========== RESULTS ==========\n");
	printf("Minimum Problems: %d\n", min);
	printf("Sequence Required: ");
	for (int i = 0; i < maxDepth; i++) {
		printf("%d ", result[i]);
	}

	return 0;
}
