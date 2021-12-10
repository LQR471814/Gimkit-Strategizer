#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "types.hpp"
#include "CLI11.hpp"

#include "misc.hpp"
#include "pinned_memory.hpp"

__host__ __device__ struct GoalResult playGoal(UpgradeLevel **data, PlayState s, Money goal, Minimum giveup)
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

	return GoalResult{uint32_t(problems), money};
}

__forceinline__ __host__ __device__ struct GoalResult playUpgrade(UpgradeLevel **data, PlayState s, UpgradeId target, Minimum giveup)
{
	uint32_t goal = data[target][getStat(s.stats, target) + 1].cost;
	GoalResult result = playGoal(data, s, goal, giveup);
	result.newMoney -= goal;

	return result;
}

struct std::vector<Permutation> permuteRecursive(PermuteContext *c, PermuteState r, Depth depth)
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

		std::vector<UpgradeId> lowerSequence;
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
	printf(" -> Current Min %lld Target %d\n", stack[depth].currentMin, stack[depth].minTarget);
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

__host__ __device__ int playIterative(
	ComputeContext *c,
	PlayState play,
	PlayStackFrame *stack,
	UpgradeId *result,
	Depth startOffset,
	Depth* depth
) {
	stack[*depth].params.state = play;

	//? To prevent crashes when the initial moneyValue is already larger than the goal
	if (stack[*depth].params.state.money >= (*c).moneyGoal) {
		return 0;
	}

	while (true) {
		if (*(*c).running == 1) {
			return 0;
		}

		if (
			(*c).currentMinimum &&
			stack[*depth].params.problems >= *(*c).currentMinimum &&
			*(*c).currentMinimum > 0
		) {
			*depth = iterativeReturn(
				stack, *depth,
				stack[*depth].params.problems + 9999
			);
			continue;
		}

		if (stack[*depth].params.state.money >= (*c).moneyGoal) {
			*depth = iterativeReturn(
				stack, *depth,
				stack[*depth].params.problems
			);
			continue;
		}

		if (stack[*depth].branch == (*c).upgradesSize) {
			result[startOffset + *depth] = stack[*depth].minTarget;
			if (*depth == 0) {
				return stack[*depth].currentMin;
			}

			*depth = iterativeReturn(
				stack, *depth,
				stack[*depth].currentMin
			);
			continue;
		}

		if (*depth == (*c).max) {
			GoalResult res = playGoal(
				(*c).data,
				stack[*depth].params.state,
				(*c).moneyGoal,
				stack[*depth].params.upperMinimum
			);

			*depth = iterativeReturn(
				stack, *depth,
				stack[*depth].params.problems + res.problems
			);
			continue;
		}

		if (getStat(
			stack[*depth].params.state.stats,
			(*c).upgrades[stack[*depth].branch]
		)+1 == MAX_LEVEL) {
			*depth = iterativeCall(stack, stack[*depth].params, *depth);
			continue;
		}

		GoalResult res = playUpgrade(
			(*c).data, stack[*depth].params.state,
			(*c).upgrades[stack[*depth].branch],
			stack[*depth].params.upperMinimum
		);

		PlayState lowerState = {
			incrementStat(
				stack[*depth].params.state.stats,
				(*c).upgrades[stack[*depth].branch]
			),
			stack[*depth].params.state.setbackChance,
			res.newMoney,
			stack[*depth].params.state.randState
		};

		*depth = iterativeCall(stack, {
			lowerState,
			stack[*depth].params.problems + res.problems,
			stack[*depth].currentMin
		}, *depth);
	}
}

template <typename T>
void assignVecToPointer(std::vector<T> vec, T *result, uint64_t size) {
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

std::vector<Permutation> getRoots(UpgradeLevel **data, std::vector<UpgradeId> upgrades, Depth syncDepth) {
	PermuteContext c = {data, upgrades, syncDepth};
	PermuteState r = {
		UpgradeStats{0, 0, 0, 0}, // play
		0, 0, NULL,      		  // play
		std::vector<UpgradeId>{},		  // sequence
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

UpgradeId* initializeSequence(std::vector<UpgradeId> init, int targetSize) {
	UpgradeId *sequence;
	cudaMallocManaged(&sequence, sizeof(int) * targetSize);

	for (int i = 0; i < targetSize; i++) {
		sequence[i] = -1;
		if (i < init.size()) {
			sequence[i] = init[i];
		}
	}

	return sequence;
}

UpgradeId* initializeUpgrades(std::vector<UpgradeId> init) {
	UpgradeId *upgrades;
	cudaMallocManaged(&upgrades, sizeof(UpgradeId) * init.size());
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

__global__ void computeStrategy(int *progress, ComputeContext *c, TComputeState *states, int rootSize, int offset)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= rootSize) {
		return;
	}

	curand_init(1234, index, 0, states[index].init.randState);
	uint32_t problems = playIterative(
		c, states[index].init,
		states[index].stack,
		states[index].sequence,
		offset, states[index].depth
	);

	states[index].problems += problems;

	if (progress)
		atomicAdd(progress, 1);
}

ComputeContext* initializeContext(std::vector<UpgradeId> upgrades, Money moneygoal, ComputeOptions options) {
	struct UpgradeLevel **data = initializeIndex();
	UpgradeId *recurseUpgrades = initializeUpgrades(upgrades);

	Depth lowerDepth = options.maxDepth - options.syncDepth;
	Minimum *globalMin;
	cudaMallocManaged(&globalMin, sizeof(Minimum));
	*globalMin = -1;

	ComputeContext c = {
		data,
		lowerDepth,
		moneygoal,
		recurseUpgrades,
		static_cast<MaxUpgradeLevel>(upgrades.size()),
		globalMin,
		NULL,
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

TComputeState* initializeThreadStates(std::vector<Permutation> roots, MaxUpgradeLevel upgrades, ComputeOptions opts) {
	TComputeState *results;
	cudaMallocManaged(&results, sizeof(TComputeState) * roots.size());

	for (int i = 0; i < roots.size(); i++) {
		UpgradeId *sequence = initializeSequence(roots[i].sequence, opts.maxDepth);
		PlayStackFrame *stack = initializeStack(opts.maxDepth - opts.syncDepth, upgrades);

		curandState *gen = NULL;
		cudaMallocManaged(&gen, sizeof(curandState));

		Depth* depth;
		cudaMallocManaged(&depth, sizeof(Depth));

		roots[i].play.randState = gen;
		results[i] = TComputeState{
			roots[i].play,
			roots[i].problems,
			depth,
			stack,
			sequence
		};
	}

	return results;
}

ProblemCount compute(std::vector<UpgradeId> upgrades, Money moneyGoal, UpgradeId *output, ComputeOptions opts) {
	// --> Initialize Roots / Compute States
	ComputeContext *rc = initializeContext(upgrades, moneyGoal, opts);
	std::vector<Permutation> roots = getRoots((*rc).data, upgrades, opts.syncDepth);
	TComputeState* states = initializeThreadStates(roots, upgrades.size(), opts);
	printf("Memory Allocation Succeeded\n");

	int threadBlocks = ceil(float(roots.size()) / float(BLOCK_SIZE));
	printf("Blocksize %d\n", BLOCK_SIZE);
	printf("Roots %zd Blocks %d\n", roots.size(), threadBlocks);

	// --> Initialize progress and gpu compute control
	printGPUInfo();

	cudaEvent_t start, stop;
	uint8_t *running = createHostPointer<uint8_t>(0);
	uint8_t *d_running = createHostPointer<uint8_t>(0);

	rc->running = d_running;

	int *progress = createHostPointer<int>(0);
	int *d_progress = createPinnedPointer<int>(progress);

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

	Depth syncDepth = 2;
	app.add_option<Depth>(
		"-r,--roots",
		syncDepth,
		"The depth to recurse synchronously to (threads spawned = <amount of upgrades>^depth) (overrides block count)"
	);

	Depth maxDepth = 5;
	app.add_option<Depth>(
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

	std::vector<UpgradeId> upgrades = {
		MONEY_PER_QUESTION,
		STREAK_BONUS,
		MULTIPLIER,
	};

	ComputeOptions opts = {
		syncDepth,
		maxDepth,
		loggingFidelity,
	};

	Minimum problems = 0;
	UpgradeId *result = new UpgradeId[maxDepth];
	problems = compute(upgrades, moneyGoal, result, opts);

	printf("========== RESULTS ==========\n");
	printf("Minimum Problems: %lld\n", problems);
	printf("Sequence Required: ");
	for (int i = 0; i < maxDepth; i++) {
		printf("%d ", result[i]);
	}

	return 0;
}
