#include <curand.h>
#include <curand_kernel.h>

#include <chrono>
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

__forceinline__ __host__ __device__ void iterativeCall(PlayStackFrame *stack, PlayStackParameters params, Depth* depth)
{
	(*depth)++;
	stack[*depth].branch = 0;
	stack[*depth].params = params;
}

__forceinline__ __host__ __device__ void iterativeReturn(PlayStackFrame *stack, Depth* depth, int value)
{
	(*depth)--;
	if (value < stack[*depth].currentMin || stack[*depth].currentMin < 0) {
		stack[*depth].currentMin = value;
		stack[*depth].minTarget = stack[*depth].branch;
	}

	stack[*depth].branch++;
}

__host__ __device__ int playIterative(
	ComputeContext *c,
	ComputeState state,
	Depth startOffset
) {
	PlayStackFrame *stack = state.stack;
	Depth *depth = state.depth;

	//? To prevent crashes when the initial moneyValue is already larger than the goal
	if (stack[*depth].params.state.money >= (*c).moneyGoal) {
		return stack[*depth].params.problems;
	}

	while (true) {
		if (*(*c).cancel == 1) {
			return state.problems;
		}

		if (
			(*c).currentMinimum &&
			stack[*depth].params.problems >= *(*c).currentMinimum &&
			*(*c).currentMinimum > 0
		) {
			iterativeReturn(
				stack, depth,
				stack[*depth].params.problems + 9999
			);
			continue;
		}

		if (stack[*depth].params.state.money >= (*c).moneyGoal) {
			iterativeReturn(
				stack, depth,
				stack[*depth].params.problems
			);
			continue;
		}

		if (stack[*depth].branch == (*c).upgradesSize) {
			state.sequence[startOffset + *depth] = stack[*depth].minTarget;
			if (*depth == 0) {
				return stack[*depth].currentMin;
			}

			iterativeReturn(
				stack, depth,
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

			iterativeReturn(
				stack, depth,
				stack[*depth].params.problems + res.problems
			);
			continue;
		}

		if (getStat(
			stack[*depth].params.state.stats,
			(*c).upgrades[stack[*depth].branch]
		)+1 == MAX_LEVEL) {
			iterativeCall(stack, stack[*depth].params, depth);
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
		};

		iterativeCall(stack, {
			lowerState,
			stack[*depth].params.problems + res.problems,
			stack[*depth].currentMin
		}, depth);
	}
}

template <typename T>
void assignVecToPointer(std::vector<T> vec, T *result, uint64_t size)
{
	for (int i = 0; i < size; i++) {
		result[i] = vec[i];
	}
}

UpgradeLevel* allocUpgradeLevels(std::vector<UpgradeLevel> levels)
{
	UpgradeLevel *result;
	cudaMallocManaged(&result, sizeof(UpgradeLevel) * levels.size());
	for (int i = 0; i < levels.size(); i++) {
		result[i] = levels[i];
	}
	return result;
}

std::vector<Permutation> getRoots(
	UpgradeLevel **data,
	std::vector<UpgradeId> upgrades,
	Depth syncDepth
) {
	PermuteContext c = {data, upgrades, syncDepth};
	PermuteState r = {
		UpgradeStats{0, 0, 0, 0}, // play
		0, 0, std::vector<UpgradeId>{},		  // sequence
	};

	return permuteRecursive(&c, r, 0);
}

struct UpgradeLevel** initializeIndex()
{
	UpgradeLevel **data;
	cudaMallocManaged(&data, sizeof(UpgradeLevel) * UPGRADE_COUNT);

	data[MONEY_PER_QUESTION] = allocUpgradeLevels(moneyPerQuestionLevels);
	data[STREAK_BONUS] = allocUpgradeLevels(streakBonusLevels);
	data[MULTIPLIER] = allocUpgradeLevels(multiplierLevels);
	data[INSURANCE] = allocUpgradeLevels(insuranceLevels);

	return data;
}

void deallocateIndex(UpgradeLevel **data)
{
	for (int i = 0; i < UPGRADE_COUNT; i++)
		cudaFree(data[i]);
	cudaFree(data);
}

UpgradeId* initializeSequence(std::vector<UpgradeId> init, int targetSize)
{
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

UpgradeId* initializeUpgrades(std::vector<UpgradeId> init)
{
	UpgradeId *upgrades;
	cudaMallocManaged(&upgrades, sizeof(UpgradeId) * init.size());
	for (int i = 0; i < init.size(); i++) {
		upgrades[i] = init[i];
	}

	return upgrades;
}

PlayStackFrame* initializeStack(int lowerDepth, int upgradesSize)
{
	PlayStackFrame *stack;
	cudaMallocManaged(&stack, sizeof(PlayStackFrame) * (lowerDepth+1));
	for (int i = 0; i < lowerDepth+1; i++) {
		stack[i] = {};
	}

	return stack;
}

__global__ void computeStrategy(
	uint32_t *progress,
	ComputeContext *c,
	ComputeState *states,
	int rootSize,
	int offset
) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= rootSize) {
		return;
	}

	uint32_t problems = playIterative(c, states[index], offset);
	states[index].problems = problems;

	if (progress)
		atomicAdd(progress, 1);
}

ComputeContext* initializeContext(
	std::vector<UpgradeId> upgrades,
	Money moneygoal,
	ComputeOptions options
) {
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

ComputeState* initializeThreadStates(
	std::vector<Permutation> roots,
	MaxUpgradeLevel upgrades,
	ComputeOptions opts
) {
	ComputeState *results;
	cudaMallocManaged(&results, sizeof(ComputeState) * roots.size());

	for (int i = 0; i < roots.size(); i++) {
		UpgradeId *sequence = initializeSequence(roots[i].sequence, opts.maxDepth);
		PlayStackFrame *stack = initializeStack(opts.maxDepth - opts.syncDepth, upgrades);
		stack[0].params = {roots[i].play, roots[i].problems, -1};

		Depth* depth;
		cudaMallocManaged(&depth, sizeof(Depth));

		results[i] = ComputeState{
			roots[i].play,
			roots[i].problems,
			depth,
			stack,
			sequence
		};
	}

	return results;
}

ComputeState compute(
	std::vector<UpgradeId> upgrades,
	Money moneyGoal,
	ComputeOptions opts
) {
	// --> Initialize Roots / Compute States
	ComputeContext *rc = initializeContext(upgrades, moneyGoal, opts);
	std::vector<Permutation> roots = getRoots((*rc).data, upgrades, opts.syncDepth);
	ComputeState* states;

	if (!opts.recoverFrom.empty()) {
		FILE *pFile;
		pFile = fopen(opts.recoverFrom.c_str(), "rb");
		if (pFile == NULL) {
			fputs("File error", stderr);
			exit(1);
		}

		char* b = (char*)malloc(sizeof(Minimum));
		fread(b, sizeof(Minimum), 1, pFile);
		*(rc->currentMinimum) = readFromPointer<Minimum>(&b);

		size_t stateSize = computeStateSize(
			opts.maxDepth - opts.syncDepth,
			opts.maxDepth
		);

		cudaMallocManaged(&states, roots.size() * sizeof(ComputeState));

		for (int i = 0; i < roots.size(); i++) {
			char* buff = (char*)malloc(stateSize);
			fread(buff, sizeof(char), stateSize, pFile);
			states[i] = unserializeComputeState(
				buff,
				opts.maxDepth - opts.syncDepth,
				opts.maxDepth
			);
			states[i].problems = roots[i].problems;
			free(buff);
		}

		fclose(pFile);
	} else {
		states = initializeThreadStates(roots, upgrades.size(), opts);
	}

	printf("Memory Allocation Succeeded\n");

	int threadBlocks = ceil(float(roots.size()) / float(BLOCK_SIZE));
	printf("Blocksize %d\n", BLOCK_SIZE);
	printf("Roots %zd Blocks %d\n", roots.size(), threadBlocks);

	// --> Initialize progress and gpu compute control
	printGPUInfo();

	cudaEvent_t start, stop;

	uint8_t *cancel = createHostPointer<uint8_t>(0);
	uint8_t *d_cancel = createPinnedPointer<uint8_t>(cancel);

	rc->cancel = d_cancel;

	uint32_t *progress = createHostPointer<uint32_t>(0);
	uint32_t *d_progress = createPinnedPointer<uint32_t>(progress);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// --> Run
	computeStrategy<<<threadBlocks, BLOCK_SIZE>>>(
		d_progress, rc, states, roots.size(), opts.syncDepth
	);

	printf("Computing...\n");

	// --> Report progress
	cudaEventRecord(stop);
	int bufProgress = 0;
	int trueProgress = 0;
	do {
		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		if (
			opts.timeout > 0 && std::chrono::duration_cast<std::chrono::seconds>(now - begin)
				>= opts.timeout * std::chrono::seconds{1}
		) {
			*cancel = 1;
			printf("Aborting and saving...\n");
			break;
		}

		cudaEventQuery(stop);
		trueProgress = *progress;
		if (trueProgress - bufProgress >= roots.size() * opts.loggingFidelity) {
			bufProgress = trueProgress;
			printf("Progress %d / %zd\n", bufProgress, roots.size());
		}
	} while (trueProgress < roots.size());

	// --> Report run performance
	cudaEventSynchronize(stop);
	printf(
		"\nCompute Status (ignore if there is no visible error) %s\n",
		cudaGetErrorString(cudaGetLastError())
	);

	float *elapsed = new float;
	cudaEventElapsedTime(elapsed, start, stop);
	printf("Completed in %fs\n", (*elapsed) / 1000);
	delete elapsed;

	// --> Initialize Serialized States
	char** serializedStates;
	if (*cancel == 1) {
		serializedStates = (char**)malloc(sizeof(char*) * roots.size());
	}

	// --> Sort results
	ComputeState minCompState = {};

	int min = -1;
	for (int i = 0; i < roots.size(); i++) {
		if (*cancel == 1) { // --> Serialize States
			serializedStates[i] = serializeComputeState(
				states[i],
				opts.maxDepth - opts.syncDepth,
				opts.maxDepth
			);
		} else if ((min < 0 || states[i].problems < min)) {
			min = states[i].problems;
			if (min != -1) {
				cudaFree(minCompState.stack);
				cudaFree(minCompState.sequence);
			}
			minCompState = copyComputeState(
				states[i],
				opts.maxDepth - opts.syncDepth,
				opts.maxDepth
			);
		}

		cudaFree(states[i].sequence);
		cudaFree(states[i].stack);
	}

	// --> Write state (if canceled)
	if (*cancel == 1) {
		FILE *pFile;
		pFile = fopen(opts.saveFilename.c_str(), "wb");

		size_t stateSize = computeStateSize(
			opts.maxDepth - opts.syncDepth,
			opts.maxDepth
		);

		fwrite(rc->currentMinimum, sizeof(Minimum), 1, pFile);

		for (int i = 0; i < roots.size(); i++) {
			fwrite(
				serializedStates[i], sizeof(char),
				stateSize, pFile
			);
			free(serializedStates[i]);
		}

		fclose(pFile);
		free(serializedStates);
	}

	// --> Cleanup
	cudaFree(states);

	cudaFree(cancel);
	cudaFree(d_cancel);
	cudaFree(progress);
	cudaFree(d_progress);

	deallocateContext(rc);
	return minCompState;
}

int main(int argc, char** argv)
{
	CLI::App app{"A program that simulates many, many gimkit games"};

	std::string saveFilename = "serialized.bin";
	app.add_option<std::string>(
		"-o,--save-filename",
		saveFilename,
		"The filename to save progress to after a timeout"
	);

	std::string recoverFrom = "";
	app.add_option<std::string>(
		"-f,--recover-from",
		recoverFrom,
		"Saved progress from computing beforehand"
	);

	uint64_t executionTimeout = 0;
	app.add_option<uint64_t>(
		"-t,--timeout",
		executionTimeout,
		"When the program should save and exit in seconds (Default: never)"
	);

	Money moneyGoal = 1000000;
	app.add_option<Money>(
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
		executionTimeout,
		saveFilename,
		recoverFrom,
		loggingFidelity,
	};

	ComputeState result = compute(upgrades, moneyGoal, opts);

	printf("========== RESULTS ==========\n");
	// printComputeState(result, maxDepth - syncDepth, maxDepth);
	printf("Minimum Problems: %u\n", result.problems);
	printf("Sequence Required: ");
	for (int i = 0; i < maxDepth; i++) {
		printf("%d ", result.sequence[i]);
	}
	printf("\n");

	return 0;
}
