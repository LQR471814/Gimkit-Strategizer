import { PlayerStatistic as ps, Wave, PlayOptions, Money } from "./types";
import { index } from "./data";
import { Worker } from 'worker_threads'

export function playUpgradeOrder(options: PlayOptions) {
	const waves: Wave[] = []
	//? Clone the initial stats (for some reason {...init} and Object.assign do not work)
	const stats = JSON.parse(JSON.stringify(options.init))
	let money = options.money

	for (const target of options.upgrades) {
		const goal = index[target][stats[target].level + 1]
		if (!goal) {
			continue
		}

		let questionsShuffled = 0
		let streak = 0

		const moneyPerQuestion = stats[ps.MONEY_PER_QUESTION].value,
			streakBonus = stats[ps.STREAK_BONUS].value,
			multiplier = stats[ps.MULTIPLIER].value,
			insurance = stats[ps.INSURANCE].value

		while (money < goal.cost) { //? Answering process
			const incorrect = Math.random() < options.setbackChance
			if (incorrect) { //? Incorrect
				const baseSubtr = moneyPerQuestion * multiplier
				money -= baseSubtr - (baseSubtr * insurance / 100)
			} else { //? Correct
				//TODO: Test if the multiplier upgrade multiplies all earnings or just money per question
				money += (moneyPerQuestion * (streakBonus * streak)) * multiplier
				streak += 1
			}

			questionsShuffled++
		}

		stats[target].level += 1
		stats[target].value = goal.value
		money -= goal.cost

		waves.push(questionsShuffled)
	}

	return waves.reduce((p, c) => p + c)
}

type UpgradePath = {
	problems: number
	sequence: number[]
}

export function optimize(options: PlayOptions, sequence: number[], depth: number, max: number): UpgradePath {
	if (depth >= max) {
		const upgradeOptions = { ...options }
		upgradeOptions.upgrades = sequence
		return {
			problems: playUpgradeOrder(upgradeOptions),
			sequence: sequence
		}
	}

	const comp = []
	for (const opt of options.upgrades) {
		comp.push(optimize(
			options,
			sequence.concat(opt),
			depth + 1, max
		))
	}

	let min: UpgradePath = {
		problems: Infinity,
		sequence: []
	}

	for (const seq of comp)
		if (seq.problems < min.problems)
			min = seq

	return min
}

export async function optimize_threaded(options: PlayOptions, sequence: number[], max: number) {
	const comp: Promise<UpgradePath>[] = []
	for (const opt of options.upgrades) {
		const optionWorker = new Worker('./worker.js', {
			workerData : {
				path: './strategizer_worker.ts',
				options: options,
				sequence: sequence.concat(opt),
				depth: 1,
				max: max,
			}
		})

		const workFinished: Promise<UpgradePath> = new Promise(
			resolve => optionWorker.on("message", (result: UpgradePath) => {
				resolve(result)
			})
		)

		comp.push(workFinished)
	}


	let min: UpgradePath = {
		problems: Infinity,
		sequence: []
	}

	for (const p of comp) {
		const seq = await p
		if (seq.problems < min.problems)
			min = seq
	}

	return min
}