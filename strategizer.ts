import { Stats, PlayerStatistic as ps, Wave, UpgradeOrder, Money } from "./types";
import { index } from "./data";

export function playUpgradeOrder(money: Money, init: Stats, order: UpgradeOrder) {
	const waves: Wave[] = []
	//? Clone the initial stats (for some reason {...init} and Object.assign do not work)
	const stats = JSON.parse(JSON.stringify(init))

	for (const target of order.upgrades) {
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
			const incorrect = Math.random() < order.setbackChance
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

	return waves
}
