import { EnumMap, MAX_LEVEL, Upgrade, upgradeData } from "./data"
import { path } from "./path.config"

export type Goal = { upgrade: Upgrade, money: number }

export class PlayState {
	currentPath: Upgrade[]
	stats: EnumMap<number>

	constructor() {
		this.currentPath = [...path]
		this.stats = {
			[Upgrade.moneyPerQuestion]: 0,
			[Upgrade.streakBonus]: 0,
			[Upgrade.multiplier]: 0,
			[Upgrade.insurance]: 0
		}
	}

	upgrade(): void {
		this.stats[this.currentPath[0]]++
		this.currentPath.shift()
	}

	nextup(): Goal {
		const upgrade = this.currentPath[0]
		if (upgrade === undefined || this.isMax(upgrade)) {
			return { upgrade: -1, money: -1 }
		}
		return { upgrade: upgrade, money: upgradeData[upgrade][this.stats[upgrade] + 1].cost }
	}

	isReady = (money: number): boolean =>
		money >= this.nextup().money

	isMax = (id: Upgrade): boolean =>
		this.stats[id] + 1 >= MAX_LEVEL
}

export class QuestionStore {
	map: Map<string, string>
	pending?: { question: string, answer: string }

	constructor() {
		this.map = new Map()
	}
}
