import { Upgrade, IDMap } from "./data"
import { path } from "./path.config"

export class PlayState {
	currentPath: Upgrade[]
	stats: IDMap<number>

	constructor() {
		this.currentPath = [...path]
		this.stats = {
			[Upgrade.moneyPerQuestion]: 0,
			[Upgrade.streakBonus]: 0,
			[Upgrade.multiplier]: 0,
			[Upgrade.insurance]: 0
		}
	}

	upgrade() {
		this.stats[this.currentPath[0]]++
		this.currentPath.shift()
	}
}
