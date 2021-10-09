import { PlayerStatistic } from "./types";

export function generatePattern(pattern: PlayerStatistic[], repeat: number) {
	const upgrades = []
	for (let i = 0; i < repeat; i++) {
		for (const stat of pattern) {
			upgrades.push(stat)
		}
	}

	return upgrades
}