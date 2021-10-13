import { PlayerStatistic } from "./types";
import { performance } from 'perf_hooks'

export function generatePattern(pattern: PlayerStatistic[], repeat: number) {
	const upgrades = []
	for (let i = 0; i < repeat; i++) {
		for (const stat of pattern) {
			upgrades.push(stat)
		}
	}

	return upgrades
}

export function timeSync(func: () => void) {
	const t1 = performance.now()
	const result = func()
	const t2 = performance.now()
	return [result, (t2 - t1) / 1000]
}

export async function timeAsync(func: () => void) {
	const t1 = performance.now()
	const result = await func()
	const t2 = performance.now()
	return [result, (t2 - t1) / 1000]
}
