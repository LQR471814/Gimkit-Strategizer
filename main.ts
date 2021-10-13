import { Money, PlayerStatistic as ps, Stats } from "./types";
import { optimize, optimize_threaded, playUpgradeOrder } from "./strategizer"
import { generatePattern, timeAsync, timeSync } from "./utils";

let money: Money = 0
let setback = 0

const stats: Stats = {
	[ps.MONEY_PER_QUESTION]: {
		level: 0,
		value: 1
	},
	[ps.STREAK_BONUS]: {
		level: 0,
		value: 1
	},
	[ps.MULTIPLIER]: {
		level: 0,
		value: 1
	},
	[ps.INSURANCE]: {
		level: 0,
		value: 0
	},
}

const testArray = [
  ps.MONEY_PER_QUESTION,
  ps.STREAK_BONUS,
  ps.MULTIPLIER
]

const testLevels = 5

console.log(timeSync(() => optimize({
	init: stats,
	money: money,
	setbackChance: setback,
	upgrades: testArray
}, [], 0, testLevels * testArray.length)))

console.log(playUpgradeOrder({
	init: stats,
	money: money,
	setbackChance: setback,
	upgrades: generatePattern(testArray, testLevels)
}))

timeAsync(() => optimize_threaded({
	init: stats,
	money: money,
	setbackChance: setback,
	upgrades: testArray
}, [], testLevels * testArray.length)).then((value) => {
	console.log(value)
})
