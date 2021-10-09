import { generatePattern } from "./utils";
import { Money, PlayerStatistic as ps, Stats, UpgradeOrder } from "./types";
import { levelMax } from "./data";
import { playUpgradeOrder } from "./strategizer"

let money: Money = 0

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

const testOrder1_1: UpgradeOrder = {
	upgrades: generatePattern([
		ps.MONEY_PER_QUESTION,
		ps.MULTIPLIER,
		ps.STREAK_BONUS,
	], levelMax),
	setbackChance: 0,
}

const testOrder1_2: UpgradeOrder = {
	upgrades: generatePattern([
		ps.MONEY_PER_QUESTION,
		ps.STREAK_BONUS,
		ps.MULTIPLIER,
	], levelMax),
	setbackChance: 0,
}

const testOrder2_1: UpgradeOrder = {
	upgrades: generatePattern([
		ps.STREAK_BONUS,
		ps.MONEY_PER_QUESTION,
		ps.MULTIPLIER,
	], levelMax),
	setbackChance: 0,
}

const testOrder2_2: UpgradeOrder = {
	upgrades: generatePattern([
		ps.STREAK_BONUS,
		ps.MULTIPLIER,
		ps.MONEY_PER_QUESTION,
	], levelMax),
	setbackChance: 0,
}

const testOrder3_1: UpgradeOrder = {
	upgrades: generatePattern([
		ps.MULTIPLIER,
		ps.STREAK_BONUS,
		ps.MONEY_PER_QUESTION,
	], levelMax),
	setbackChance: 0,
}

const testOrder3_2: UpgradeOrder = {
	upgrades: generatePattern([
		ps.MULTIPLIER,
		ps.MONEY_PER_QUESTION,
		ps.STREAK_BONUS,
	], levelMax),
	setbackChance: 0,
}

console.log(playUpgradeOrder(money, stats, testOrder1_1).reduce((p, c) => p + c))
console.log(playUpgradeOrder(money, stats, testOrder1_2).reduce((p, c) => p + c))
console.log(playUpgradeOrder(money, stats, testOrder2_1).reduce((p, c) => p + c))
console.log(playUpgradeOrder(money, stats, testOrder2_2).reduce((p, c) => p + c))
console.log(playUpgradeOrder(money, stats, testOrder3_1).reduce((p, c) => p + c))
console.log(playUpgradeOrder(money, stats, testOrder3_2).reduce((p, c) => p + c))