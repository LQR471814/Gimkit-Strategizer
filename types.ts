export enum PlayerStatistic {
	MONEY_PER_QUESTION,
	STREAK_BONUS,
	MULTIPLIER,
	INSURANCE,
}

export type Money = number

export type UpgradeIndex = {
	[key in PlayerStatistic]: UpgradeLevel[]
}

export type UpgradeLevel = {
	value: number
	cost: number
}

export type Stats = {
	[key in PlayerStatistic]: {
		value: number
		level: number //? Levels count by index
	}
}

export type PlayOptions = {
	init: Stats
	upgrades: PlayerStatistic[]
	setbackChance: number //? Probability (0 - 1) that problem will be incorrect, defined as (avg incorrect) / (avg correct)
	money: number
}

//? A "wave" is defined loosely as the number of questions answered correctly required
//?  to be able to increase the requirement given by one level
export type Wave = number
