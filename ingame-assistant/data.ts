export enum Upgrade {
	moneyPerQuestion,
	streakBonus,
	multiplier,
	insurance,
}

export type EnumMap<T> = { [key in keyof Upgrade as Upgrade]: T };

export const nameMap: {
	[key: string]: Upgrade
} = {
	"Money Per Question": Upgrade.moneyPerQuestion,
	"Streak Bonus": Upgrade.streakBonus,
	"Multiplier": Upgrade.multiplier,
	"Amount Covered": Upgrade.insurance,
}

export const idMap: EnumMap<string> = {
	[Upgrade.moneyPerQuestion]: "Money Per Question",
	[Upgrade.streakBonus]: "Streak Bonus",
	[Upgrade.multiplier]: "Multiplier",
	[Upgrade.insurance]: "Amount Covered",
}

export const maxLevel = 10

export const upgradeData = {
	[Upgrade.moneyPerQuestion]: [
		{
			value: 1,
			cost: 0
		},
		{
			value: 5,
			cost: 10
		},
		{
			value: 50,
			cost: 100
		},
		{
			value: 100,
			cost: 1000
		},
		{
			value: 500,
			cost: 10000
		},
		{
			value: 2000,
			cost: 75000
		},
		{
			value: 5000,
			cost: 300000
		},
		{
			value: 10000,
			cost: 1000000
		},
		{
			value: 250000,
			cost: 10000000
		},
		{
			value: 1000000,
			cost: 100000000
		},
	],
	[Upgrade.streakBonus]: [
		{
			value: 1,
			cost: 0
		},
		{
			value: 3,
			cost: 20
		},
		{
			value: 10,
			cost: 200
		},
		{
			value: 50,
			cost: 2000
		},
		{
			value: 250,
			cost: 20000
		},
		{
			value: 1200,
			cost: 200000
		},
		{
			value: 6500,
			cost: 2000000
		},
		{
			value: 35000,
			cost: 20000000
		},
		{
			value: 175000,
			cost: 200000000
		},
		{
			value: 1000000,
			cost: 2000000000
		},
	],
	[Upgrade.multiplier]: [
		{
			value: 1,
			cost: 0
		},
		{
			value: 1.5,
			cost: 50
		},
		{
			value: 2,
			cost: 300
		},
		{
			value: 3,
			cost: 2000
		},
		{
			value: 5,
			cost: 12000
		},
		{
			value: 8,
			cost: 85000
		},
		{
			value: 12,
			cost: 700000
		},
		{
			value: 18,
			cost: 6500000
		},
		{
			value: 30,
			cost: 65000000
		},
		{
			value: 100,
			cost: 1000000000
		},
	],
	[Upgrade.insurance]: [
		{
			value: 0,
			cost: 0
		},
		{
			value: 10,
			cost: 10
		},
		{
			value: 25,
			cost: 250
		},
		{
			value: 40,
			cost: 1000
		},
		{
			value: 50,
			cost: 25000
		},
		{
			value: 70,
			cost: 100000
		},
		{
			value: 80,
			cost: 1000000
		},
		{
			value: 90,
			cost: 5000000
		},
		{
			value: 95,
			cost: 25000000
		},
		{
			value: 99,
			cost: 500000000
		},
	],
}
