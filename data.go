package main

type UpgradeIndex = map[int][]UpgradeLevel

type UpgradeLevel struct {
	Value float32
	Cost  float32
}

const (
	MONEY_PER_QUESTION int = iota
	STREAK_BONUS
	MULTIPLIER
	INSURANCE
)

var Index *UpgradeIndex = &UpgradeIndex{
	MONEY_PER_QUESTION: {
		{Value: 1, Cost: 0},
		{Value: 5, Cost: 10},
		{Value: 50, Cost: 100},
		{Value: 100, Cost: 1000},
		{Value: 500, Cost: 10000},
		{Value: 2000, Cost: 75000},
		{Value: 5000, Cost: 300000},
		{Value: 10000, Cost: 1000000},
		{Value: 250000, Cost: 10000000},
		{Value: 1000000, Cost: 100000000},
	},
	STREAK_BONUS: {
		{Value: 1, Cost: 0},
		{Value: 3, Cost: 20},
		{Value: 10, Cost: 200},
		{Value: 50, Cost: 2000},
		{Value: 250, Cost: 20000},
		{Value: 1200, Cost: 200000},
		{Value: 6500, Cost: 2000000},
		{Value: 35000, Cost: 20000000},
		{Value: 175000, Cost: 200000000},
		{Value: 1000000, Cost: 2000000000},
	},
	MULTIPLIER: {
		{Value: 1, Cost: 0},
		{Value: 1.5, Cost: 50},
		{Value: 2, Cost: 300},
		{Value: 3, Cost: 2000},
		{Value: 5, Cost: 12000},
		{Value: 8, Cost: 85000},
		{Value: 12, Cost: 700000},
		{Value: 18, Cost: 6500000},
		{Value: 30, Cost: 65000000},
		{Value: 100, Cost: 1000000000},
	},
	INSURANCE: {
		{Value: 0, Cost: 0},
		{Value: 10, Cost: 10},
		{Value: 25, Cost: 250},
		{Value: 40, Cost: 1000},
		{Value: 50, Cost: 25000},
		{Value: 70, Cost: 100000},
		{Value: 80, Cost: 1000000},
		{Value: 90, Cost: 5000000},
		{Value: 95, Cost: 25000000},
		{Value: 99, Cost: 500000000},
	},
}
