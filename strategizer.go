package main

import (
	"math/rand"
)

type UpgradeStats = map[int]int
type UpgradePath struct {
	Problems int
	Sequence []int
}

type PlayOptions struct {
	Stats         UpgradeStats
	SetbackChance float32
	Money         float32
	Rand          *rand.Rand
}

func DefaultPlayOptions() PlayOptions {
	return PlayOptions{
		Stats: map[int]int{
			MONEY_PER_QUESTION: 0,
			STREAK_BONUS:       0,
			MULTIPLIER:         0,
			INSURANCE:          0,
		},
		SetbackChance: 0,
		Money:         0,
		Rand:          nil,
	}
}

func GeneratePattern(pattern []int, repeat int) []int {
	result := []int{}
	for i := 0; i < repeat; i++ {
		result = append(result, pattern...)
	}

	return result
}

func CopyStats(stats UpgradeStats) UpgradeStats {
	newMap := make(map[int]int)
	for k, v := range stats {
		newMap[k] = v
	}

	return newMap
}

func PlayToGoal(o PlayOptions, goal float32) (int, float32) {
	streak := 0
	problems := 0
	money := o.Money

	moneyPerQuestion := (*Index)[MONEY_PER_QUESTION][o.Stats[MONEY_PER_QUESTION]]
	streakBonus := (*Index)[STREAK_BONUS][o.Stats[STREAK_BONUS]]
	multiplier := (*Index)[MULTIPLIER][o.Stats[MULTIPLIER]]
	insurance := (*Index)[INSURANCE][o.Stats[INSURANCE]]

	for money < goal {
		var incorrect bool
		if o.Rand == nil {
			incorrect = rand.Float32() < o.SetbackChance
		} else {
			incorrect = o.Rand.Float32() < o.SetbackChance
		}

		if incorrect {
			baseSubtr := moneyPerQuestion.Value * multiplier.Value
			money -= baseSubtr - (baseSubtr * insurance.Value / 100)
		} else {
			baseAdd := moneyPerQuestion.Value * (streakBonus.Value * float32(streak))
			money += baseAdd * multiplier.Value
			streak += 1
		}

		problems++
	}

	return problems, money
}

func PlayMoney(o PlayOptions, moneygoal float32) int {
	var problems int
	o.Stats = CopyStats(o.Stats) //? Golang map is a reference

	problems, o.Money = PlayToGoal(o, moneygoal)
	return problems
}

func PlayUpgrade(o PlayOptions, target int) (PlayOptions, int) {
	o.Stats = CopyStats(o.Stats) //? Golang map is a reference
	upgrades, ok := (*Index)[target]
	if !(o.Stats[target]+1 < len(upgrades)) && ok {
		return o, 0
	}

	var problems int
	goal := upgrades[o.Stats[target]+1].Cost
	problems, o.Money = PlayToGoal(o, goal)

	o.Stats[target]++
	o.Money -= goal

	return o, problems
}

func PlayRecurse(options PlayOptions, moneyGoal float32, upgrades, sequence []int, depth, max int) UpgradePath {
	if depth == max {
		problems := PlayMoney(options, moneyGoal)

		return UpgradePath{
			Sequence: sequence,
			Problems: problems,
		}
	}

	min := UpgradePath{Problems: -1}
	for u := range upgrades {
		lowerOption, problemsToUpgrade := PlayUpgrade(options, u)
		path := PlayRecurse(
			lowerOption, moneyGoal,
			upgrades, append(sequence, u),
			depth+1, max,
		)

		if min.Problems < 0 || problemsToUpgrade < min.Problems {
			min = UpgradePath{
				Sequence: path.Sequence,
				Problems: problemsToUpgrade + path.Problems,
			}
		}
	}

	return min
}
