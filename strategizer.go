package main

import (
	"log"
	"math/rand"
	"time"
)

type UpgradeStats = map[int]int
type UpgradePath struct {
	Problems int
	Sequence []int
}

type PlayOptions struct {
	Stats         UpgradeStats
	Upgrades      []int
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
		Upgrades:      []int{},
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

func Play(o PlayOptions) int {
	problems := 0
	money := o.Money

	stats := CopyStats(o.Stats)

	for _, target := range o.Upgrades {
		upgrades, ok := (*Index)[target]
		if !(stats[target]+1 < len(upgrades)) && ok {
			continue
		}

		goal := upgrades[stats[target]+1]
		streak := 0

		moneyPerQuestion := (*Index)[MONEY_PER_QUESTION][stats[MONEY_PER_QUESTION]]
		streakBonus := (*Index)[STREAK_BONUS][stats[STREAK_BONUS]]
		multiplier := (*Index)[MULTIPLIER][stats[MULTIPLIER]]
		insurance := (*Index)[INSURANCE][stats[INSURANCE]]

		for money < goal.Cost {
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

		stats[target]++
		money -= goal.Cost
	}

	return problems
}

func Permute(base, upgrades []int, depth, max int) [][]int {
	if !(depth < max) {
		result := [][]int{}
		for _, opt := range upgrades {
			result = append(result, append(base, opt))
		}

		return result
	}

	comp := [][]int{}
	for _, opt := range upgrades {
		possibilities := Permute(append(base, opt), upgrades, depth+1, max)
		comp = append(comp, possibilities...)
	}

	return comp
}

func Optimize(sequence, upgrades []int, options *PlayOptions, depth, max int) UpgradePath {
	if !(depth < max) {
		newOptions := *options
		newOptions.Upgrades = sequence
		return UpgradePath{
			Problems: Play(newOptions),
			Sequence: sequence,
		}
	}

	comp := []UpgradePath{}
	for _, opt := range upgrades {
		comp = append(comp, Optimize(
			append(sequence, opt),
			upgrades, options,
			depth+1, max,
		))
	}

	min := UpgradePath{Problems: -1}
	for _, seq := range comp {
		if min.Problems < 0 || seq.Problems < min.Problems {
			min = seq
		}
	}

	return min
}

func ThreadedOptimize(upgrades []int, options *PlayOptions, syncDepth, max int) UpgradePath {
	if syncDepth <= 0 {
		log.Fatal("syncDepth cannot be equal to or less than 0")
	}

	roots := Permute([]int{}, upgrades, 0, syncDepth-1) //? Calculate branches synchronously to a certain depth

	results := make(chan UpgradePath, len(roots))
	for _, r := range roots {
		go func(seq []int, done chan UpgradePath) {
			o := *options
			o.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
			done <- Optimize(seq, upgrades, &o, syncDepth, max)
		}(r, results)
	}

	min := UpgradePath{Problems: -1}
	for i := 0; i < len(roots); i++ {
		path := <-results

		if min.Problems < 0 || path.Problems < min.Problems {
			min = path
		}
	}

	return min
}
