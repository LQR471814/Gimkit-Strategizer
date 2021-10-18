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

type PlayState struct {
	Stats         UpgradeStats
	SetbackChance float32
	Money         float32
	Rand          *rand.Rand
}

func DefaultPlayOptions() PlayState {
	return PlayState{
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

func PlayToGoal(s PlayState, goal float32) (int, float32) {
	streak := 0
	problems := 0
	money := s.Money

	moneyPerQuestion := (*Index)[MONEY_PER_QUESTION][s.Stats[MONEY_PER_QUESTION]]
	streakBonus := (*Index)[STREAK_BONUS][s.Stats[STREAK_BONUS]]
	multiplier := (*Index)[MULTIPLIER][s.Stats[MULTIPLIER]]
	insurance := (*Index)[INSURANCE][s.Stats[INSURANCE]]

	for money < goal {
		var incorrect bool
		if s.Rand == nil {
			incorrect = rand.Float32() < s.SetbackChance
		} else {
			incorrect = s.Rand.Float32() < s.SetbackChance
		}

		if incorrect {
			baseSubtr := moneyPerQuestion.Value * multiplier.Value
			money -= baseSubtr - (baseSubtr * insurance.Value / 100)
		} else {
			baseAdd := moneyPerQuestion.Value + (streakBonus.Value * float32(streak))
			money += baseAdd * multiplier.Value
			streak += 1
		}

		problems++
	}

	return problems, money
}

func PlayMoney(s PlayState, moneygoal float32) int {
	var problems int
	s.Stats = CopyStats(s.Stats) //? Golang map is a reference

	problems, s.Money = PlayToGoal(s, moneygoal)
	return problems
}

func PlayUpgrade(s PlayState, target int) (PlayState, int) {
	s.Stats = CopyStats(s.Stats) //? Golang map is a reference
	upgrades, ok := (*Index)[target]
	if !(s.Stats[target]+1 < len(upgrades)) && ok {
		return s, 0
	}

	var problems int
	goal := upgrades[s.Stats[target]+1].Cost
	problems, s.Money = PlayToGoal(s, goal)

	s.Stats[target]++
	s.Money -= goal

	return s, problems
}

func PermutePlay(state PlayState, upgrades, sequence []int, depth, max int) ([]UpgradePath, []PlayState) {
	if depth == max {
		return []UpgradePath{{
			Sequence: sequence,
			Problems: 0,
		}}, []PlayState{state}
	}

	sequences := []UpgradePath{}
	states := []PlayState{}
	for u := range upgrades {
		lowerOption, problems := PlayUpgrade(state, u)
		// log.Println(lowerOption, problems)

		p, o := PermutePlay(
			lowerOption, upgrades,
			append(sequence, u), depth+1, max,
		)

		for i := range p {
			p[i].Problems += problems
		}

		sequences = append(sequences, p...)
		states = append(states, o...)
	}

	return sequences, states
}

func PlayRecurse(state PlayState, moneyGoal float32, upgrades, sequence []int, depth, max int) UpgradePath {
	if state.Money >= moneyGoal {
		return UpgradePath{
			Sequence: sequence,
			Problems: 0,
		}
	}

	if depth == max {
		problems := PlayMoney(state, moneyGoal)

		return UpgradePath{
			Sequence: sequence,
			Problems: problems,
		}
	}

	min := UpgradePath{Problems: -1}
	for u := range upgrades {
		lowerOption, problemsToUpgrade := PlayUpgrade(state, u)
		path := PlayRecurse(
			lowerOption, moneyGoal,
			upgrades, append(sequence, u),
			depth+1, max,
		)

		if min.Problems < 0 || problemsToUpgrade+path.Problems < min.Problems {
			min = UpgradePath{
				Sequence: path.Sequence,
				Problems: problemsToUpgrade + path.Problems,
			}
		}
	}

	return min
}

func RecurseThreaded(options PlayState, moneyGoal float32, upgrades []int, threadDepth, max int) UpgradePath {
	if !(threadDepth > 0) {
		log.Fatal("Thread depth must be greater than 0")
	}

	roots, states := PermutePlay(options, upgrades, []int{}, 0, threadDepth)
	results := make(chan UpgradePath, len(roots))
	for i := range roots {
		go func(s PlayState, r UpgradePath) {
			s.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
			result := PlayRecurse(s, moneyGoal, upgrades, r.Sequence, threadDepth, max)
			result.Problems += r.Problems //? Add initial problems
			results <- result
		}(states[i], roots[i])
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
