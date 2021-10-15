package main

import (
	"log"
	"net/http"
	_ "net/http/pprof"
	"time"
)

func main() {
	log.SetFlags(log.Lshortfile | log.Ltime)

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	testLevels := 5
	testArray := []int{
		MONEY_PER_QUESTION,
		STREAK_BONUS,
		MULTIPLIER,
	}

	options := DefaultPlayOptions()

	log.Println("Timing...")
	recurseTime, result := TimeResult(func() UpgradePath {
		return PlayRecurse(
			options, 1000, testArray, []int{},
			0, testLevels*len(testArray),
		)
	})
	log.Println(recurseTime, result)

	threadedTime, result := TimeResult(func() UpgradePath {
		return RecurseThreaded(
			options, 1000, testArray,
			2, testLevels*len(testArray),
		)
	})

	log.Println(threadedTime, result)
}

func TimeResult(f func() UpgradePath) (time.Duration, UpgradePath) {
	t1 := time.Now()
	result := f()
	elapsed := time.Since(t1)
	return elapsed, result
}
