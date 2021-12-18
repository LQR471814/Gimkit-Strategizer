# Gimkit-Strategizer

Something
that
simulates
different upgrade paths and strategies to find which is most effective

## A testing tool for paths calculated

```javascript
let listener = (e) => {
    if (
        e.code === "Digit1" ||
        e.code === "Digit2" ||
        e.code === "Digit3" ||
        e.code === "Digit4"
    ) {
        console.log("Problem Answered")
    }
}

window.addEventListener("keydown", listener)
```

Paste this JS snippet into chrome devtools while in a testing session and go through the upgrade path. The number next to "Problem Answered" should be indicative of how many collective answers was required to achieve the money goal and the upgrades listed.

Note: You will have to use the number keys `1-4` in order to answer questions otherwise it will not count it

## Precalculated Solutions

The upgrades allowed to be purchased (this does not include insurance these are ideal conditions)

`Upgrades: Money Per Question, Streak Bonus, Multiplier`

This is the # of upgrades allowed to be purchased, it's 30 because of the max upgrade level is 10 and there are 3 possible upgrades

`Upgrade Depth: 30`

This is the money goal, it's the amount of money you wish to achieve the fastest

`Goal: $1,000,000,000`

### Ideal

Under ideal conditions, considering you can get each problem right and buy at the precise time you reach a money goal.

```text
1. Money Per Question
2. Money Per Question
3. Multiplier
4. Streak Bonus
5. Streak Bonus
6. Multiplier
7. Streak Bonus
8. Streak Bonus
9. Money Per Question
10. Multiplier
11. Streak Bonus
12. Multiplier
13. Multiplier
14. Money Per Question
15. Streak Bonus
16. Money Per Question
17. Multiplier
18. Money Per Question
19. Money Per Question
20. Money Per Question
21. Multiplier
22. Streak Bonus
23. Money Per Question
24. Money Per Question
25. Money Per Question
```
