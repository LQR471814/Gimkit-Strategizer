# Gimkit-Strategizer

Something that simulates different upgrade paths and strategies to find which is most effective

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
