## Gimkit-Strategizer

***A tool that simulates different upgrade paths and strategies to find which is most effective***

### Using the tool

The current tool is implemented using C++ and CUDA. The most current version only works on windows although one can easily add in the code to make it work on linux.

Usage can be viewed through `-h`

```text
Options:
  -h,--help                   Print this help message and exit
  -o,--save-filename TEXT     The filename to save progress to after a timeout
  -f,--recover-from TEXT      Saved progress from computing beforehand
  -t,--timeout UINT           When the program should save and exit in seconds (Default: never)
  -g,--goal FLOAT             Amount of money to reach before stopping
  -r,--roots UINT             The depth to recurse synchronously to (threads spawned = <amount of upgrades>^depth) (overrides block count)
  -d,--depth UINT             The amount of upgrades to be purchased
  -l,--logging-fidelity FLOAT The fidelity in which progress is reported (smaller makes progress update more frequently)
```

### Building

To build the tool, just navigate to the `cuda` folder and run either `build.cmd` or `build.sh` depending on which operating system you are on

### In-Game Assistant

I've also written an in-game assistant in the form of a bookmarklet to help keep track of where you are in upgrade path.

To make use of it, just copy this code and paste it into the url of a bookmark (you can name the bookmark anything), when you're inside a game of Gimkit, just click the bookmark and it'll open a small window at the top right of your screen reminding you of where you are.

```javascript
javascript:var e,t,n,o=function(){return o=Object.assign||function(e){for(var t,n=1,o=arguments.length;n<o;n++)for(var i in t=arguments[n])Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i]);return e},o.apply(this,arguments)};!function(e){e[e.moneyPerQuestion=0]="moneyPerQuestion",e[e.streakBonus=1]="streakBonus",e[e.multiplier=2]="multiplier",e[e.insurance=3]="insurance"}(n||(n={}));var i={"Money Per Question":n.moneyPerQuestion,"Streak Bonus":n.streakBonus,Multiplier:n.multiplier,"Amount Covered":n.insurance},r=((e={})[n.moneyPerQuestion]="Money Per Question",e[n.streakBonus]="Streak Bonus",e[n.multiplier]="Multiplier",e[n.insurance]="Amount Covered",e),s=((t={})[n.moneyPerQuestion]=[{value:1,cost:0},{value:5,cost:10},{value:50,cost:100},{value:100,cost:1e3},{value:500,cost:1e4},{value:2e3,cost:75e3},{value:5e3,cost:3e5},{value:1e4,cost:1e6},{value:25e4,cost:1e7},{value:1e6,cost:1e8}],t[n.streakBonus]=[{value:1,cost:0},{value:3,cost:20},{value:10,cost:200},{value:50,cost:2e3},{value:250,cost:2e4},{value:1200,cost:2e5},{value:6500,cost:2e6},{value:35e3,cost:2e7},{value:175e3,cost:2e8},{value:1e6,cost:2e9}],t[n.multiplier]=[{value:1,cost:0},{value:1.5,cost:50},{value:2,cost:300},{value:3,cost:2e3},{value:5,cost:12e3},{value:8,cost:85e3},{value:12,cost:7e5},{value:18,cost:65e5},{value:30,cost:65e6},{value:100,cost:1e9}],t[n.insurance]=[{value:0,cost:0},{value:10,cost:10},{value:25,cost:250},{value:40,cost:1e3},{value:50,cost:25e3},{value:70,cost:1e5},{value:80,cost:1e6},{value:90,cost:5e6},{value:95,cost:25e6},{value:99,cost:5e8}],t),a=[n.moneyPerQuestion,n.moneyPerQuestion,n.multiplier,n.streakBonus,n.streakBonus,n.multiplier,n.streakBonus,n.streakBonus,n.moneyPerQuestion,n.multiplier,n.streakBonus,n.multiplier,n.multiplier,n.moneyPerQuestion,n.streakBonus,n.moneyPerQuestion,n.multiplier,n.moneyPerQuestion,n.moneyPerQuestion,n.moneyPerQuestion,n.multiplier,n.streakBonus,n.moneyPerQuestion,n.moneyPerQuestion,n.moneyPerQuestion],l=function(){function e(){var e;this.currentPath=function(e,t,n){if(n||2===arguments.length)for(var o,i=0,r=t.length;i<r;i++)!o&&i in t||(o||(o=Array.prototype.slice.call(t,0,i)),o[i]=t[i]);return e.concat(o||Array.prototype.slice.call(t))}([],a,!0),this.stats=((e={})[n.moneyPerQuestion]=0,e[n.streakBonus]=0,e[n.multiplier]=0,e[n.insurance]=0,e)}return e.prototype.upgrade=function(){this.stats[this.currentPath[0]]++,this.currentPath.shift()},e}();function u(e){return document.evaluate(e,document,null,XPathResult.FIRST_ORDERED_NODE_TYPE).singleNodeValue}var c=function(){function e(e){this.state=e}return e.prototype.getMoneyElement=function(){return u('//div[contains(text(), "$") and contains(@style, "font-weight: 900")]')},e.prototype.getMoney=function(){var e,t=this.getMoneyElement(),n=null===(e=null==t?void 0:t.textContent)||void 0===e?void 0:e.replace(/[\$,\,]/g,"");return n?parseInt(n):null},e.prototype.listenMoney=function(){var e=this,t=this.getMoneyElement();null!==t&&new MutationObserver((function(){var t,n=e.getMoney();null!==n&&(null===(t=e.onMoney)||void 0===t||t.call(e,n))})).observe(t,{subtree:!0,childList:!0,characterData:!0})},e.prototype.getLevel=function(){var e,t=null===(e=u('(//div[contains(@style, "color: gray")]/../div[2])[last()]'))||void 0===e?void 0:e.textContent;if(t)return parseInt(t.replace(/^\D+/g,""))-1},e.prototype.checkUpgrade=function(e){var t,n,o,r,a=this,l=i[e],c='//div[contains(@style, "white-space: nowrap")]';if(u("".concat(c,'//div[text()="').concat(e,'"]'))){var p=this.getLevel();console.info("Update level ".concat(this.state.stats[l]," -> ").concat(p)),p&&(this.state.stats[l]=p);var v=null===(r=null===(o=null===(n=null===(t=u("".concat(c,'//div[contains(text(), "$').concat(s[l][this.state.stats[l]+1].cost.toLocaleString("en-US"),'")]')))||void 0===t?void 0:t.parentElement)||void 0===n?void 0:n.parentElement)||void 0===o?void 0:o.parentElement)||void 0===r?void 0:r.parentElement;if(!v)return;if("true"===v.getAttribute("listening")||v.hasAttribute("disabled"))return;v.addEventListener("click",(function(){var e;null===(e=a.onUpgrade)||void 0===e||e.call(a,l)})),v.setAttribute("listening","true")}},e.prototype.listenUpgrade=function(){var e=this;new MutationObserver((function(){for(var t in i)e.checkUpgrade(t)})).observe(document,{subtree:!0,characterData:!0,childList:!0})},e.prototype.listen=function(){this.listenMoney(),this.listenUpgrade()},e}(),p={top:"5%",right:"5%",position:"fixed",zIndex:9999,backdropFilter:"blur(4px)",border:"2px solid #8f8f8f",borderRadius:"5px",padding:"10px",backgroundColor:"#00000050"},v={fontWeight:"bold",marginLeft:"5px"},d={color:"white",fontSize:"18px"},y={backgroundColor:"transparent",border:"1px solid white",cursor:"pointer",borderRadius:"5px"},h=function(){function e(){var e=this;this.root=document.createElement("div"),Object.assign(this.root.style,p),document.body.append(this.root);var t=document.createElement("div");t.style.display="flex",t.style.justifyContent="space-between";var n=document.createElement("div");n.style.display="flex",this.title=document.createElement("div"),this.title.textContent="Next up",this.readyDisplay=document.createElement("div"),Object.assign(this.readyDisplay.style,o(o({},d),v)),n.append(this.title,this.readyDisplay);var i=document.createElement("button");i.textContent="Skip",Object.assign(i.style,o(o({},y),d)),i.onclick=function(){var t;return null===(t=e.onSkip)||void 0===t?void 0:t.call(e)},t.append(n,i),this.moneyDisplay=document.createElement("div"),this.targetDisplay=document.createElement("div"),Object.assign(this.title.style,d),Object.assign(this.moneyDisplay.style,d),Object.assign(this.targetDisplay.style,d),this.root.append(t,this.targetDisplay,this.moneyDisplay)}return e.prototype.updateReadiness=function(e){this.readyDisplay.innerHTML=e?"&#10003;":"&#10007;",this.readyDisplay.style.color=e?"#97ff90":"#ff5a5a"},e.prototype.updateGoal=function(e,t){this.moneyDisplay.textContent="Money - ".concat(null!==e?e.toLocaleString():"DONE"),this.targetDisplay.textContent="Upgrade - ".concat(null!==t?r[t]:"DONE")},e}();function m(e){return e.currentPath.length>0?e.currentPath[0]:null}function f(e){var t=m(e);return null!==t?s[t][e.stats[t]+1].cost:null}var g=new l,b=new c(g),P=new h,k=function(e){null===e&&(e=b.getMoney());var t=f(g);console.log(e,t),null!==e&&null!==t&&P.updateReadiness(e>=t)},x=function(){return P.updateGoal(f(g),m(g))};P.onSkip=function(){g.upgrade(),k(null),x()},b.onMoney=function(e){return k(e)},b.onUpgrade=function(e){g.currentPath.length>0&&g.currentPath[0]!==e&&alert("Incorrect upgrade purchased!"),g.upgrade(),k(null),x()},b.listen(),k(null),x();
```

### A testing tool for paths calculated

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

### Precalculated Solutions

The upgrades allowed to be purchased (this does not include insurance these are ideal conditions)

`Upgrades: Money Per Question, Streak Bonus, Multiplier`

This is the # of upgrades allowed to be purchased, it's 30 because of the max upgrade level is 10 and there are 3 possible upgrades

`Upgrade Depth: 30`

This is the money goal, it's the amount of money you wish to achieve the fastest

`Goal: $1,000,000,000`

#### Ideal Solution

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
