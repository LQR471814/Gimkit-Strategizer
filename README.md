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
javascript:var t=function(e,n){return t=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(t,e){t.__proto__=e}||function(t,e){for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&(t[n]=e[n])},t(e,n)};var e,n,o,i=function(){return i=Object.assign||function(t){for(var e,n=1,o=arguments.length;n<o;n++)for(var i in e=arguments[n])Object.prototype.hasOwnProperty.call(e,i)&&(t[i]=e[i]);return t},i.apply(this,arguments)};function s(t,e,n){if(n||2===arguments.length)for(var o,i=0,s=e.length;i<s;i++)!o&&i in e||(o||(o=Array.prototype.slice.call(e,0,i)),o[i]=e[i]);return t.concat(o||Array.prototype.slice.call(e))}!function(t){t[t.moneyPerQuestion=0]="moneyPerQuestion",t[t.streakBonus=1]="streakBonus",t[t.multiplier=2]="multiplier",t[t.insurance=3]="insurance"}(o||(o={}));var r,a={"Money Per Question":o.moneyPerQuestion,"Streak Bonus":o.streakBonus,Multiplier:o.multiplier,"Amount Covered":o.insurance},u=((e={})[o.moneyPerQuestion]="Money Per Question",e[o.streakBonus]="Streak Bonus",e[o.multiplier]="Multiplier",e[o.insurance]="Amount Covered",e),l=((n={})[o.moneyPerQuestion]=[{value:1,cost:0},{value:5,cost:10},{value:50,cost:100},{value:100,cost:1e3},{value:500,cost:1e4},{value:2e3,cost:75e3},{value:5e3,cost:3e5},{value:1e4,cost:1e6},{value:25e4,cost:1e7},{value:1e6,cost:1e8}],n[o.streakBonus]=[{value:1,cost:0},{value:3,cost:20},{value:10,cost:200},{value:50,cost:2e3},{value:250,cost:2e4},{value:1200,cost:2e5},{value:6500,cost:2e6},{value:35e3,cost:2e7},{value:175e3,cost:2e8},{value:1e6,cost:2e9}],n[o.multiplier]=[{value:1,cost:0},{value:1.5,cost:50},{value:2,cost:300},{value:3,cost:2e3},{value:5,cost:12e3},{value:8,cost:85e3},{value:12,cost:7e5},{value:18,cost:65e5},{value:30,cost:65e6},{value:100,cost:1e9}],n[o.insurance]=[{value:0,cost:0},{value:10,cost:10},{value:25,cost:250},{value:40,cost:1e3},{value:50,cost:25e3},{value:70,cost:1e5},{value:80,cost:1e6},{value:90,cost:5e6},{value:95,cost:25e6},{value:99,cost:5e8}],n),c=[o.moneyPerQuestion,o.moneyPerQuestion,o.multiplier,o.streakBonus,o.streakBonus,o.multiplier,o.streakBonus,o.streakBonus,o.moneyPerQuestion,o.multiplier,o.streakBonus,o.multiplier,o.multiplier,o.moneyPerQuestion,o.streakBonus,o.moneyPerQuestion,o.multiplier,o.moneyPerQuestion,o.moneyPerQuestion,o.moneyPerQuestion,o.multiplier,o.streakBonus,o.moneyPerQuestion,o.moneyPerQuestion,o.moneyPerQuestion],p=function(){function t(){var t,e=this;this.isReady=function(t){return t>=e.nextup().money},this.isMax=function(t){return e.stats[t]+1>=10},this.currentPath=s([],c,!0),this.stats=((t={})[o.moneyPerQuestion]=0,t[o.streakBonus]=0,t[o.multiplier]=0,t[o.insurance]=0,t)}return t.prototype.upgrade=function(){this.stats[this.currentPath[0]]++,this.currentPath.shift()},t.prototype.nextup=function(){var t=this.currentPath[0];return void 0===t||this.isMax(t)?{upgrade:-1,money:-1}:{upgrade:t,money:l[t][this.stats[t]+1].cost}},t}();function d(t,e){return void 0===e&&(e=document),document.evaluate(t,null!=e?e:document,null,XPathResult.FIRST_ORDERED_NODE_TYPE).singleNodeValue}!function(t){t[t.QUESTION=0]="QUESTION",t[t.SHOP_LEVELS_SCREEN=1]="SHOP_LEVELS_SCREEN",t[t.MONEY_INDICATOR=2]="MONEY_INDICATOR"}(r||(r={}));var y=function(){function t(t){var e=this;this.targets=t,this.observer=new MutationObserver((function(){return e.onUpdate()})),this.observer.observe(document.body,{subtree:!0,childList:!0,characterData:!0})}return t.prototype.onUpdate=function(){for(var t=0,e=this.targets;t<e.length;t++){var n=e[t],o=n.screen.element();null!==o&&n.callback(o,n.screen)}},t}(),v=function(){function t(){}return t.prototype.element=function(){return null},t}(),h=function(){function t(){this.element=function(){return d('//div[contains(text(), "$") and contains(@style, "font-weight: 900")]')}}return t.prototype.money=function(){var t,e=this.element(),n=null===(t=null==e?void 0:e.textContent)||void 0===t?void 0:t.replace(/[\$,\,]/g,"");return parseInt(n)},t}(),m=function(){function t(){}return t.prototype.question=function(){return d("//span",this.element())},t.prototype.element=function(){return d("//div[contains(@class, 'enter-done')]")},t}(),f=function(e){function n(t,n){var o=e.call(this)||this;return o.buttonRoot='//div[contains(@style, "white-space: nowrap")]',o.element=function(){return d("".concat(o.buttonRoot,'//div[text()="').concat(o.displayName,'"]'))},o.displayName=t,o.state=n,o}return function(e,n){if("function"!=typeof n&&null!==n)throw new TypeError("Class extends value "+String(n)+" is not a constructor or null");function o(){this.constructor=e}t(e,n),e.prototype=null===n?Object.create(n):(o.prototype=n.prototype,new o)}(n,e),n.prototype.level=function(){var t,e=null===(t=d('(//div[contains(@style, "color: gray")]/../div[2])[last()]'))||void 0===t?void 0:t.textContent;if(e)return parseInt(e.replace(/^\D+/g,""))-1},n.prototype.attachUpgradeTrigger=function(t){var e,n,o,i,s=this,r=a[this.displayName],u=this.level();if(void 0!==u&&u>this.state.stats[r]&&(console.info("Update level ".concat(this.state.stats[r]," -> ").concat(u)),this.state.stats[r]=u,t()),!this.state.isMax(r)){var c=null===(i=null===(o=null===(n=null===(e=d("".concat(this.buttonRoot,'//div[contains(text(), "$').concat(l[r][this.state.stats[r]+1].cost.toLocaleString("en-US"),'")]')))||void 0===e?void 0:e.parentElement)||void 0===n?void 0:n.parentElement)||void 0===o?void 0:o.parentElement)||void 0===i?void 0:i.parentElement;c&&("true"===c.getAttribute("listening")||c.hasAttribute("disabled")||(c.addEventListener("click",(function(){s.state.upgrade(),t()})),c.setAttribute("listening","true")))}},n}(v),g=function(t,e){var n=this;this.fetchMoney=function(){return n._moneyIndicator.money()},this.state=t,this.eventHooks=e,this._moneyIndicator=new h,this.monitor=new y(s([{screen:this._moneyIndicator,callback:function(){return n.eventHooks.onMoney(n._moneyIndicator.money())}},{screen:new m,callback:function(){}}],Object.keys(a).map((function(e){return{screen:new f(e,t),callback:function(t,o){o.attachUpgradeTrigger((function(){return n.eventHooks.onUpgrade(a[e])}))}}})),!0))},b={top:"5%",right:"5%",position:"fixed",zIndex:9999,backdropFilter:"blur(4px)",border:"2px solid #8f8f8f",borderRadius:"5px",padding:"10px",backgroundColor:"#00000050"},k={fontWeight:"bold",marginLeft:"5px"},x={color:"white",fontSize:"18px"},E={backgroundColor:"transparent",border:"1px solid white",cursor:"pointer",borderRadius:"5px"},P=function(){function t(t){var e=this;this.hooks=t,this.root=document.createElement("div"),Object.assign(this.root.style,b),document.children[0].append(this.root);var n=document.createElement("div");n.style.display="flex",n.style.justifyContent="space-between";var o=document.createElement("div");o.style.display="flex",this.title=document.createElement("div"),this.title.textContent="Next up",this.readyDisplay=document.createElement("div"),Object.assign(this.readyDisplay.style,i(i({},x),k)),o.append(this.title,this.readyDisplay);var s=document.createElement("button");s.textContent="Skip",Object.assign(s.style,i(i({},E),x)),s.onclick=function(){return e.hooks.onSkip()},n.append(o,s),this.moneyDisplay=document.createElement("div"),this.targetDisplay=document.createElement("div"),Object.assign(this.title.style,x),Object.assign(this.moneyDisplay.style,x),Object.assign(this.targetDisplay.style,x),this.root.append(n,this.targetDisplay,this.moneyDisplay)}return t.prototype.updateReadiness=function(t){this.readyDisplay.innerHTML=t?"&#10003;":"&#10007;",this.readyDisplay.style.color=t?"#97ff90":"#ff5a5a"},t.prototype.updateGoal=function(t){if(t.money<0&&t.upgrade<0)return this.updateReadiness(!0),this.moneyDisplay.textContent="Money - DONE",void(this.targetDisplay.textContent="Upgrade - DONE");this.moneyDisplay.textContent="Money - ".concat(t.money.toLocaleString()),this.targetDisplay.textContent="Upgrade - ".concat(u[t.upgrade])},t}(),O=new p,D=new P({onSkip:function(){O.upgrade(),D.updateGoal(O.nextup())}}),Q=new g(O,{onMoney:function(t){return D.updateReadiness(O.isReady(t))},onUpgrade:function(){return D.updateGoal(O.nextup())}});D.updateReadiness(O.isReady(Q.fetchMoney())),D.updateGoal(O.nextup());
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
