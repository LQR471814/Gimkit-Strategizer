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

I've also written an in-game assistant in the form of a bookmarklet to help keep track of where you are in upgrade path. It also memorizes questions you've already gotten right to make it easier to get on high streaks.

To make use of it, just copy this code and paste it into the url of a bookmark (you can name the bookmark anything), when you're inside a game of Gimkit, just click the bookmark and it'll open a small window at the top right of your screen reminding you of where you are.

```javascript
javascript:var t,e,n,o=function(){return o=Object.assign||function(t){for(var e,n=1,o=arguments.length;n<o;n++)for(var i in e=arguments[n])Object.prototype.hasOwnProperty.call(e,i)&&(t[i]=e[i]);return t},o.apply(this,arguments)};function i(t,e,n){if(n||2===arguments.length)for(var o,i=0,s=e.length;i<s;i++)!o&&i in e||(o||(o=Array.prototype.slice.call(e,0,i)),o[i]=e[i]);return t.concat(o||Array.prototype.slice.call(e))}!function(t){t[t.moneyPerQuestion=0]="moneyPerQuestion",t[t.streakBonus=1]="streakBonus",t[t.multiplier=2]="multiplier",t[t.insurance=3]="insurance"}(n||(n={}));var s={"Money Per Question":n.moneyPerQuestion,"Streak Bonus":n.streakBonus,Multiplier:n.multiplier,"Amount Covered":n.insurance},r=((t={})[n.moneyPerQuestion]="Money Per Question",t[n.streakBonus]="Streak Bonus",t[n.multiplier]="Multiplier",t[n.insurance]="Amount Covered",t),a=((e={})[n.moneyPerQuestion]=[{value:1,cost:0},{value:5,cost:10},{value:50,cost:100},{value:100,cost:1e3},{value:500,cost:1e4},{value:2e3,cost:75e3},{value:5e3,cost:3e5},{value:1e4,cost:1e6},{value:25e4,cost:1e7},{value:1e6,cost:1e8}],e[n.streakBonus]=[{value:1,cost:0},{value:3,cost:20},{value:10,cost:200},{value:50,cost:2e3},{value:250,cost:2e4},{value:1200,cost:2e5},{value:6500,cost:2e6},{value:35e3,cost:2e7},{value:175e3,cost:2e8},{value:1e6,cost:2e9}],e[n.multiplier]=[{value:1,cost:0},{value:1.5,cost:50},{value:2,cost:300},{value:3,cost:2e3},{value:5,cost:12e3},{value:8,cost:85e3},{value:12,cost:7e5},{value:18,cost:65e5},{value:30,cost:65e6},{value:100,cost:1e9}],e[n.insurance]=[{value:0,cost:0},{value:10,cost:10},{value:25,cost:250},{value:40,cost:1e3},{value:50,cost:25e3},{value:70,cost:1e5},{value:80,cost:1e6},{value:90,cost:5e6},{value:95,cost:25e6},{value:99,cost:5e8}],e),u=[n.moneyPerQuestion,n.moneyPerQuestion,n.multiplier,n.streakBonus,n.streakBonus,n.multiplier,n.streakBonus,n.streakBonus,n.moneyPerQuestion,n.multiplier,n.streakBonus,n.multiplier,n.multiplier,n.moneyPerQuestion,n.streakBonus,n.moneyPerQuestion,n.multiplier,n.moneyPerQuestion,n.moneyPerQuestion,n.moneyPerQuestion,n.multiplier,n.streakBonus,n.moneyPerQuestion,n.moneyPerQuestion,n.moneyPerQuestion],c=function(){function t(){var t,e=this;this.isReady=function(t){return t>=e.nextup().money},this.isMax=function(t){return e.stats[t]+1>=10},this.currentPath=i([],u,!0),this.stats=((t={})[n.moneyPerQuestion]=0,t[n.streakBonus]=0,t[n.multiplier]=0,t[n.insurance]=0,t)}return t.prototype.upgrade=function(){this.stats[this.currentPath[0]]++,this.currentPath.shift()},t.prototype.nextup=function(){var t=this.currentPath[0];return void 0===t||this.isMax(t)?{upgrade:-1,money:-1}:{upgrade:t,money:a[t][this.stats[t]+1].cost}},t}(),l=function(){this.map=new Map};function p(t,e){return document.evaluate(t,null!=e?e:document,null,XPathResult.FIRST_ORDERED_NODE_TYPE).singleNodeValue}function d(t,e){t.hasAttribute("listening")||(t.addEventListener("click",e),t.setAttribute("listening",""))}var h=function(){function t(){this.triggers=[]}return t.prototype.addTrigger=function(t,e){var n=function(n){t.includes(n.code)&&e(n)};document.body.addEventListener("keydown",n),this.triggers.push(n)},t.prototype.close=function(){for(var t=0,e=this.triggers;t<e.length;t++){var n=e[t];document.body.removeEventListener("keydown",n)}},t}(),v=function(t){return p(".//*[text()]",t).textContent},y=function(){function t(t){var e=this;this.targets=t,this.observer=new MutationObserver((function(){return e.onUpdate()})),this.observer.observe(document.body,{subtree:!0,childList:!0,characterData:!0})}return t.prototype.onUpdate=function(){for(var t=function(t){var e=function(n){t.screen.element()?t.callback(t.screen):t.retryDelay&&n<1&&setTimeout((function(){return e(n+1)}),t.retryDelay)};e(0)},e=0,n=this.targets;e<n.length;e++){t(n[e])}},t}(),f=function(){function t(){this.element=function(){return p('//div[contains(text(), "$") and contains(@style, "font-weight: 900")]')}}return t.prototype.money=function(){var t,e=this.element(),n=null===(t=null==e?void 0:e.textContent)||void 0===t?void 0:t.replace(/[\$,\,]/g,"");return parseInt(n)},t}(),m=function(){function t(t){var e=this;this.root="//div[contains(@style, 'opacity: 1') and contains(@style, 'translateY')]",this.question=function(){return v(p("".concat(e.root,"/div[1]")))},this.choice=function(t){return p("".concat(e.root,"/div[2]/div[").concat(t+1,"]"))},this.choiceText=function(t){return function(t){for(var e=[],n=document.evaluate(t,document,null,XPathResult.ORDERED_NODE_SNAPSHOT_TYPE),o=0;o<n.snapshotLength;o++)e.push(n.snapshotItem(o));return e}("".concat(e.root,"/div[2]/div[.//span[text()='").concat(t,"']]"))},this.input=function(){return p("".concat(e.root,"/div[2]//input"))},this.inputSubmit=function(){return p("".concat(e.root,"/div[2]//div[text()='Submit']"))},this.store=t}return t.prototype.setup=function(){if(null!==this.element()){var t=this.question();null!==this.input()?this._setupInput(t):this._setupMultiChoice(t)}},t.prototype._setupInput=function(t){var e=this,n=this.input(),o=this.inputSubmit();void 0!==this.store.map.get(t)&&function(t,e){Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value").set.call(t,e),t.dispatchEvent(new InputEvent("input",{bubbles:!0}))}(n,this.store.map.get(t));var i=function(){e.store.pending={question:t,answer:n.value},s.close()},s=new h;s.addTrigger(["Enter"],i),d(o,i)},t.prototype._setupMultiChoice=function(t){var e=this;if(void 0!==this.store.map.get(t))for(var n=0,o=this.choiceText(this.store.map.get(t));n<o.length;n++){var i=o[n];i&&(i.style.border="5px solid white")}for(var s=new h,r=function(n){var o=a.choice(n);if(null!==o){var i=v(o),r={question:t,answer:i};s.addTrigger(["Digit".concat(n+1)],(function(){return e.store.pending=r})),d(o,(function(){return e.store.pending=r}))}},a=this,u=0;u<4;u++)r(u)},t.prototype.element=function(){return p(this.root)},t}(),g=function(){var t=this;this.verify=function(){return v(t.element()).includes("+")},this.element=function(){return p("//div[contains(@class, 'animated tada')]")}},b=function(){function t(t,e){var n=this;this.buttonRoot='//div[contains(@style, "white-space: nowrap")]',this.element=function(){return p("".concat(n.buttonRoot,'//div[text()="').concat(n.displayName,'"]'))},this.displayName=t,this.state=e}return t.prototype.level=function(){var t,e=null===(t=p('(//div[contains(@style, "color: gray")]/../div[2])[last()]'))||void 0===t?void 0:t.textContent;if(e)return parseInt(e.replace(/^\D+/g,""))-1},t.prototype.attachUpgradeTrigger=function(t){var e,n,o,i,r=this,u=s[this.displayName],c=this.level();if(void 0!==c&&c>this.state.stats[u]&&(console.info("Update level ".concat(this.state.stats[u]," -> ").concat(c)),this.state.stats[u]=c,t()),!this.state.isMax(u)){var l=null===(i=null===(o=null===(n=null===(e=p("".concat(this.buttonRoot,'//div[contains(text(), "$').concat(a[u][this.state.stats[u]+1].cost.toLocaleString("en-US"),'")]')))||void 0===e?void 0:e.parentElement)||void 0===n?void 0:n.parentElement)||void 0===o?void 0:o.parentElement)||void 0===i?void 0:i.parentElement;l&&(l.hasAttribute("disabled")||d(l,(function(){r.state.upgrade(),t()})))}},t}(),x=function(t,e){var n=this;this.fetchMoney=function(){return n._moneyIndicator.money()},this.state=t,this.eventHooks=e,this.store=new l,this._moneyIndicator=new f,this._questionScreen=new m(this.store),this._verifyScreen=new g,this.monitor=new y(i([{screen:this._moneyIndicator,callback:function(){return n.eventHooks.onMoney(n._moneyIndicator.money())}},{screen:this._questionScreen,retryDelay:200,callback:function(){n._questionScreen.setup()}},{screen:this._verifyScreen,callback:function(){n._verifyScreen.verify()&&n.store.pending&&(n.store.map.set(n.store.pending.question,n.store.pending.answer),n.store.pending=void 0)}}],Object.keys(s).map((function(e){return{screen:new b(e,t),callback:function(t){t.attachUpgradeTrigger((function(){return n.eventHooks.onUpgrade(s[e])}))}}})),!0)),this._questionScreen.setup()},k={top:"5%",right:"5%",position:"fixed",zIndex:9999,backdropFilter:"blur(4px)",border:"2px solid #8f8f8f",borderRadius:"5px",padding:"10px",backgroundColor:"#00000050"},w={fontWeight:"bold",marginLeft:"5px"},D={color:"white",fontSize:"18px"},P={backgroundColor:"transparent",border:"1px solid white",cursor:"pointer",borderRadius:"5px"},E=function(){function t(t){var e=this;this.hooks=t,this.root=document.createElement("div"),Object.assign(this.root.style,k),document.children[0].append(this.root);var n=document.createElement("div");n.style.display="flex",n.style.justifyContent="space-between";var i=document.createElement("div");i.style.display="flex",this.title=document.createElement("div"),this.title.textContent="Next up",this.readyDisplay=document.createElement("div"),Object.assign(this.readyDisplay.style,o(o({},D),w)),i.append(this.title,this.readyDisplay);var s=document.createElement("button");s.textContent="Skip",Object.assign(s.style,o(o({},P),D)),s.onclick=function(){return e.hooks.onSkip()},n.append(i,s),this.moneyDisplay=document.createElement("div"),this.targetDisplay=document.createElement("div"),Object.assign(this.title.style,D),Object.assign(this.moneyDisplay.style,D),Object.assign(this.targetDisplay.style,D),this.root.append(n,this.targetDisplay,this.moneyDisplay)}return t.prototype.updateReadiness=function(t){this.readyDisplay.innerHTML=t?"&#10003;":"&#10007;",this.readyDisplay.style.color=t?"#97ff90":"#ff5a5a"},t.prototype.updateGoal=function(t){if(t.money<0&&t.upgrade<0)return this.updateReadiness(!0),this.moneyDisplay.textContent="Money - DONE",void(this.targetDisplay.textContent="Upgrade - DONE");this.moneyDisplay.textContent="Money - ".concat(t.money.toLocaleString()),this.targetDisplay.textContent="Upgrade - ".concat(r[t.upgrade])},t}(),S=new c,R=new E({onSkip:function(){S.upgrade(),R.updateReadiness(S.isReady(_.fetchMoney())),R.updateGoal(S.nextup())}}),_=new x(S,{onMoney:function(t){return R.updateReadiness(S.isReady(t))},onUpgrade:function(){return R.updateGoal(S.nextup())}});R.updateReadiness(S.isReady(_.fetchMoney())),R.updateGoal(S.nextup());
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
