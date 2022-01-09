import { PlayState } from "./backend"
import { maxLevel, nameMap, Upgrade, upgradeData } from "./data"
import { getXPathElement } from "./common"

export enum Screens {
	QUESTION,
	SHOP_LEVELS_SCREEN,
	MONEY_INDICATOR,
}

type MonitorTarget = { screen: GameScreen, callback: (element: Node, screen: GameScreen) => void }

class ScreenMonitor {
	targets: MonitorTarget[]
	observer: MutationObserver

	constructor(targets: MonitorTarget[]) {
		this.targets = targets
		this.observer = new MutationObserver(() => this.onUpdate())
		this.observer.observe(document.body, {
			subtree: true,
			childList: true,
			characterData: true,
		})
	}

	onUpdate() {
		for (const target of this.targets) {
			const element = target.screen.element()
			if (element !== null) {
				target.callback(element, target.screen)
			}
		}
	}
}

abstract class GameScreen {
	element(): Node | null {
		return null
	}
}


class MoneyIndicator implements GameScreen {
	money() {
		let moneyElement = this.element()
		let money = moneyElement?.textContent?.replace(/[\$,\,]/g, "")
		return parseInt(money!)
	}

	element = () => getXPathElement(
		'//div[contains(text(), "$") and contains(@style, "font-weight: 900")]'
	)
}

class QuestionScreen implements GameScreen {
	question() {
		return getXPathElement("//span", this.element())
	}

	element() {
		return getXPathElement("//div[contains(@class, 'enter-done')]")
	}
}

class ShopLevelsScreen extends GameScreen {
	state: PlayState
	displayName: string
	buttonRoot = '//div[contains(@style, "white-space: nowrap")]'

	constructor(name: string, state: PlayState) {
		super()
		this.displayName = name
		this.state = state
	}

	level() {
		const levelStr = getXPathElement(
			'(//div[contains(@style, "color: gray")]/../div[2])[last()]'
		)?.textContent

		if (levelStr) {
			return parseInt(levelStr.replace(/^\D+/g, '')) - 1
		}
	}

	attachUpgradeTrigger(trigger: () => void) {
		const id = nameMap[this.displayName]
		const displayLevel = this.level()

		if (displayLevel !== undefined) {
			if (displayLevel > this.state.stats[id]) {
				console.info(`Update level ${this.state.stats[id]} -> ${displayLevel}`)
				this.state.stats[id] = displayLevel
				trigger()
			}
		}

		if (this.state.isMax(id)) return

		const button = getXPathElement(
			`${this.buttonRoot}//div[contains(text(), "$${upgradeData[id][
				this.state.stats[id] + 1
			].cost.toLocaleString("en-US")
			}")]`
		)?.parentElement?.parentElement?.parentElement?.parentElement
		if (!button) return

		if (
			button.getAttribute("listening") === "true" ||
			button.hasAttribute("disabled")
		) return

		button.addEventListener("click", () => {
			this.state.upgrade()
			trigger()
		})

		button.setAttribute("listening", "true")
	}

	element = () => getXPathElement(
		`${this.buttonRoot}//div[text()="${this.displayName}"]`
	)
}

type EventHooks = {
	onMoney: (money: number) => void
	onUpgrade: (upgrade: Upgrade) => void
}

export class GameEvents {
	state: PlayState
	monitor: ScreenMonitor
	eventHooks: EventHooks

	private _moneyIndicator: MoneyIndicator

	constructor(state: PlayState, eventHooks: EventHooks) {
		this.state = state
		this.eventHooks = eventHooks

		this._moneyIndicator = new MoneyIndicator()

		this.monitor = new ScreenMonitor([
			{
				screen: this._moneyIndicator,
				callback: () => this.eventHooks.onMoney(this._moneyIndicator.money())
			},
			{
				screen: new QuestionScreen(),
				callback: () => {}
			},
			...(
				Object.keys(nameMap).map(
					e => ({
						screen: new ShopLevelsScreen(e, state),
						callback: (_, s) => {
							const screen = s as ShopLevelsScreen
							screen.attachUpgradeTrigger(() =>
								this.eventHooks.onUpgrade(nameMap[e])
							)
						}
					} as MonitorTarget)
				)
			)
		])
	}

	fetchMoney = () => this._moneyIndicator.money()
}
