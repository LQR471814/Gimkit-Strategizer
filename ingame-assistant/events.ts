import { PlayState, QuestionStore } from "./backend"
import { addKeyboardTrigger, addTrigger, getMultipleXPathElements, getXPathElement, textOf } from "./common"
import { nameMap, Upgrade, upgradeData } from "./data"

type MonitorTarget = {
	screen: GameScreen,
	callback: (screen: GameScreen) => void,
	retryDelay?: number
}

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
			const checkCall = (depth: number) => {
				if (target.screen.element()) {
					target.callback(target.screen)
				} else if (target.retryDelay && depth < 1) {
					setTimeout(() => checkCall(depth+1), target.retryDelay)
				}
			}
			checkCall(0)
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
	root = "//div[contains(@style, 'opacity: 1') and contains(@style, 'translateY')]"
	store: QuestionStore

	constructor(store: QuestionStore) {
		this.store = store
	}

	question = () => textOf(getXPathElement(`${this.root}/div[1]`)!)

	choice = (index: number) => getXPathElement(`${this.root}/div[2]/div[${index + 1}]`)

	choiceText = (text: string) => getMultipleXPathElements(`${this.root}/div[2]/div[.//span[text()='${text}']]`)

	setup() {
		if (this.element() === null) return

		const question = this.question()
		if (this.store.map.get(question) !== undefined) {
			const choices = this.choiceText(this.store.map.get(question)!)
			for (const choice of choices) {
				if (choice) {
					(choice as HTMLElement).style.border = "5px solid white"
				}
			}
			return
		}

		for (let i = 0; i < 4; i++) {
			const choice = this.choice(i)
			if (choice !== null) {
				const choiceText = textOf(choice)
				const answer = {
					question: question,
					answer: choiceText
				}

				addKeyboardTrigger(
					[`Digit${i + 1}`],
					() => this.store.pending = answer
				)

				addTrigger(
					choice as HTMLElement,
					() => this.store.pending = answer
				)
			}
		}

		return
	}

	element() {
		return getXPathElement(this.root)
	}
}

class VerificationScreen implements GameScreen {
	verify = (): boolean => textOf(this.element()!).includes("+")
	element = () => getXPathElement("//div[contains(@class, 'animated tada')]")
}

class ShopLevelsScreen implements GameScreen {
	state: PlayState
	displayName: string
	buttonRoot = '//div[contains(@style, "white-space: nowrap")]'

	constructor(name: string, state: PlayState) {
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
			button.hasAttribute("disabled")
		) return

		addTrigger(button, () => {
			this.state.upgrade()
			trigger()
		})
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
	store: QuestionStore

	private _moneyIndicator: MoneyIndicator
	private _questionScreen: QuestionScreen
	private _verifyScreen: VerificationScreen

	constructor(state: PlayState, eventHooks: EventHooks) {
		this.state = state
		this.eventHooks = eventHooks
		this.store = new QuestionStore()

		this._moneyIndicator = new MoneyIndicator()
		this._questionScreen = new QuestionScreen(this.store)
		this._verifyScreen = new VerificationScreen()

		this.monitor = new ScreenMonitor([
			{
				screen: this._moneyIndicator,
				callback: () => this.eventHooks.onMoney(this._moneyIndicator.money())
			},
			{
				screen: this._questionScreen,
				retryDelay: 200,
				callback: () => {
					this._questionScreen.setup()
				}
			},
			{
				screen: this._verifyScreen,
				callback: () => {
					if (this._verifyScreen.verify() && this.store.pending) {
						this.store.map.set(
							this.store.pending!.question,
							this.store.pending!.answer,
						)
						this.store.pending = undefined
					}
				}
			},
			...(
				Object.keys(nameMap).map(
					e => ({
						screen: new ShopLevelsScreen(e, state),
						callback: (s) => {
							const screen = s as ShopLevelsScreen
							screen.attachUpgradeTrigger(() =>
								this.eventHooks.onUpgrade(nameMap[e])
							)
						}
					} as MonitorTarget)
				)
			)
		])

		this._questionScreen.setup()
	}

	fetchMoney = () => this._moneyIndicator.money()
}
