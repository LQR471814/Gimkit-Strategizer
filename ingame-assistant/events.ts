import { PlayState } from "./backend"
import { nameMap, Upgrade, upgradeData } from "./data"
import { getXPathElement } from "./common"

export class GameEvents {
	state: PlayState
	onMoney?: (money: number) => void
	onUpgrade?: (upgrade: Upgrade) => void

	constructor(state: PlayState) {
		this.state = state
	}

	getMoneyElement() {
		return getXPathElement(
			'//div[contains(text(), "$") and contains(@style, "font-weight: 900")]'
		)
	}

	getMoney() {
		let moneyElement = this.getMoneyElement()
		let money = moneyElement?.textContent?.replace(/[\$,\,]/g, "")
		if (!money) return null
		return parseInt(money)
	}

	listenMoney() {
		const moneyElement = this.getMoneyElement()
		if (moneyElement !== null) {
			new MutationObserver(() => {
				let money = this.getMoney()
				if (money !== null) {
					this.onMoney?.call(this, money)
				}
			}).observe(
				moneyElement,
				{
					subtree: true,
					childList: true,
					characterData: true
				},
			)
		}
	}

	getLevel() {
		const levelStr = getXPathElement(
			'(//div[contains(@style, "color: gray")]/../div[2])[last()]'
		)?.textContent

		if (levelStr) {
			return parseInt(levelStr.replace(/^\D+/g, '')) - 1
		}
	}

	checkUpgrade(upgradeStr: string) {
		const id = nameMap[upgradeStr]
		const buttonRoot = '//div[contains(@style, "white-space: nowrap")]'
		const titleExpression = getXPathElement(
			`${buttonRoot}//div[text()="${upgradeStr}"]`
		)

		if (titleExpression) {
			const displayLevel = this.getLevel()
			console.info(`Update level ${this.state.stats[id]} -> ${displayLevel}`)
			if (displayLevel) {
				this.state.stats[id] = displayLevel
			}

			const button = getXPathElement(
				`${buttonRoot}//div[contains(text(), "$${
					upgradeData[id][
						this.state.stats[id]+1
					].cost.toLocaleString("en-US")
				}")]`
			)?.parentElement?.parentElement?.parentElement?.parentElement
			if (!button) return

			if (
				button.getAttribute("listening") === "true" ||
				button.hasAttribute("disabled")
			) return

			button.addEventListener("click", () => {
				this.onUpgrade?.call(this, id)
			})
			button.setAttribute("listening", "true")
		}
	}

	listenUpgrade() {
		let upgradeObserver = new MutationObserver(() => {
			for (const upgradeStr in nameMap) {
				this.checkUpgrade(upgradeStr)
			}
		})

		upgradeObserver.observe(
			document,
			{
				subtree: true,
				characterData: true,
				childList: true
			}
		)
	}

	listen() {
		this.listenMoney()
		this.listenUpgrade()
	}
}
