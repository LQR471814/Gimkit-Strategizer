import { idMap, Upgrade } from "./data"

const rootStyle = {
	top: "5%",
	right: "5%",
	position: "fixed",
	zIndex: 9999,
	backdropFilter: "blur(4px)",
	border: "2px solid #8f8f8f",
	borderRadius: "5px",
	padding: "10px",
	backgroundColor: "#00000050"
}

const readyChar = "&#10003;"
const unreadyChar = "&#10007;"

const unreadyColor = "#ff5a5a"
const readyColor = "#97ff90"

const readyDisplayStyle = {
	fontWeight: "bold",
	marginLeft: "5px"
}

const textStyle = {
	color: "white",
	fontSize: "18px"
}

const buttonStyle = {
	backgroundColor: "transparent",
	border: "1px solid white",
	cursor: "pointer",
	borderRadius: "5px"
}

export class GUIPanel {
	root: HTMLDivElement

	title: HTMLDivElement
	readyDisplay: HTMLDivElement

	moneyDisplay: HTMLDivElement
	targetDisplay: HTMLDivElement

	onSkip?: () => void

	constructor() {
		this.root = document.createElement("div")
		Object.assign(this.root.style, rootStyle)
		document.body.append(this.root)

		const labelContainer = document.createElement("div")
		labelContainer.style.display = "flex"
		labelContainer.style.justifyContent = "space-between"

		const titleContainer = document.createElement("div")
		titleContainer.style.display = "flex"

		this.title = document.createElement("div")
		this.title.textContent = "Next up"
		this.readyDisplay = document.createElement("div")
		Object.assign(this.readyDisplay.style, {...textStyle, ...readyDisplayStyle})

		titleContainer.append(this.title, this.readyDisplay)

		const skipButton = document.createElement("button")
		skipButton.textContent = "Skip"
		Object.assign(skipButton.style, {...buttonStyle, ...textStyle})
		skipButton.onclick = () => this.onSkip?.call(this)

		labelContainer.append(titleContainer, skipButton)

		this.moneyDisplay = document.createElement("div")
		this.targetDisplay = document.createElement("div")

		Object.assign(this.title.style, textStyle)
		Object.assign(this.moneyDisplay.style, textStyle)
		Object.assign(this.targetDisplay.style, textStyle)

		this.root.append(
			labelContainer,
			this.targetDisplay,
			this.moneyDisplay,
		)
	}

	updateReadiness(ready: boolean) {
		this.readyDisplay.innerHTML = ready ? readyChar : unreadyChar
		this.readyDisplay.style.color = ready ? readyColor : unreadyColor
	}

	updateGoal(money: number | null, target: Upgrade | null) {
		this.moneyDisplay.textContent = `Money - ${money !== null ? money.toLocaleString() : "DONE"}`
		this.targetDisplay.textContent = `Upgrade - ${target !== null ? idMap[target] : "DONE"}`
	}
}