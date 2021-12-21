import { PlayState } from "./backend";
import { upgradeData } from "./data";
import { GameEvents } from "./events";
import { GUIPanel } from "./gui";

function getNextUpgrade(state: PlayState) {
	if (state.currentPath.length > 0) {
		return state.currentPath[0]
	}
	return null
}

function getMoneyGoal(state: PlayState) {
	const upgrade = getNextUpgrade(state)
	if (upgrade !== null) {
		return upgradeData[upgrade][state.stats[upgrade]+1].cost
	}
	return null
}

const state = new PlayState()
const events = new GameEvents(state)
const gui = new GUIPanel()

const updateGUIReadiness = (money: number | null) => {
	if (money === null) {
		money = events.getMoney()
	}

	let moneyGoal = getMoneyGoal(state)
	console.log(money, moneyGoal)
	if (money !== null && moneyGoal !== null) {
		gui.updateReadiness(money >= moneyGoal)
	}
}

const updateGUIGoal = () =>
	gui.updateGoal(getMoneyGoal(state), getNextUpgrade(state))

gui.onSkip = () => {
	state.upgrade()
	updateGUIReadiness(null)
	updateGUIGoal()
}

events.onMoney = (money) => updateGUIReadiness(money)

events.onUpgrade = (upgrade) => {
	if (state.currentPath.length > 0 && state.currentPath[0] !== upgrade) {
		alert("Incorrect upgrade purchased!")
	}
	state.upgrade()
	updateGUIReadiness(null)
	updateGUIGoal()
}

events.listen()

updateGUIReadiness(null)
updateGUIGoal()
