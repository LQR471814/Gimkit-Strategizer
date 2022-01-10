import { PlayState } from "./backend";
import { GameEvents } from "./events";
import { GUIPanel } from "./gui";

const state = new PlayState()
const gui = new GUIPanel({
	onSkip: () => {
		state.upgrade()
		gui.updateGoal(state.nextup())
	}
})

const events = new GameEvents(state, {
	onMoney: money => gui.updateReadiness(state.isReady(money)),
	onUpgrade: () => gui.updateGoal(state.nextup())
})

gui.updateReadiness(state.isReady(events.fetchMoney()))
gui.updateGoal(state.nextup())
