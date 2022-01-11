export function getXPathElement(expression: string, contextNode?: Node) {
	return document.evaluate(
		expression, contextNode ?? document, null,
		XPathResult.FIRST_ORDERED_NODE_TYPE
	).singleNodeValue
}

export function getMultipleXPathElements(expression: string) {
	const elements = []
	const evalResult = document.evaluate(
		expression, document, null,
		XPathResult.ORDERED_NODE_SNAPSHOT_TYPE
	)

	for (let i = 0; i < evalResult.snapshotLength; i++) {
		elements.push(evalResult.snapshotItem(i))
	}

	return elements
}

export function addTrigger(element: HTMLElement, onclick: () => void) {
	if (element.hasAttribute("listening")) return
	element.addEventListener("click", onclick)
	element.setAttribute("listening", "")
}

export class KeyboardTriggerGroup {
	private triggers: ((ev: KeyboardEvent) => void)[]

	constructor() {
		this.triggers = []
	}

	addTrigger(whitelist: string[], trigger: (ev: KeyboardEvent) => void) {
		const func = (ev: KeyboardEvent) => {
			if (whitelist.includes(ev.code)) trigger(ev)
		}
		document.body.addEventListener("keydown", func)
		this.triggers.push(func)
	}

	close() {
		for (const trigger of this.triggers) {
			document.body.removeEventListener("keydown", trigger)
		}
	}
}

export const textOf = (element: Node) => getXPathElement(".//*[text()]", element)!.textContent!

export const setInputValue = (input: HTMLInputElement, value: string) => {
	//? This is needed because react overrides the default "input" setter
	const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
		window.HTMLInputElement.prototype, "value"
	)!.set!;
	nativeInputValueSetter.call(input, value)
	//? Then to trigger an internal input listener
	input.dispatchEvent(new InputEvent("input", { bubbles: true }))
}
