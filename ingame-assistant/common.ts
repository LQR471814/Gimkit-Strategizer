export function getXPathElement(expression: string, contextNode?: Node) {
	return document.evaluate(
		`.${expression}`, contextNode ?? document, null,
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

export const addKeyboardTrigger = (keys: string[], onkey: (ev: KeyboardEvent) => void) =>
	document.body.addEventListener("keydown", (e) => {
		if (keys.includes(e.code)) onkey(e)
	}, { once: true })

export const textOf = (element: Node) => getXPathElement("//*[text()]", element)!.textContent!
