export function getXPathElement(expression: string, contextNode: Node | null = document) {
	return document.evaluate(
		expression, contextNode ?? document, null,
		XPathResult.FIRST_ORDERED_NODE_TYPE
	).singleNodeValue
}
