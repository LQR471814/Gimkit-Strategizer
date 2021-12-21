export function getXPathElement(expression: string) {
	return document.evaluate(
		expression, document, null,
		XPathResult.FIRST_ORDERED_NODE_TYPE
	).singleNodeValue
}
