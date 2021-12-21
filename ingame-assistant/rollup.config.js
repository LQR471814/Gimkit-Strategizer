import typescript from 'rollup-plugin-typescript'
import { terser } from 'rollup-plugin-terser'

export default {
	input: "main.ts",
	plugins: [
		typescript(),
		terser()
	],
	output: {
		file: 'assistant.min.js'
	}
}