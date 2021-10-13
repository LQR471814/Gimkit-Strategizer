import { parentPort, workerData } from 'worker_threads'
import { optimize } from './strategizer'

parentPort?.postMessage(optimize(
	workerData.options,
	workerData.sequence,
	workerData.depth,
	workerData.max
))
