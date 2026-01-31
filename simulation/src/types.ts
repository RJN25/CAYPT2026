import type { InputParams, PhysicsResult, WettingMode } from './physics'

export type { InputParams, PhysicsResult, WettingMode }

export interface RunRecord {
  id: string
  params: InputParams
  result: PhysicsResult
  timestamp: number
}
