import { useState, useCallback } from 'react'
import { computePhysics } from './physics'
import type { InputParams, PhysicsResult, WettingMode } from './physics'
import type { RunRecord } from './types'
import { ControlPanel } from './components/ControlPanel'
import { ResultsPanel } from './components/ResultsPanel'
import { RunHistoryTable } from './components/RunHistoryTable'
import { Scene3D } from './components/Scene3D'

// Defaults: impressive but plausible H_max
const DEFAULT_PARAMS: InputParams = {
  H_drop_cm: 72,
  D_o_cm: 5,
  D_i_cm: 4,
  m_g: 20,
  wetting: 'neutral' as WettingMode,
}

function nextId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export default function App() {
  const [params, setParams] = useState<InputParams>(DEFAULT_PARAMS)
  const [result, setResult] = useState<PhysicsResult | null>(null)
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [runTrigger, setRunTrigger] = useState(0)
  const [running, setRunning] = useState(false)

  const handleRun = useCallback(() => {
    const res = computePhysics(params)
    setResult(res)
    setRuns((prev) => [
      { id: nextId(), params: { ...params }, result: res, timestamp: Date.now() },
      ...prev,
    ])
    setRunTrigger((t) => t + 1)
    setRunning(true)
  }, [params])

  const handleSimulationComplete = useCallback(() => {
    setRunning(false)
  }, [])

  return (
    <div className="flex min-h-screen flex-col bg-slate-900 text-slate-100 lg:flex-row">
      {/* Left: 3D scene + run history */}
      <div className="relative flex-1">
        <div className="absolute left-3 top-3 z-10 w-64">
          <RunHistoryTable runs={runs} />
        </div>
        <div className="h-[50vh] lg:h-screen">
          <Scene3D
            params={params}
            result={result}
            runTrigger={runTrigger}
            onSimulationComplete={handleSimulationComplete}
          />
        </div>
      </div>

      {/* Right: controls + results */}
      <aside className="flex w-full flex-col gap-4 border-l border-slate-700 bg-slate-900/95 p-4 lg:w-[380px] lg:flex-shrink-0">
        <ControlPanel
          params={params}
          onChange={setParams}
          onRun={handleRun}
          running={running}
        />
        <ResultsPanel result={result} />
      </aside>
    </div>
  )
}
