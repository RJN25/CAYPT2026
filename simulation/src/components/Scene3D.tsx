import { Canvas } from '@react-three/fiber'
import { Environment } from '@react-three/drei'
import type { InputParams, PhysicsResult } from '../physics'
import { SimulationScene } from './SimulationScene'

/** Scale: 1 unit = 1 m. Tank ~1.2m wide, 0.6m deep. */
const TANK_WIDTH = 1.2
const TANK_DEPTH = 1.2
const TANK_HEIGHT = 0.6
const WATER_LEVEL = 0

interface Scene3DProps {
  params: InputParams
  result: PhysicsResult | null
  runTrigger: number
  onSimulationComplete: () => void
}

export function Scene3D({
  params,
  result,
  runTrigger,
  onSimulationComplete,
}: Scene3DProps) {
  return (
    <div className="h-full w-full min-h-[400px] bg-slate-900">
      <Canvas
        camera={{ position: [2.2, 1.2, 2.2], fov: 45 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#0f172a']} />
        <ambient intensity={0.4} />
        <directionalLight position={[4, 6, 4]} intensity={1.2} castShadow />
        <directionalLight position={[-2, 4, -2]} intensity={0.4} />
        <Environment preset="night" />
        <SimulationScene
          params={params}
          result={result}
          runTrigger={runTrigger}
          onSimulationComplete={onSimulationComplete}
          tankWidth={TANK_WIDTH}
          tankDepth={TANK_DEPTH}
          tankHeight={TANK_HEIGHT}
          waterLevel={WATER_LEVEL}
        />
      </Canvas>
    </div>
  )
}

