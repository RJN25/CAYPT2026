import { useRef, useState, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { InputParams, PhysicsResult } from '../physics'
import { G, jetHeightAtTime, jetTimeOfFlight } from '../physics'

interface SimulationSceneProps {
  params: InputParams
  result: PhysicsResult | null
  runTrigger: number
  onSimulationComplete: () => void
  tankWidth: number
  tankDepth: number
  tankHeight: number
  waterLevel: number
}

export function SimulationScene({
  params,
  result,
  runTrigger,
  onSimulationComplete,
  tankWidth,
  tankDepth,
  tankHeight,
  waterLevel,
}: SimulationSceneProps) {
  const [simTime, setSimTime] = useState(0)
  const runningRef = useRef(false)
  const prevTriggerRef = useRef(runTrigger)

  const H_drop_m = params.H_drop_cm / 100
  const v_impact = result ? result.v_impact : Math.sqrt(2 * G * H_drop_m)
  const t_impact = v_impact / G
  const u_j = result ? result.u_j : 0
  const t_jet_end = result ? jetTimeOfFlight(u_j) : 0

  // Reset simulation when run is triggered
  useEffect(() => {
    if (runTrigger !== prevTriggerRef.current) {
      prevTriggerRef.current = runTrigger
      setSimTime(0)
      runningRef.current = true
    }
  }, [runTrigger])

  useFrame((_, delta) => {
    if (!runningRef.current) return
    setSimTime((t) => {
      const next = t + delta
      const totalDuration = t_impact + t_jet_end + 0.5
      if (next >= totalDuration) {
        runningRef.current = false
        onSimulationComplete()
        return next
      }
      return next
    })
  })

  return (
    <group>
      <Tank
        width={tankWidth}
        depth={tankDepth}
        height={tankHeight}
      />
      <Water
        width={tankWidth}
        depth={tankDepth}
        waterLevel={waterLevel}
      />
      <FallingRing
        params={params}
        simTime={simTime}
        t_impact={t_impact}
        waterLevel={waterLevel}
      />
      {result && (
        <>
          <Jet
            u_j={u_j}
            simTime={simTime}
            t_impact={t_impact}
            waterLevel={waterLevel}
          />
          <Ripples
            simTime={simTime}
            t_impact={t_impact}
            waterLevel={waterLevel}
            radius={tankWidth * 0.4}
          />
        </>
      )}
    </group>
  )
}

// --- Tank (glass-like walls) ---
function Tank({
  width,
  depth,
  height,
}: {
  width: number
  depth: number
  height: number
}) {
  const thickness = 0.04
  const geo = useMemo(
    () =>
      new THREE.BoxGeometry(
        width + 2 * thickness,
        height + thickness,
        depth + 2 * thickness
      ),
    [width, depth, height]
  )
  return (
    <mesh
      position={[0, (height + thickness) / 2, 0]}
      geometry={geo}
      receiveShadow
    >
      <meshStandardMaterial
        color="#1e3a5f"
        transparent
        opacity={0.4}
        roughness={0.2}
        metalness={0.1}
      />
    </mesh>
  )
}

// --- Water surface ---
function Water({
  width,
  depth,
  waterLevel,
}: {
  width: number
  depth: number
  waterLevel: number
}) {
  return (
    <mesh
      position={[0, waterLevel, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
    >
      <planeGeometry args={[width, depth]} />
      <meshStandardMaterial
        color="#0e4c92"
        roughness={0.1}
        metalness={0.05}
        transparent
        opacity={0.92}
        envMapIntensity={0.5}
      />
    </mesh>
  )
}

// --- Ring: falls from H_drop, then rests at water level ---
function FallingRing({
  params,
  simTime,
  t_impact,
  waterLevel,
}: {
  params: InputParams
  simTime: number
  t_impact: number
  waterLevel: number
}) {
  const D_o_m = params.D_o_cm / 100
  const D_i_m = params.D_i_cm / 100
  const R = (D_o_m + D_i_m) / 4
  const r = (D_o_m - D_i_m) / 4
  const tubeGeo = useRef<THREE.TorusGeometry | null>(null)
  if (!tubeGeo.current) tubeGeo.current = new THREE.TorusGeometry(R, r, 32, 48)

  // Ballistic fall: y = H_drop - 0.5*g*t^2 until impact
  const H_drop_m = params.H_drop_cm / 100
  let y: number
  if (simTime < t_impact) {
    const t = simTime
    y = H_drop_m - 0.5 * G * t * t
  } else {
    y = waterLevel + 0.002
  }

  return (
    <mesh position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]} geometry={tubeGeo} castShadow>
      <meshStandardMaterial color="#94a3b8" metalness={0.6} roughness={0.3} />
    </mesh>
  )
}

// --- Worthington-like jet: tapered cone, ballistic height, slight wobble ---
const JET_RADIUS_BASE = 0.04
const JET_HEIGHT_UNIT = 1
const WOBBLE_AMPLITUDE = 0.008
const WOBBLE_DECAY = 2.5

const jetGeometry = new THREE.CylinderGeometry(
  JET_RADIUS_BASE * 0.15,
  JET_RADIUS_BASE,
  JET_HEIGHT_UNIT,
  24
)

function Jet({
  u_j,
  simTime,
  t_impact,
  waterLevel,
}: {
  u_j: number
  simTime: number
  t_impact: number
  waterLevel: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const t_jet = Math.max(0, simTime - t_impact)
  const h = jetHeightAtTime(u_j, t_jet)
  const wobbleSeed = useRef(Math.random() * 1000)

  useFrame((state) => {
    if (!meshRef.current) return
    const decay = Math.exp(-WOBBLE_DECAY * t_jet)
    const nx =
      (Math.sin(state.clock.elapsedTime * 12 + wobbleSeed.current) *
        WOBBLE_AMPLITUDE +
        Math.sin(state.clock.elapsedTime * 7) * WOBBLE_AMPLITUDE * 0.5) *
      decay
    const nz =
      (Math.cos(state.clock.elapsedTime * 11 + wobbleSeed.current * 1.3) *
        WOBBLE_AMPLITUDE +
        Math.cos(state.clock.elapsedTime * 8) * WOBBLE_AMPLITUDE * 0.5) *
      decay
    meshRef.current.position.set(nx, waterLevel + h / 2, nz)
    meshRef.current.scale.set(1, h / JET_HEIGHT_UNIT, 1)
  })

  if (h <= 0.001) return null

  return (
    <mesh
      ref={meshRef}
      position={[0, waterLevel + h / 2, 0]}
      geometry={jetGeometry}
      scale={[1, h / JET_HEIGHT_UNIT, 1]}
      castShadow
    >
      <meshStandardMaterial
        color="#7dd3fc"
        transparent
        opacity={0.9}
        roughness={0.2}
        metalness={0.1}
        emissive="#38bdf8"
        emissiveIntensity={0.15}
      />
    </mesh>
  )
}

// --- Ripples: expanding ring on water surface ---
const RIPPLE_DURATION = 1.2
const RIPPLE_MAX_SCALE = 2.5

function Ripples({
  simTime,
  t_impact,
  waterLevel,
  radius,
}: {
  simTime: number
  t_impact: number
  waterLevel: number
  radius: number
}) {
  const t_ripple = simTime - t_impact
  if (t_ripple < 0 || t_ripple > RIPPLE_DURATION) return null

  const progress = t_ripple / RIPPLE_DURATION
  const scale = 0.1 + progress * RIPPLE_MAX_SCALE
  const opacity = 0.4 * (1 - progress) * (1 - progress)

  return (
    <mesh
      position={[0, waterLevel + 0.001, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      scale={[scale, scale, 1]}
    >
      <ringGeometry args={[radius * 0.3, radius, 64]} />
      <meshBasicMaterial
        color="#7dd3fc"
        transparent
        opacity={opacity}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}
