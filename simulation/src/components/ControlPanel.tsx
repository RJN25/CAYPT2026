import type { InputParams, WettingMode } from '../physics'

interface ControlPanelProps {
  params: InputParams
  onChange: (params: InputParams) => void
  onRun: () => void
  running: boolean
  className?: string
}

const WETTING_OPTIONS: { value: WettingMode; label: string }[] = [
  { value: 'hydrophilic', label: 'Hydrophilic' },
  { value: 'neutral', label: 'Neutral' },
  { value: 'hydrophobic', label: 'Hydrophobic' },
]

export function ControlPanel({
  params,
  onChange,
  onRun,
  running,
  className = '',
}: ControlPanelProps) {
  const D_o = params.D_o_cm
  const D_iMin = 0.5
  const D_iMax = Math.max(D_iMin, D_o - 0.5)
  const D_iClamped = Math.max(D_iMin, Math.min(D_iMax, params.D_i_cm))

  const update = (patch: Partial<InputParams>) => {
    onChange({ ...params, ...patch })
  }

  return (
    <div
      className={
        'flex flex-col gap-6 rounded-xl border border-slate-600/60 bg-slate-900/95 p-5 shadow-xl backdrop-blur-sm ' +
        className
      }
    >
      <h2 className="text-lg font-semibold text-white">
        Ring Fountain Simulator
      </h2>
      <p className="text-xs text-slate-400">
        Independent variables — adjust and run to see the 3D simulation and
        results.
      </p>

      <div className="space-y-4">
        <label className="block">
          <span className="mb-1 block text-sm text-slate-300">
            Drop height H_drop (cm)
          </span>
          <input
            type="range"
            min={10}
            max={150}
            step={1}
            value={params.H_drop_cm}
            onChange={(e) =>
              update({ H_drop_cm: Number(e.target.value) })
            }
            className="h-2 w-full accent-cyan-500"
          />
          <span className="mt-1 block font-tabular text-sm text-cyan-300">
            {params.H_drop_cm} cm
          </span>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm text-slate-300">
            Outer diameter D_o (cm)
          </span>
          <input
            type="range"
            min={2}
            max={15}
            step={0.1}
            value={params.D_o_cm}
            onChange={(e) => {
              const newDo = Number(e.target.value)
              const newDiMax = Math.max(0.5, newDo - 0.5)
              update({
                D_o_cm: newDo,
                D_i_cm: Math.min(params.D_i_cm, newDiMax),
              })
            }}
            className="h-2 w-full accent-cyan-500"
          />
          <span className="mt-1 block font-tabular text-sm text-cyan-300">
            {params.D_o_cm.toFixed(1)} cm
          </span>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm text-slate-300">
            Inner diameter D_i (cm) — 0.5 to {D_iMax.toFixed(1)}
          </span>
          <input
            type="range"
            min={D_iMin}
            max={D_iMax}
            step={0.1}
            value={D_iClamped}
            onChange={(e) =>
              update({ D_i_cm: Number(e.target.value) })
            }
            className="h-2 w-full accent-cyan-500"
          />
          <span className="mt-1 block font-tabular text-sm text-cyan-300">
            {D_iClamped.toFixed(1)} cm
          </span>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm text-slate-300">
            Ring mass m (g)
          </span>
          <input
            type="range"
            min={5}
            max={200}
            step={1}
            value={params.m_g}
            onChange={(e) => update({ m_g: Number(e.target.value) })}
            className="h-2 w-full accent-cyan-500"
          />
          <span className="mt-1 block font-tabular text-sm text-cyan-300">
            {params.m_g} g
          </span>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm text-slate-300">
            Wetting
          </span>
          <select
            value={params.wetting}
            onChange={(e) =>
              update({ wetting: e.target.value as WettingMode })
            }
            className="w-full rounded border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          >
            {WETTING_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        type="button"
        onClick={onRun}
        disabled={running}
        className="w-full rounded-lg bg-cyan-600 py-3 font-semibold text-white transition hover:bg-cyan-500 disabled:opacity-60 disabled:hover:bg-cyan-600"
      >
        {running ? 'Simulation in progress…' : 'Run Simulation'}
      </button>
    </div>
  )
}
