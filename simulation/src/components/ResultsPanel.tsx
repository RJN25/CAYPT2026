import type { PhysicsResult } from '../physics'

interface ResultsPanelProps {
  result: PhysicsResult | null
  className?: string
}

function Row({
  label,
  value,
  unit,
}: {
  label: string
  value: string | number
  unit?: string
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-slate-600/40 py-2 last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="font-tabular text-lg font-semibold text-white">
        {value}
        {unit != null && (
          <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>
        )}
      </span>
    </div>
  )
}

export function ResultsPanel({ result, className = '' }: ResultsPanelProps) {
  return (
    <div
      className={
        'rounded-lg border border-slate-600/60 bg-slate-800/80 p-4 backdrop-blur-sm ' +
        className
      }
    >
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Results (dependent variables)
      </h3>
      {result == null ? (
        <p className="text-sm text-slate-500">Run a simulation to see results.</p>
      ) : (
        <div className="space-y-0">
          <Row
            label="Impact speed v_impact"
            value={result.v_impact.toFixed(2)}
            unit="m/s"
          />
          <Row label="Froude number Fr" value={result.Fr.toFixed(2)} />
          <Row label="Weber number We" value={Math.round(result.We)} />
          <Row
            label="Geometry factor η_geom"
            value={result.eta_geom.toFixed(3)}
          />
          <Row
            label="Predicted max fountain height"
            value={result.H_max_cm.toFixed(1)}
            unit="cm"
          />
        </div>
      )}
    </div>
  )
}
