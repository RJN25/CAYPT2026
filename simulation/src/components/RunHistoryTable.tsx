import type { RunRecord } from '../types'

const MAX_ROWS = 10

interface RunHistoryTableProps {
  runs: RunRecord[]
  className?: string
}

export function RunHistoryTable({ runs, className = '' }: RunHistoryTableProps) {
  const display = runs.slice(0, MAX_ROWS)

  return (
    <div
      className={
        'rounded-lg border border-slate-600/60 bg-slate-800/80 backdrop-blur-sm ' +
        className
      }
    >
      <div className="border-b border-slate-600/60 px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Recent runs
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-600/40 text-slate-400">
              <th className="px-2 py-1.5 font-medium">H (cm)</th>
              <th className="px-2 py-1.5 font-medium">D_o (cm)</th>
              <th className="px-2 py-1.5 font-medium">D_i (cm)</th>
              <th className="px-2 py-1.5 font-medium">m (g)</th>
              <th className="px-2 py-1.5 font-medium">H_max (cm)</th>
            </tr>
          </thead>
          <tbody>
            {display.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-2 py-3 text-slate-500">
                  No runs yet
                </td>
              </tr>
            ) : (
              display.map((r) => (
                <tr
                  key={r.id}
                  className="border-b border-slate-700/40 font-tabular last:border-0"
                >
                  <td className="px-2 py-1.5">{r.params.H_drop_cm}</td>
                  <td className="px-2 py-1.5">{r.params.D_o_cm}</td>
                  <td className="px-2 py-1.5">{r.params.D_i_cm}</td>
                  <td className="px-2 py-1.5">{r.params.m_g}</td>
                  <td className="px-2 py-1.5 text-cyan-300">
                    {r.result.H_max_cm.toFixed(2)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
