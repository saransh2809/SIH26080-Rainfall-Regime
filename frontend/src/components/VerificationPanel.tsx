import { useState } from 'react'
import type { ModelScores, VerificationReport } from '../api'
import { hasKey, useT } from '../i18n'

type Direction = 'lower' | 'higher' | 'zero'

interface Row {
  label: string
  get: (s: ModelScores) => number | null | undefined
  better: Direction
}

const HEAVY = '64.5'

const ROWS: Row[] = [
  { label: 'RMSE (mm)', get: (s) => s.rmse, better: 'lower' },
  { label: 'MAE (mm)', get: (s) => s.mae, better: 'lower' },
  { label: 'Bias (mm)', get: (s) => s.bias, better: 'zero' },
  { label: 'Correlation', get: (s) => s.correlation, better: 'higher' },
  { label: `POD ≥${HEAVY} mm`, get: (s) => s.categorical[HEAVY]?.pod, better: 'higher' },
  { label: `FAR ≥${HEAVY} mm`, get: (s) => s.categorical[HEAVY]?.far, better: 'lower' },
  { label: `CSI ≥${HEAVY} mm`, get: (s) => s.categorical[HEAVY]?.csi, better: 'higher' },
  { label: `ETS ≥${HEAVY} mm`, get: (s) => s.categorical[HEAVY]?.ets, better: 'higher' },
  { label: `FSS ≥${HEAVY} mm (9×9)`, get: (s) => s.fss[HEAVY]?.['9'], better: 'higher' },
]

function bestIndex(values: (number | null | undefined)[], better: Direction): number {
  let best = -1
  values.forEach((v, i) => {
    if (v == null || Number.isNaN(v)) return
    const cur = values[best]
    if (best < 0 || cur == null) best = i
    else if (better === 'lower' ? v < cur : better === 'higher' ? v > cur : Math.abs(v) < Math.abs(cur)) best = i
  })
  return best
}

/** Real validation scores; undefined scores show "n/a", never 0. */
export function VerificationPanel({ report, title, models: only }: { report: VerificationReport; title: string; models?: string[] }) {
  const leads = Object.keys(report.results)
  const t = useT()
  const modelLabel = (m: string) => (hasKey(`model_${m}`) ? t(`model_${m}` as 'model_A_raw_nwp') : m)
  const [lead, setLead] = useState(leads[0])
  const byModel = report.results[lead]
  const models = Object.keys(byModel).filter((m) => !only || only.includes(m))

  return (
    <section className="panel" aria-labelledby="verif-title">
      <h2 id="verif-title">{title}</h2>
      <p className="muted">{report.scope} · data: {report.data_kind.toUpperCase()}</p>
      <div className="segmented" role="group" aria-label={t('verifLead')}>
        {leads.map((l) => (
          <button key={l} type="button" aria-pressed={l === lead} onClick={() => setLead(l)}>
            {t('day', { n: l })}
          </button>
        ))}
      </div>
      <table className="scores" style={{ marginTop: 12 }}>
        <thead>
          <tr>
            <th scope="col">{t('metric')}</th>
            {models.map((m) => <th key={m} scope="col">{modelLabel(m)}</th>)}
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => {
            const values = models.map((m) => row.get(byModel[m]))
            const best = bestIndex(values, row.better)
            return (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                {values.map((v, i) => (
                  <td key={models[i]} className={`num${i === best ? ' best' : ''}`}>
                    {v == null || Number.isNaN(v) ? t('na') : v.toFixed(3)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="muted" style={{ marginTop: 8 }}>{t('verifNote')}</p>
    </section>
  )
}
