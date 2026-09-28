import type { DistrictForecast, ForecastSummary, LeadSummary } from '../api'
import { hasKey, useT } from '../i18n'

const TOP_N = 25

/** Printable one-page summary (hidden on screen): the districts most likely to get heavy rain. */
export function Bulletin({ init, summary, lead, districts }: {
  init: string
  summary: ForecastSummary
  lead: LeadSummary
  districts: DistrictForecast[]
}) {
  const t = useT()
  const live = summary.source === 'live'
  const leadDay = districts[0]?.lead_day ?? 1
  const top = districts
    .filter((d) => d.p_heavy_max != null)
    .sort((a, b) => (b.p_heavy_max ?? 0) - (a.p_heavy_max ?? 0))
    .slice(0, TOP_N)
  const hasObserved = top.some((d) => d.observed_mm != null)
  const mm = (v: number | null) => (v == null ? t('na') : v.toFixed(1))
  const pct = (v: number | null) => (v == null ? t('na') : `${(v * 100).toFixed(0)}%`)
  const regime = hasKey(`regime_${lead.predicted_regime}`)
    ? t(`regime_${lead.predicted_regime}` as 'regime_NORMAL') : lead.predicted_regime

  return (
    <article className="bulletin" aria-hidden="true">
      <header>
        <h1>{t('bulletinTitle')}</h1>
        <p>{t('bulletinIssued', { init, n: leadDay, date: lead.valid_date })}</p>
        <p>
          {live ? t('live') : t('archive')} · {live ? t('forecastLive') : t('forecastArchive')} ·{' '}
          {summary.verification?.startsWith('pending') ? t('verificationPending') : t('verificationAttached')}
        </p>
        <p>{t('regimeTitle', { date: lead.valid_date })}: <strong>{regime}</strong> ({(lead.regime_confidence * 100).toFixed(0)}%)</p>
        {summary.in_season === false && <p><strong>{t('outOfSeason')}</strong></p>}
      </header>
      <h2>{t('bulletinTop')}</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>{t('district')}</th><th>{t('state')}</th>
            <th>{t('rowHeavy')}</th><th>{t('rowVeryHeavy')}</th>
            <th>{t('rowRaw')} (mm)</th><th>{t('rowCorrected')} (mm)</th>
            {hasObserved && <th>{t('rowObserved')} (mm)</th>}
          </tr>
        </thead>
        <tbody>
          {top.map((d, i) => (
            <tr key={d.district_id}>
              <td>{i + 1}</td><td>{d.district_name}</td><td>{d.state_name}</td>
              <td>{pct(d.p_heavy_max)}</td><td>{pct(d.p_very_heavy_max)}</td>
              <td>{mm(d.raw_mm)}</td><td>{mm(d.corrected_mm)}</td>
              {hasObserved && <td>{mm(d.observed_mm)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="bulletin-footer">{t('bulletinFooter')}</p>
    </article>
  )
}
