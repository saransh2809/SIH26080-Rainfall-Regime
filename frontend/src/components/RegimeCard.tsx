import type { ForecastSummary, LeadSummary } from '../api'
import { hasKey, useT } from '../i18n'

/** Predicted synoptic regime with its probability; bars are one hue — identity is the text label. */
export function RegimeCard({ lead, summary }: { lead: LeadSummary; summary: ForecastSummary }) {
  const t = useT()
  const name = (regime: string) => (hasKey(`regime_${regime}`) ? t(`regime_${regime}` as 'regime_NORMAL') : regime)
  const entries = Object.entries(lead.regime_probabilities).sort((a, b) => b[1] - a[1])
  const f1 = summary.classifier_macro_f1
  const rule = summary.rule_macro_f1
  return (
    <section className="panel" aria-labelledby="regime-title">
      <h2 id="regime-title">{t('regimeTitle', { date: lead.valid_date })}</h2>
      <p className="regime-headline">
        {name(lead.predicted_regime)}
        <span className="num"> {(lead.regime_confidence * 100).toFixed(0)}%</span>
      </p>
      <ul className="bars" aria-label={t('regimeBars')}>
        {entries.map(([regime, p]) => (
          <li key={regime}>
            <span className="bar-label">{name(regime)}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${p * 100}%` }} /></span>
            <span className="num bar-value">{(p * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
      <p className="muted">
        {t('regimeNote', {
          model: summary.source === 'live' ? t('classifierLive') : t('classifierArchive'),
          f1: f1 == null ? t('notEvaluated') : f1.toFixed(2),
          rule: rule == null ? '' : t('ruleScore', { v: rule.toFixed(2) }),
        })}
      </p>
    </section>
  )
}
