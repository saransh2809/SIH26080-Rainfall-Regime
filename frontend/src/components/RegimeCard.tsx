import type { LeadSummary } from '../api'

const NAMES: Record<string, string> = {
  NORMAL: 'Normal', ACTIVE: 'Active monsoon', BREAK: 'Break monsoon', MONSOON_DEPRESSION: 'Monsoon depression',
}

/** Predicted synoptic regime with its probability; bars are one hue — identity is the text label. */
export function RegimeCard({ lead }: { lead: LeadSummary }) {
  const entries = Object.entries(lead.regime_probabilities).sort((a, b) => b[1] - a[1])
  return (
    <section className="panel" aria-labelledby="regime-title">
      <h2 id="regime-title">Weather regime · valid {lead.valid_date}</h2>
      <p className="regime-headline">
        {NAMES[lead.predicted_regime] ?? lead.predicted_regime}
        <span className="num"> {(lead.regime_confidence * 100).toFixed(0)}%</span>
      </p>
      <ul className="bars" aria-label="Regime probabilities">
        {entries.map(([name, p]) => (
          <li key={name}>
            <span className="bar-label">{NAMES[name] ?? name}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${p * 100}%` }} /></span>
            <span className="num bar-value">{(p * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
      <p className="muted">
        Predicted from forecast fields by a LightGBM classifier (validation macro F1 0.52; a simple rule scores 0.54).
        Treat as guidance, not certainty.
      </p>
    </section>
  )
}
