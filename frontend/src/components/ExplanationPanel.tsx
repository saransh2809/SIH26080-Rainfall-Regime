import type { ForecastSummary, LeadSummary } from '../api'

const FEATURE_NAMES: Record<string, string> = {
  nwp_precip_mm: 'Forecast rain at the cell',
  nwp_log1p: 'Forecast rain (log)',
  nwp_mean_3x3: 'Forecast rain, 3×3 neighbourhood mean',
  nwp_max_3x3: 'Forecast rain, 3×3 neighbourhood max',
  nwp_mean_7x7: 'Forecast rain, 7×7 neighbourhood mean',
  nwp_max_7x7: 'Forecast rain, 7×7 neighbourhood max',
  obs_climatology_mm: 'Observed climatology (training years)',
  doy_sin: 'Season (day of year)',
  doy_cos: 'Season (day of year)',
  lat: 'Latitude',
  lon: 'Longitude',
  lead_day: 'Forecast lead time',
  elevation_m: 'Terrain elevation',
  coast_km: 'Distance to coast',
  p_NORMAL: 'Regime probability: normal',
  p_ACTIVE: 'Regime probability: active',
  p_BREAK: 'Regime probability: break',
  p_MONSOON_DEPRESSION: 'Regime probability: depression',
  local_INLAND: 'Local regime: inland',
  local_COASTAL: 'Local regime: coastal',
  local_OROGRAPHIC: 'Local regime: orographic',
}

/** Explanation assembled only from the product: model names, validation scores and SHAP contributions. */
export function ExplanationPanel({ summary, lead }: { summary: ForecastSummary; lead: LeadSummary }) {
  const rmse = lead.validation_rmse_mm
  return (
    <section className="panel" aria-labelledby="explain-title">
      <h2 id="explain-title">Why this correction</h2>
      <dl className="kv">
        <dt>Correction</dt><dd>{summary.correction_model}</dd>
        <dt>Heavy rain</dt><dd>{summary.heavy_rain_model}</dd>
        {rmse && (
          <>
            <dt>Validation RMSE</dt>
            <dd className="num">raw {rmse.raw.toFixed(2)} mm → corrected {rmse.corrected.toFixed(2)} mm (2016–2017, this lead)</dd>
          </>
        )}
      </dl>
      <p className="muted" style={{ margin: '12px 0 6px' }}>Largest contributions to the correction on this day (mean |SHAP|, share):</p>
      <ul className="bars" aria-label="Feature contributions">
        {lead.top_correction_features.map((f) => (
          <li key={f.feature}>
            <span className="bar-label">{FEATURE_NAMES[f.feature] ?? f.feature}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${f.share * 100}%` }} /></span>
            <span className="num bar-value">{(f.share * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
