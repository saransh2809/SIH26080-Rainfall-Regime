import type { District, DistrictForecast } from '../api'

function mm(v: number | null | undefined): string {
  return v == null ? 'n/a' : `${v.toFixed(1)} mm`
}

function pct(v: number | null | undefined): string {
  return v == null ? 'n/a' : `${(v * 100).toFixed(0)}%`
}

/** Selected district: every model's value side by side, observation clearly separated. */
export function DistrictPanel({ district, forecast }: { district: District | undefined; forecast?: DistrictForecast }) {
  return (
    <section className="panel" aria-labelledby="district-title">
      <h2 id="district-title">District forecast</h2>
      {!district ? (
        <p className="muted">Select a district on the map.</p>
      ) : (
        <>
          <p className="district-name">{district.district_name}, <span className="muted">{district.state_name}</span></p>
          {district.coverage_fraction === 0 ? (
            <p className="notice notice-warn">No IMD 0.25° land cells overlap this district; no values are shown.</p>
          ) : district.low_coverage ? (
            <p className="notice notice-warn">
              Only {(district.coverage_fraction * 100).toFixed(0)}% of this district is covered by the 0.25° grid; treat values with caution.
            </p>
          ) : null}
          {forecast && district.coverage_fraction > 0 && (
            <table className="scores">
              <caption className="muted" style={{ textAlign: 'left', captionSide: 'top', paddingBottom: 6 }}>
                Area-mean rainfall, valid {forecast.valid_date} (day {forecast.lead_day})
              </caption>
              <tbody>
                <tr><th scope="row">Raw NWP (GEFS)</th><td className="num">{mm(forecast.raw_mm)}</td></tr>
                <tr><th scope="row">Regime-aware correction</th><td className="num">{mm(forecast.corrected_mm)}</td></tr>
                <tr><th scope="row">Quantile-mapped</th><td className="num">{mm(forecast.qm_mm)}</td></tr>
                <tr><th scope="row">P(heavy ≥ 64.5 mm), highest cell</th><td className="num">{pct(forecast.p_heavy_max)}</td></tr>
                <tr><th scope="row">P(very heavy ≥ 115.6 mm), highest cell</th><td className="num">{pct(forecast.p_very_heavy_max)}</td></tr>
                <tr className="observed-row"><th scope="row">Observed (IMD) — verification</th><td className="num">{mm(forecast.observed_mm)}</td></tr>
              </tbody>
            </table>
          )}
        </>
      )}
    </section>
  )
}
