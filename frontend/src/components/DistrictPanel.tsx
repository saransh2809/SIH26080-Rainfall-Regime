import type { District } from '../api'

export function DistrictPanel({ district }: { district: District | undefined }) {
  return (
    <section className="panel" aria-labelledby="district-title">
      <h2 id="district-title">District</h2>
      {district ? (
        <>
          <dl className="kv">
            <dt>District</dt><dd>{district.district_name}</dd>
            <dt>State</dt><dd>{district.state_name}</dd>
            <dt>Grid coverage</dt><dd className="num">{(district.coverage_fraction * 100).toFixed(0)}%</dd>
          </dl>
          {district.coverage_fraction === 0 ? (
            <p className="notice notice-warn">No IMD 0.25° land cells overlap this district; no values are shown.</p>
          ) : district.low_coverage ? (
            <p className="notice notice-warn">Under half of this district is covered by the 0.25° grid; treat values with caution.</p>
          ) : null}
        </>
      ) : (
        <p className="muted">Select a district on the map.</p>
      )}
    </section>
  )
}
