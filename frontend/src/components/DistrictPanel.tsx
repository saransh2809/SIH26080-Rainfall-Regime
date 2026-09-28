import type { District, DistrictForecast } from '../api'
import { useT } from '../i18n'

/** Selected district: every model's value side by side, observation clearly separated. */
export function DistrictPanel({ district, forecast }: { district: District | undefined; forecast?: DistrictForecast }) {
  const t = useT()
  const mm = (v: number | null | undefined) => (v == null ? t('na') : `${v.toFixed(1)} mm`)
  const pct = (v: number | null | undefined) => (v == null ? t('na') : `${(v * 100).toFixed(0)}%`)
  return (
    <section className="panel" aria-labelledby="district-title">
      <h2 id="district-title">{t('districtTitle')}</h2>
      {!district ? (
        <p className="muted">{t('selectDistrict')}</p>
      ) : (
        <>
          <p className="district-name">{district.district_name}, <span className="muted">{district.state_name}</span></p>
          {district.coverage_fraction === 0 ? (
            <p className="notice notice-warn">{t('noCoverage')}</p>
          ) : district.low_coverage ? (
            <p className="notice notice-warn">{t('lowCoverage', { pct: (district.coverage_fraction * 100).toFixed(0) })}</p>
          ) : null}
          {forecast && district.coverage_fraction > 0 && (
            <table className="scores">
              <caption className="muted" style={{ textAlign: 'left', captionSide: 'top', paddingBottom: 6 }}>
                {t('areaMean', { date: forecast.valid_date, n: forecast.lead_day })}
              </caption>
              <tbody>
                <tr><th scope="row">{t('rowRaw')}</th><td className="num">{mm(forecast.raw_mm)}</td></tr>
                <tr><th scope="row">{t('rowCorrected')}</th><td className="num">{mm(forecast.corrected_mm)}</td></tr>
                {forecast.unet_mm !== undefined && (
                  <tr><th scope="row">{t('rowUnet')}</th><td className="num">{mm(forecast.unet_mm)}</td></tr>
                )}
                <tr><th scope="row">{t('rowQm')}</th><td className="num">{mm(forecast.qm_mm)}</td></tr>
                <tr><th scope="row">{t('rowHeavy')}</th><td className="num">{pct(forecast.p_heavy_max)}</td></tr>
                <tr><th scope="row">{t('rowVeryHeavy')}</th><td className="num">{pct(forecast.p_very_heavy_max)}</td></tr>
                <tr className="observed-row"><th scope="row">{t('rowObserved')}</th><td className="num">{mm(forecast.observed_mm)}</td></tr>
              </tbody>
            </table>
          )}
        </>
      )}
    </section>
  )
}
