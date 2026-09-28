import type { ForecastSummary, LeadSummary } from '../api'
import { hasKey, translate, useT, type StringKey } from '../i18n'

/** Explanation assembled only from the product: model names, validation scores and SHAP contributions. */
export function ExplanationPanel({ summary, lead }: { summary: ForecastSummary; lead: LeadSummary }) {
  const t = useT()
  const rmse = lead.validation_rmse_mm
  const feature = (name: string) => (hasKey(`f_${name}`) ? t(`f_${name}` as 'f_lat') : name)
  // The backend names the model that actually ran; a translation is used only for that exact text.
  const describe = (backend: string, key: StringKey) => (translate('en', key) === backend ? t(key) : backend)
  return (
    <section className="panel" aria-labelledby="explain-title">
      <h2 id="explain-title">{t('explainTitle')}</h2>
      <dl className="kv">
        <dt>{t('correction')}</dt><dd>{describe(summary.correction_model, 'correctionModel')}</dd>
        <dt>{t('heavyRain')}</dt><dd>{describe(summary.heavy_rain_model, 'heavyModel')}</dd>
        {rmse && (
          <>
            <dt>{t('validationRmse')}</dt>
            <dd className="num">{t('rmseLine', { raw: rmse.raw.toFixed(2), corrected: rmse.corrected.toFixed(2) })}</dd>
          </>
        )}
      </dl>
      <p className="muted" style={{ margin: '12px 0 6px' }}>{t('contributions')}</p>
      <ul className="bars" aria-label={t('contribBars')}>
        {lead.top_correction_features.map((f) => (
          <li key={f.feature}>
            <span className="bar-label">{feature(f.feature)}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${f.share * 100}%` }} /></span>
            <span className="num bar-value">{(f.share * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
