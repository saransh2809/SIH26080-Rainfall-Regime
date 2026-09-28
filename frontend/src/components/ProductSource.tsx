import type { ForecastSummary } from '../api'
import { useT } from '../i18n'

/** Where this forecast came from and whether it can be checked against observations yet. */
export function ProductSource({ summary }: { summary: ForecastSummary }) {
  const t = useT()
  const live = summary.source === 'live'
  const pending = summary.verification?.startsWith('pending')
  return (
    <section className="panel" aria-label={t('sourceLabel')}>
      <div className="source-row">
        <span className={`badge ${live ? 'badge-live' : 'badge-archive'}`} role="status">
          {live ? t('live') : t('archive')}
        </span>
        <span className="muted">{live ? t('forecastLive') : t('forecastArchive')}</span>
      </div>
      {summary.verification && (
        <p className="muted source-note">{pending ? t('verificationPending') : t('verificationAttached')}</p>
      )}
      {summary.in_season === false && <p className="notice notice-warn" role="alert">{t('outOfSeason')}</p>}
    </section>
  )
}
