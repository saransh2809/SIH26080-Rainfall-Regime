import type { Metadata } from '../api'
import { useT } from '../i18n'

/** Always-visible statement of what kind of data is on screen. */
export function ModeBadge({ mode }: { mode: Metadata['mode'] }) {
  const t = useT()
  return mode === 'real' ? (
    <span className="badge badge-real" role="status">{t('modeReal')}</span>
  ) : (
    <span className="badge badge-demo" role="status" title={t('modeDemoTitle')}>{t('modeDemo')}</span>
  )
}
