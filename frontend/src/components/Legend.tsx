import { useT } from '../i18n'
import { classColor, type Scale } from '../scales'

export function Legend({ scale, title }: { scale: Scale; title: string }) {
  const t = useT()
  const label = (cls: number) => (scale.kind === 'rain' ? t(`rain${cls}` as 'rain0') : scale.labels[cls])
  return (
    <figure className="legend" aria-label={`${t('legend')}: ${title}`}>
      <figcaption>{title}</figcaption>
      <ul>
        {scale.labels.map((key, cls) => (
          <li key={key}>
            <span className="swatch" style={{ background: classColor(cls, scale) ?? 'var(--map-land)' }} aria-hidden="true" />
            {label(cls)}
          </li>
        ))}
      </ul>
    </figure>
  )
}
