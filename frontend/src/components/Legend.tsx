import { classColor, type Scale } from '../scales'

export function Legend({ scale, title }: { scale: Scale; title: string }) {
  return (
    <figure className="legend" aria-label={`Legend: ${title}`}>
      <figcaption>{title}</figcaption>
      <ul>
        {scale.labels.map((label, cls) => (
          <li key={label}>
            <span className="swatch" style={{ background: classColor(cls, scale) ?? 'var(--map-land)' }} aria-hidden="true" />
            {label}
          </li>
        ))}
      </ul>
    </figure>
  )
}
