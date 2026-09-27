import type { Metadata } from '../api'

/** Always-visible statement of what kind of data is on screen. */
export function ModeBadge({ mode }: { mode: Metadata['mode'] }) {
  return mode === 'real' ? (
    <span className="badge badge-real" role="status">Real data</span>
  ) : (
    <span className="badge badge-demo" role="status" title="Archived real data replayed offline; not a live forecast">
      Offline demonstration · archived real data
    </span>
  )
}
