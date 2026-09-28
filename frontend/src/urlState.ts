import { LAYERS, type LayerKey } from './scales'

/** Dashboard view encoded in the URL so a view can be shared or bookmarked. */
export interface ViewState {
  init: string | null
  lead: number
  layer: LayerKey
  district: number | null
}

const DEFAULTS: ViewState = { init: null, lead: 1, layer: 'corrected_mm', district: null }
const LEADS = [1, 2, 3]

/** Parse a query string; anything malformed falls back to the default rather than being trusted. */
export function readViewState(search: string): ViewState {
  const q = new URLSearchParams(search)
  const init = q.get('init')
  const lead = Number(q.get('lead'))
  const layer = q.get('layer')
  const district = Number(q.get('district'))
  return {
    init: init && /^\d{4}-\d{2}-\d{2}$/.test(init) ? init : DEFAULTS.init,
    lead: LEADS.includes(lead) ? lead : DEFAULTS.lead,
    layer: LAYERS.some((l) => l.key === layer) ? (layer as LayerKey) : DEFAULTS.layer,
    district: Number.isInteger(district) && district > 0 ? district : DEFAULTS.district,
  }
}

export function writeViewState(state: ViewState): string {
  const q = new URLSearchParams()
  if (state.init) q.set('init', state.init)
  q.set('lead', String(state.lead))
  q.set('layer', state.layer)
  if (state.district != null) q.set('district', String(state.district))
  return `?${q.toString()}`
}
