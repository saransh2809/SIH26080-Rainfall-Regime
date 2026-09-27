// Map classes. Rainfall bins follow IMD 24-hour categories; probability bins are for rare events
// (heavy-rain base rate ≈ 1.4%). Class 0 is drawn without fill.

export interface Scale {
  kind: 'rain' | 'prob'
  edges: number[]              // lower bounds of classes 1..5
  labels: string[]             // labels for classes 0..5
  format: (v: number) => string
}

export const RAIN_SCALE: Scale = {
  kind: 'rain',
  edges: [2.5, 15.6, 64.5, 115.6, 204.5],
  labels: ['< 2.5 mm (dry / very light)', '2.5–15.5 light', '15.6–64.4 moderate', '64.5–115.5 heavy',
    '115.6–204.4 very heavy', '≥ 204.5 extremely heavy'],
  format: (v) => `${v.toFixed(1)} mm`,
}

export const PROB_SCALE: Scale = {
  kind: 'prob',
  edges: [0.05, 0.1, 0.2, 0.4, 0.6],
  labels: ['< 5%', '5–10%', '10–20%', '20–40%', '40–60%', '≥ 60%'],
  format: (v) => `${(v * 100).toFixed(0)}%`,
}

export function classOf(value: number, scale: Scale): number {
  let c = 0
  scale.edges.forEach((edge, i) => {
    if (value >= edge) c = i + 1
  })
  return c
}

export function classColor(cls: number, scale: Scale): string | null {
  if (cls === 0) return null
  return getComputedStyle(document.documentElement).getPropertyValue(`--${scale.kind}-${cls}`).trim()
}

export type LayerKey = 'raw_mm' | 'corrected_mm' | 'qm_mm' | 'observed_mm' | 'p_heavy_max' | 'p_very_heavy_max'

export const LAYERS: { key: LayerKey; label: string; scale: Scale; note?: string }[] = [
  { key: 'raw_mm', label: 'Raw NWP', scale: RAIN_SCALE },
  { key: 'corrected_mm', label: 'Regime-aware', scale: RAIN_SCALE },
  { key: 'qm_mm', label: 'Quantile-mapped', scale: RAIN_SCALE, note: 'preserves heavy-rain frequency' },
  { key: 'observed_mm', label: 'Observed (IMD)', scale: RAIN_SCALE, note: 'observation, for verification only' },
  { key: 'p_heavy_max', label: 'P(≥64.5 mm)', scale: PROB_SCALE },
  { key: 'p_very_heavy_max', label: 'P(≥115.6 mm)', scale: PROB_SCALE },
]
