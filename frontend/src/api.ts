// Typed client for the FastAPI backend (proxied at /api by Vite).

export interface Metadata {
  mode: 'demo' | 'real'
  data_kind: string
  system: string
  lead_days: number[]
  split: { train: [number, number]; validation: [number, number]; test: [number, number] }
  thresholds_mm: Record<string, number>
}

export interface District {
  district_id: number
  district_name: string
  state_name: string
  coverage_fraction: number
  low_coverage: boolean
}

export interface Contingency {
  pod: number | null
  far: number | null
  csi: number | null
  ets: number | null
  frequency_bias: number | null
  hits: number
  misses: number
}

export interface ModelScores {
  n: number
  rmse: number
  mae: number
  bias: number
  correlation: number
  categorical: Record<string, Contingency>
  fss: Record<string, Record<string, number | null>>
}

export interface VerificationReport {
  scope: string
  data_kind: string
  results: Record<string, Record<string, ModelScores>>
}

export type GeoJson = GeoJSON.FeatureCollection

export interface LeadSummary {
  valid_date: string
  predicted_regime: string
  regime_confidence: number
  regime_probabilities: Record<string, number>
  validation_rmse_mm: { raw: number; corrected: number } | null
  top_correction_features: { feature: string; share: number }[]
}

export interface ForecastSummary {
  init_date: string
  data_kind: string
  mode: string
  system: string
  correction_model: string
  heavy_rain_model: string
  classifier: string
  leads: Record<string, LeadSummary>
  source?: 'archive' | 'live'
  forecast?: string
  in_season?: boolean
  verification?: string
  classifier_macro_f1?: number | null
  rule_macro_f1?: number | null
}

export interface DistrictForecast {
  district_id: number
  district_name: string
  state_name: string
  coverage_fraction: number
  lead_day: number
  valid_date: string
  raw_mm: number | null
  corrected_mm: number | null
  unet_mm?: number | null
  qm_mm: number | null
  observed_mm: number | null
  p_heavy_max: number | null
  p_very_heavy_max: number | null
}

export interface CaseStudy {
  id: string
  init: string
  period: 'validation' | 'test' | 'operational_test'
  source: 'archive' | 'operational'
  district: number
  observed_max_mm: number
  label: string
}

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new ApiError(response.status, body.detail ?? response.statusText)
  }
  return response.json() as Promise<T>
}

export const api = {
  metadata: () => get<Metadata>('/metadata'),
  districts: () => get<District[]>('/districts'),
  geo: (layer: 'states' | 'districts') => get<GeoJson>(`/geo/${layer}`),
  verification: (phase: 'phase4' | 'phase5' | 'phase6' | 'phase7') => get<VerificationReport>(`/verification/${phase}`),
  products: () => get<string[]>('/products'),
  events: () => get<CaseStudy[]>('/events'),
  forecast: (init: string) => get<ForecastSummary>(`/forecast/${init}`),
  forecastDistricts: (init: string, lead: number) => get<DistrictForecast[]>(`/forecast/${init}/districts?lead=${lead}`),
}
