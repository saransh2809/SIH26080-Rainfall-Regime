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
}
