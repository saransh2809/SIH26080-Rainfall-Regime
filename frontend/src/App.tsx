import { useMemo, useState } from 'react'
import { api } from './api'
import './App.css'
import { DistrictPanel } from './components/DistrictPanel'
import { ExplanationPanel } from './components/ExplanationPanel'
import { IndiaMap } from './components/IndiaMap'
import { Legend } from './components/Legend'
import { ModeBadge } from './components/ModeBadge'
import { RegimeCard } from './components/RegimeCard'
import { VerificationPanel } from './components/VerificationPanel'
import { LAYERS, type LayerKey } from './scales'
import { useApi } from './useApi'

const COMPARISON_MODELS = ['A_raw_nwp', 'B1_quantile_mapping', 'B2_global_lgbm', 'C1_regime_features', 'C2_regime_split']

export default function App() {
  const metadata = useApi(api.metadata)
  const states = useApi(() => api.geo('states'))
  const districtShapes = useApi(() => api.geo('districts'))
  const districts = useApi(api.districts)
  const products = useApi(api.products)
  const phase6 = useApi(() => api.verification('phase6'))

  const [chosenInit, setInit] = useState<string | null>(null)
  const [lead, setLead] = useState(1)
  const [layer, setLayer] = useState<LayerKey>('corrected_mm')
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const init = chosenInit ?? (products.state === 'ready' && products.data.length ? products.data[0] : null)

  const summary = useApi(() => (init ? api.forecast(init) : Promise.reject(new Error('no product'))), init ?? '')
  const districtForecast = useApi(
    () => (init ? api.forecastDistricts(init, lead) : Promise.reject(new Error('no product'))), `${init}-${lead}`)

  const layerSpec = LAYERS.find((l) => l.key === layer)!
  const choropleth = useMemo(() => {
    if (districtForecast.state !== 'ready') return undefined
    return {
      values: new Map(districtForecast.data.map((d) => [d.district_id, d[layer]])),
      scale: layerSpec.scale,
      label: layerSpec.label,
    }
  }, [districtForecast, layer, layerSpec])

  const selected = districts.state === 'ready' ? districts.data.find((d) => d.district_id === selectedId) : undefined
  const selectedForecast = districtForecast.state === 'ready'
    ? districtForecast.data.find((d) => d.district_id === selectedId) : undefined
  const leadSummary = summary.state === 'ready' ? summary.data.leads[String(lead)] : undefined

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Regime-Aware Rainfall Forecast Post-Processing</h1>
          <div className="subtitle">SIH 26080 · NCMRWF / MoES · corrects NWP rainfall forecasts; does not forecast weather from scratch</div>
        </div>
        {metadata.state === 'ready' && <ModeBadge mode={metadata.data.mode} />}
        {metadata.state === 'error' && <span className="badge badge-demo" role="alert">Backend unavailable</span>}
      </header>

      <nav className="controls" aria-label="Forecast controls">
        <label className="control">
          <span>Forecast issued (00 UTC)</span>
          <select value={init ?? ''} onChange={(e) => setInit(e.target.value)} disabled={products.state !== 'ready'}>
            {products.state === 'ready' && products.data.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <div className="control">
          <span>Lead</span>
          <div className="segmented" role="group" aria-label="Lead day">
            {[1, 2, 3].map((l) => (
              <button key={l} type="button" aria-pressed={l === lead} onClick={() => setLead(l)}>Day {l}</button>
            ))}
          </div>
        </div>
        <div className="control">
          <span>Map layer</span>
          <div className="segmented" role="group" aria-label="Map layer">
            {LAYERS.map((l) => (
              <button key={l.key} type="button" aria-pressed={l.key === layer} onClick={() => setLayer(l.key)} title={l.note}>
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="main">
        <section className="panel map-panel" aria-label="Map">
          {states.state === 'ready' && districtShapes.state === 'ready' ? (
            <IndiaMap states={states.data} districts={districtShapes.data} selectedId={selectedId}
              onSelect={setSelectedId} choropleth={choropleth} />
          ) : (
            <p className="muted" style={{ padding: 16 }}>
              {states.state === 'error' || districtShapes.state === 'error' ? 'Map boundaries unavailable.' : 'Loading map…'}
            </p>
          )}
          <div className="map-footer">
            <Legend scale={layerSpec.scale} title={`${layerSpec.label}${layerSpec.note ? ` — ${layerSpec.note}` : ''}`} />
            <p className="map-caption">
              {leadSummary ? `Valid ${leadSummary.valid_date} (24 h ending 08:30 IST). ` : ''}
              District values: area-weighted mean of 0.25° cells (probabilities: highest cell).
              Boundaries: Survey of India index maps via DataMeet (Census 2011 districts).
            </p>
          </div>
        </section>

        <div className="side">
          {products.state === 'ready' && products.data.length === 0 && (
            <p className="notice">No forecast products have been generated yet.</p>
          )}
          {leadSummary && <RegimeCard lead={leadSummary} />}
          <DistrictPanel district={selected} forecast={selectedForecast} />
          {summary.state === 'ready' && leadSummary && <ExplanationPanel summary={summary.data} lead={leadSummary} />}
          {phase6.state === 'ready' && (
            <VerificationPanel report={phase6.data} models={COMPARISON_MODELS}
              title="Model comparison (validation 2016–2017)" />
          )}
        </div>
      </main>
    </div>
  )
}
