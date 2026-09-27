import { useMemo, useState } from 'react'
import { api } from './api'
import './App.css'
import { DistrictPanel } from './components/DistrictPanel'
import { IndiaMap } from './components/IndiaMap'
import { ModeBadge } from './components/ModeBadge'
import { VerificationPanel } from './components/VerificationPanel'
import { useApi } from './useApi'

export default function App() {
  const metadata = useApi(api.metadata)
  const states = useApi(() => api.geo('states'))
  const districtShapes = useApi(() => api.geo('districts'))
  const districts = useApi(api.districts)
  const phase4 = useApi(() => api.verification('phase4'))
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const selected = useMemo(
    () => (districts.state === 'ready' ? districts.data.find((d) => d.district_id === selectedId) : undefined),
    [districts, selectedId],
  )

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

      <main className="main">
        <section className="panel map-panel" aria-label="Map">
          {states.state === 'ready' && districtShapes.state === 'ready' ? (
            <IndiaMap states={states.data} districts={districtShapes.data} selectedId={selectedId} onSelect={setSelectedId} />
          ) : (
            <p className="muted" style={{ padding: 16 }}>
              {states.state === 'error' || districtShapes.state === 'error' ? 'Map boundaries unavailable.' : 'Loading map…'}
            </p>
          )}
          <div className="map-caption">
            Boundaries: Survey of India index maps via DataMeet (Census 2011 districts). Rainfall layers appear once forecast products are generated.
          </div>
        </section>

        <div className="side">
          <section className="panel" aria-labelledby="forecast-title">
            <h2 id="forecast-title">Forecast</h2>
            <p className="notice">
              Forecast products are not generated yet. They will appear here after the regime classifier, regime-aware
              correction and heavy-rain probability models are validated.
            </p>
          </section>
          <DistrictPanel district={selected} />
          {phase4.state === 'ready' && (
            <VerificationPanel report={phase4.data} title="Baseline verification (Phase 4)" />
          )}
          {phase4.state === 'error' && <p className="notice">Verification report not available: {phase4.message}</p>}
        </div>
      </main>
    </div>
  )
}
