import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import './App.css'
import { Bulletin } from './components/Bulletin'
import { DistrictPanel } from './components/DistrictPanel'
import { DistrictSearch } from './components/DistrictSearch'
import { ExplanationPanel } from './components/ExplanationPanel'
import { IndiaMap } from './components/IndiaMap'
import { Legend } from './components/Legend'
import { ModeBadge } from './components/ModeBadge'
import { ProductSource } from './components/ProductSource'
import { RegimeCard } from './components/RegimeCard'
import { VerificationPanel } from './components/VerificationPanel'
import { hasKey, LangContext, translate, type Lang, type StringKey } from './i18n'
import { LAYERS, type LayerKey } from './scales'
import { useApi } from './useApi'
import { readViewState, writeViewState } from './urlState'

const COMPARISON_MODELS = ['A_raw_nwp', 'B1_quantile_mapping', 'B2_global_lgbm', 'C1_regime_features', 'C2_regime_split']

export default function App() {
  const metadata = useApi(api.metadata)
  const states = useApi(() => api.geo('states'))
  const districtShapes = useApi(() => api.geo('districts'))
  const districts = useApi(api.districts)
  const products = useApi(api.products)
  const phase6 = useApi(() => api.verification('phase6'))
  const events = useApi(api.events)

  const [initial] = useState(() => readViewState(window.location.search))
  const [chosenInit, setInit] = useState<string | null>(initial.init)
  const [lead, setLead] = useState(initial.lead)
  const [layer, setLayer] = useState<LayerKey>(initial.layer)
  const [selectedId, setSelectedId] = useState<number | null>(initial.district)
  const [lang, setLang] = useState<Lang>(initial.lang)
  const t = (key: StringKey, vars?: Record<string, string | number>) => translate(lang, key, vars)

  const available = products.state === 'ready' ? products.data : []
  const init = chosenInit && available.includes(chosenInit) ? chosenInit : (available[0] ?? null)

  // Keep the URL in step with the view so it can be shared or bookmarked.
  const productsReady = products.state === 'ready'
  useEffect(() => {
    if (!productsReady) return  // keep the requested date in the URL until the product list has loaded
    window.history.replaceState(null, '', writeViewState({ init, lead, layer, district: selectedId, lang }))
  }, [productsReady, init, lead, layer, selectedId, lang])

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const summary = useApi(() => (init ? api.forecast(init) : Promise.reject(new Error('no product'))), init ?? '')
  const districtForecast = useApi(
    () => (init ? api.forecastDistricts(init, lead) : Promise.reject(new Error('no product'))), `${init}-${lead}`)

  const layerSpec = LAYERS.find((l) => l.key === layer)!
  const layerLabel = (key: LayerKey) => t(`layer_${key}`)
  const layerNote = (key: LayerKey) => (hasKey(`note_${key}`) ? t(`note_${key}` as StringKey) : undefined)
  const choropleth = useMemo(() => {
    if (districtForecast.state !== 'ready') return undefined
    return {
      values: new Map(districtForecast.data.map((d) => [d.district_id, d[layer] ?? null])),
      scale: layerSpec.scale,
      label: translate(lang, `layer_${layer}`),
    }
  }, [districtForecast, layer, layerSpec, lang])

  const selected = districts.state === 'ready' ? districts.data.find((d) => d.district_id === selectedId) : undefined
  const selectedForecast = districtForecast.state === 'ready'
    ? districtForecast.data.find((d) => d.district_id === selectedId) : undefined
  const leadSummary = summary.state === 'ready' ? summary.data.leads[String(lead)] : undefined
  const note = layerNote(layer)
  const activeCase = events.state === 'ready' ? events.data.find((e) => e.init === init) : undefined
  const caseDistrict = activeCase && districts.state === 'ready'
    ? districts.data.find((d) => d.district_id === activeCase.district) : undefined
  const openCase = (id: string) => {
    const e = events.state === 'ready' ? events.data.find((x) => x.id === id) : undefined
    if (!e) return
    setInit(e.init)
    setLead(1)
    setLayer('p_heavy_max')
    setSelectedId(e.district)
  }

  return (
    <LangContext.Provider value={lang}>
      <div className="app">
        <header className="header">
          <div>
            <h1>{t('title')}</h1>
            <div className="subtitle">{t('subtitle')}</div>
          </div>
          <div className="header-actions">
            <button type="button" className="lang-toggle" onClick={() => setLang(lang === 'en' ? 'hi' : 'en')}
              aria-label={t('langToggleLabel')} lang={lang === 'en' ? 'hi' : 'en'}>
              {t('langToggle')}
            </button>
            {metadata.state === 'ready' && <ModeBadge mode={metadata.data.mode} />}
            {metadata.state === 'error' && <span className="badge badge-demo" role="alert">{t('backendDown')}</span>}
          </div>
        </header>

        <nav className="controls" aria-label={t('controlsLabel')}>
          {events.state === 'ready' && events.data.length > 0 && (
            <label className="control">
              <span>{t('caseStudy')}</span>
              <select value={activeCase?.id ?? ''} onChange={(e) => openCase(e.target.value)}>
                <option value="" disabled>{t('chooseCase')}</option>
                {events.data.map((e) => (
                  <option key={e.id} value={e.id}>{e.label} · {t(`period_${e.period}`)}</option>
                ))}
              </select>
            </label>
          )}
          <label className="control">
            <span>{t('issued')}</span>
            <select value={init ?? ''} onChange={(e) => setInit(e.target.value)} disabled={products.state !== 'ready'}>
              {products.state === 'ready' && products.data.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <div className="control">
            <span>{t('lead')}</span>
            <div className="segmented" role="group" aria-label={t('leadGroup')}>
              {[1, 2, 3].map((l) => (
                <button key={l} type="button" aria-pressed={l === lead} onClick={() => setLead(l)}>{t('day', { n: l })}</button>
              ))}
            </div>
          </div>
          {districts.state === 'ready' && <DistrictSearch districts={districts.data} onSelect={setSelectedId} />}
          <div className="control">
            <span>{t('mapLayer')}</span>
            <div className="segmented" role="group" aria-label={t('mapLayer')}>
              {LAYERS.map((l) => (
                <button key={l.key} type="button" aria-pressed={l.key === layer} onClick={() => setLayer(l.key)}
                  title={layerNote(l.key)}>
                  {layerLabel(l.key)}
                </button>
              ))}
            </div>
          </div>
          {init && (
            <div className="control">
              <span>{t('export')}</span>
              <div className="export-actions">
                <a className="button" href={`/api/forecast/${init}/districts.csv?lead=${lead}`} download>{t('downloadCsv')}</a>
                <button type="button" className="button" onClick={() => window.print()}
                  disabled={districtForecast.state !== 'ready'}>{t('printPdf')}</button>
              </div>
            </div>
          )}
        </nav>

        <main className="main">
          <section className="panel map-panel" aria-label={t('mapLabel')}>
            {states.state === 'ready' && districtShapes.state === 'ready' ? (
              <IndiaMap states={states.data} districts={districtShapes.data} selectedId={selectedId}
                onSelect={setSelectedId} choropleth={choropleth} />
            ) : (
              <p className="muted" style={{ padding: 16 }}>
                {states.state === 'error' || districtShapes.state === 'error' ? t('mapUnavailable') : t('loadingMap')}
              </p>
            )}
            <div className="map-footer">
              <Legend scale={layerSpec.scale} title={`${layerLabel(layer)}${note ? ` — ${note}` : ''}`} />
              <p className="map-caption">
                {leadSummary ? t('validFor', { date: leadSummary.valid_date }) : ''}
                {t('mapCaption')}
              </p>
            </div>
          </section>

          <div className="side">
            {products.state === 'ready' && products.data.length === 0 && <p className="notice">{t('noProducts')}</p>}
            {activeCase && lead === 1 && (
              <p className="notice">
                <strong>{activeCase.label}.</strong>{' '}
                {t('caseNote', { mm: activeCase.observed_max_mm, district: caseDistrict?.district_name ?? String(activeCase.district) })}
              </p>
            )}
            {summary.state === 'ready' && <ProductSource summary={summary.data} />}
            {summary.state === 'ready' && leadSummary && <RegimeCard lead={leadSummary} summary={summary.data} />}
            <DistrictPanel district={selected} forecast={selectedForecast} />
            {summary.state === 'ready' && leadSummary && <ExplanationPanel summary={summary.data} lead={leadSummary} />}
            {phase6.state === 'ready' && (
              <VerificationPanel report={phase6.data} models={COMPARISON_MODELS} title={t('verifTitle')} />
            )}
          </div>
        </main>

        {init && summary.state === 'ready' && leadSummary && districtForecast.state === 'ready' && (
          <Bulletin init={init} summary={summary.data} lead={leadSummary} districts={districtForecast.data} />
        )}
      </div>
    </LangContext.Provider>
  )
}
