import type { GeoJSON as LeafletGeoJSON, Layer, LeafletMouseEvent, PathOptions } from 'leaflet'
import { useEffect, useRef } from 'react'
import { GeoJSON, MapContainer, Pane } from 'react-leaflet'
import type { GeoJson } from '../api'
import { useT } from '../i18n'
import { classColor, classOf, type Scale } from '../scales'

const INDIA_BOUNDS: [[number, number], [number, number]] = [[6.0, 66.0], [38.8, 100.5]]

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export interface Choropleth {
  values: Map<number, number | null>
  scale: Scale
  label: string
}

interface Props {
  states: GeoJson
  districts: GeoJson
  selectedId: number | null
  onSelect: (districtId: number) => void
  choropleth?: Choropleth
}

/** Offline map: boundaries from our own data, no external tiles; districts coloured by real product values. */
export function IndiaMap({ states, districts, selectedId, onSelect, choropleth }: Props) {
  const districtLayer = useRef<LeafletGeoJSON | null>(null)
  const onSelectRef = useRef(onSelect)
  onSelectRef.current = onSelect
  const choroplethRef = useRef(choropleth)
  choroplethRef.current = choropleth
  const t = useT()
  const noDataRef = useRef(t('noData'))
  noDataRef.current = t('noData')

  const districtStyle = (feature?: GeoJSON.Feature): PathOptions => {
    const id = feature?.properties?.district_id as number
    const selected = id === selectedId
    const value = choropleth?.values.get(id)
    const fill = value == null || !choropleth ? null : classColor(classOf(value, choropleth.scale), choropleth.scale)
    return {
      color: selected ? cssVar('--map-selected') : cssVar('--map-district'),
      weight: selected ? 2.5 : 0.5,
      fillColor: fill ?? cssVar('--map-land'),
      fillOpacity: 1,
    }
  }

  // Restyle in place (no rebuild of 641 polygons) when selection or data changes.
  useEffect(() => {
    const layer = districtLayer.current
    if (!layer) return
    layer.setStyle(districtStyle)
    layer.eachLayer((l) => {
      const f = (l as unknown as { feature?: GeoJSON.Feature }).feature
      if (f?.properties?.district_id === selectedId) (l as unknown as { bringToFront: () => void }).bringToFront()
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, choropleth])

  const onEachDistrict = (feature: GeoJSON.Feature, layer: Layer) => {
    const p = feature.properties ?? {}
    layer.bindTooltip(() => {
      const c = choroplethRef.current
      const v = c?.values.get(p.district_id as number)
      const value = c ? (v == null ? ` — ${noDataRef.current}` : ` — ${c.label}: ${c.scale.format(v)}`) : ''
      return `${p.district_name}, ${p.state_name}${value}`
    }, { sticky: true })
    layer.on('click', (e: LeafletMouseEvent) => {
      e.originalEvent.stopPropagation()
      onSelectRef.current(p.district_id as number)
    })
  }

  return (
    <MapContainer bounds={INDIA_BOUNDS} maxBounds={INDIA_BOUNDS} minZoom={4} maxZoom={9} zoomSnap={0.25}
      attributionControl={false} aria-label="Map of India districts">
      <GeoJSON ref={districtLayer} data={districts} style={districtStyle} onEachFeature={onEachDistrict} />
      <Pane name="state-borders" style={{ zIndex: 450, pointerEvents: 'none' }}>
        <GeoJSON data={states} style={{ color: cssVar('--map-state'), weight: 1.2, fill: false }} interactive={false} />
      </Pane>
    </MapContainer>
  )
}
