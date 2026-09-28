import { describe, expect, it } from 'vitest'
import { readViewState, writeViewState } from './urlState'

const DEFAULT_VIEW = { init: null, lead: 1, layer: 'corrected_mm', district: null, lang: 'en' }

describe('URL view state', () => {
  it('round-trips a full view', () => {
    const view = { init: '2017-08-29', lead: 2, layer: 'p_heavy_max' as const, district: 42, lang: 'hi' as const }
    expect(readViewState(writeViewState(view))).toEqual(view)
  })

  it('falls back to defaults for malformed or unknown values', () => {
    expect(readViewState('?init=../../etc&lead=9&layer=secret&district=-3&lang=xx')).toEqual(DEFAULT_VIEW)
  })

  it('treats an empty query as the default view', () => {
    expect(readViewState('')).toEqual(DEFAULT_VIEW)
  })

  it('leaves English out of the URL', () => {
    expect(writeViewState({ ...DEFAULT_VIEW, layer: 'corrected_mm', lang: 'en' })).not.toContain('lang')
  })
})
