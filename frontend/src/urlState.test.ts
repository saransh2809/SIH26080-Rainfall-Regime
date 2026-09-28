import { describe, expect, it } from 'vitest'
import { readViewState, writeViewState } from './urlState'

describe('URL view state', () => {
  it('round-trips a full view', () => {
    const view = { init: '2017-08-29', lead: 2, layer: 'p_heavy_max' as const, district: 42 }
    expect(readViewState(writeViewState(view))).toEqual(view)
  })

  it('falls back to defaults for malformed or unknown values', () => {
    expect(readViewState('?init=../../etc&lead=9&layer=secret&district=-3')).toEqual({
      init: null, lead: 1, layer: 'corrected_mm', district: null,
    })
  })

  it('treats an empty query as the default view', () => {
    expect(readViewState('')).toEqual({ init: null, lead: 1, layer: 'corrected_mm', district: null })
  })
})
