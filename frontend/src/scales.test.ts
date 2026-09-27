import { describe, expect, it } from 'vitest'
import { classOf, PROB_SCALE, RAIN_SCALE } from './scales'

describe('map classes', () => {
  it('uses IMD lower bounds inclusively', () => {
    expect(classOf(2.4, RAIN_SCALE)).toBe(0)
    expect(classOf(2.5, RAIN_SCALE)).toBe(1)
    expect(classOf(64.4, RAIN_SCALE)).toBe(2)
    expect(classOf(64.5, RAIN_SCALE)).toBe(3)
    expect(classOf(115.6, RAIN_SCALE)).toBe(4)
    expect(classOf(300, RAIN_SCALE)).toBe(5)
  })

  it('bins rare-event probabilities', () => {
    expect(classOf(0.014, PROB_SCALE)).toBe(0)   // near the heavy-rain base rate: no fill
    expect(classOf(0.5, PROB_SCALE)).toBe(4)
    expect(PROB_SCALE.format(0.51)).toBe('51%')
  })

  it('has one label per class including the unfilled class', () => {
    expect(RAIN_SCALE.labels).toHaveLength(RAIN_SCALE.edges.length + 1)
    expect(PROB_SCALE.labels).toHaveLength(PROB_SCALE.edges.length + 1)
  })
})
