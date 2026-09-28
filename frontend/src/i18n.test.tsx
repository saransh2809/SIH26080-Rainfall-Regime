import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DistrictPanel } from './components/DistrictPanel'
import { LangContext, STRING_KEYS, translate } from './i18n'

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()

describe('translations', () => {
  it('every string exists in both languages with the same placeholders', () => {
    for (const key of STRING_KEYS) {
      const en = translate('en', key)
      const hi = translate('hi', key)
      expect(en.length, key).toBeGreaterThan(0)
      expect(hi.length, key).toBeGreaterThan(0)
      expect(placeholders(hi), key).toEqual(placeholders(en))
    }
  })

  it('fills placeholders and leaves unknown ones visible', () => {
    expect(translate('en', 'day', { n: 2 })).toBe('Day 2')
    expect(translate('hi', 'day', { n: 2 })).toBe('दिन 2')
    expect(translate('en', 'day')).toBe('Day {n}')
  })

  it('renders a component in Hindi from the context', () => {
    render(
      <LangContext.Provider value="hi">
        <DistrictPanel district={undefined} />
      </LangContext.Provider>,
    )
    expect(screen.getByRole('heading')).toHaveTextContent('ज़िला पूर्वानुमान')
  })
})
