import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ForecastSummary, LeadSummary } from '../api'
import { RegimeCard } from './RegimeCard'

const lead = {
  valid_date: '2025-08-02', predicted_regime: 'BREAK', regime_confidence: 0.47,
  regime_probabilities: { BREAK: 0.47, NORMAL: 0.25, MONSOON_DEPRESSION: 0.18, ACTIVE: 0.1 },
} as unknown as LeadSummary

const summary: ForecastSummary = {
  init_date: '2025-08-01', data_kind: 'real', mode: 'demo', system: 's', correction_model: 'c',
  heavy_rain_model: 'h', classifier: 'LightGBM on forecast fields only', leads: {},
}

describe('RegimeCard', () => {
  it('quotes the validation score of the classifier that made this forecast', () => {
    render(<RegimeCard lead={lead} summary={{ ...summary, source: 'live', classifier_macro_f1: 0.4812, rule_macro_f1: 0.54 }} />)
    expect(screen.getByText(/forecast fields only/)).toHaveTextContent('macro F1 0.48; a simple rule scores 0.54')
  })

  it('says "not evaluated" instead of inventing a score', () => {
    render(<RegimeCard lead={lead} summary={summary} />)
    expect(screen.getByText(/rainfall observed before/)).toHaveTextContent('not evaluated')
  })
})
