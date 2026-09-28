import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ForecastSummary } from '../api'
import { ProductSource } from './ProductSource'

const base: ForecastSummary = {
  init_date: '2025-08-01', data_kind: 'real', mode: 'demo', system: 's', correction_model: 'c',
  heavy_rain_model: 'h', classifier: 'k', leads: {},
}

describe('ProductSource', () => {
  it('marks live products and shows pending verification', () => {
    render(<ProductSource summary={{ ...base, source: 'live', in_season: true, verification: 'pending: not published' }} />)
    expect(screen.getByRole('status')).toHaveTextContent(/live forecast/i)
    expect(screen.getByText(/pending/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('warns when the forecast is valid outside the monsoon season', () => {
    render(<ProductSource summary={{ ...base, source: 'live', in_season: false }} />)
    expect(screen.getByRole('alert')).toHaveTextContent(/outside June–September/)
  })

  it('treats products without a source as archived', () => {
    render(<ProductSource summary={base} />)
    expect(screen.getByRole('status')).toHaveTextContent(/archived forecast/i)
  })
})
