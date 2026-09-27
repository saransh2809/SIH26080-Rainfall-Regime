import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ModelScores, VerificationReport } from '../api'
import { VerificationPanel } from './VerificationPanel'

function scores(rmse: number, pod: number | null): ModelScores {
  return {
    n: 10, rmse, mae: 1, bias: 0.1, correlation: 0.5,
    categorical: { '64.5': { pod, far: 0.5, csi: 0.2, ets: 0.1, frequency_bias: 1, hits: 1, misses: 1 } },
    fss: { '64.5': { '9': 0.4 } },
  }
}

// SYNTHETIC report for UI testing only.
const report: VerificationReport = {
  scope: 'VALIDATION years 2016-2017', data_kind: 'real',
  results: {
    '1': { A_raw_nwp: scores(15, 0.2), B2_global_lgbm: scores(13, null) },
    '2': { A_raw_nwp: scores(16, 0.1), B2_global_lgbm: scores(20, 0.05) },
  },
}

describe('VerificationPanel', () => {
  it('shows undefined scores as n/a, never 0', () => {
    render(<VerificationPanel report={report} title="t" />)
    const row = screen.getByRole('row', { name: /POD/ })
    expect(within(row).getByText('n/a')).toBeInTheDocument()
  })

  it('marks the better value using the metric direction', () => {
    render(<VerificationPanel report={report} title="t" />)
    const rmse = screen.getByRole('row', { name: /RMSE/ })
    expect(within(rmse).getByText('13.000')).toHaveClass('best')
  })

  it('switches lead day', () => {
    render(<VerificationPanel report={report} title="t" />)
    fireEvent.click(screen.getByRole('button', { name: 'Day 2' }))
    expect(screen.getByRole('button', { name: 'Day 2' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(screen.getByRole('row', { name: /RMSE/ })).getByText('16.000')).toHaveClass('best')
  })
})
