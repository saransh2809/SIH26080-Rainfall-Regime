import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { District } from '../api'
import { DistrictSearch } from './DistrictSearch'

// SYNTHETIC district list for UI testing only.
const districts: District[] = [
  { district_id: 1, district_name: 'Raigarh', state_name: 'Chhattisgarh', coverage_fraction: 1, low_coverage: false },
  { district_id: 2, district_name: 'Raigarh', state_name: 'Maharashtra', coverage_fraction: 1, low_coverage: false },
]

describe('DistrictSearch', () => {
  it('selects the district matching name and state, keeping duplicates distinct', () => {
    const onSelect = vi.fn()
    render(<DistrictSearch districts={districts} onSelect={onSelect} />)
    fireEvent.change(screen.getByRole('combobox', { name: /find district/i }), { target: { value: 'Raigarh, Maharashtra' } })
    expect(onSelect).toHaveBeenCalledWith(2)
  })

  it('does not select on partial text', () => {
    const onSelect = vi.fn()
    render(<DistrictSearch districts={districts} onSelect={onSelect} />)
    fireEvent.change(screen.getByRole('combobox', { name: /find district/i }), { target: { value: 'Raig' } })
    expect(onSelect).not.toHaveBeenCalled()
  })
})
