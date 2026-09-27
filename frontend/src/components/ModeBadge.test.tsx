import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ModeBadge } from './ModeBadge'

describe('ModeBadge', () => {
  it('labels demo mode as an offline demonstration of archived data', () => {
    render(<ModeBadge mode="demo" />)
    expect(screen.getByRole('status')).toHaveTextContent(/offline demonstration/i)
  })

  it('labels real mode as real data', () => {
    render(<ModeBadge mode="real" />)
    expect(screen.getByRole('status')).toHaveTextContent(/real data/i)
  })
})
