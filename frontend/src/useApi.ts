import { useEffect, useState } from 'react'

export type Loadable<T> =
  | { state: 'loading' }
  | { state: 'ready'; data: T }
  | { state: 'error'; message: string; status?: number }

const LOADING = { state: 'loading' } as const

/**
 * Run an API call on mount and whenever `key` changes. The result is stored with the key that
 * produced it, so a stale result is never shown for a new key (it reads as loading instead).
 */
export function useApi<T>(call: () => Promise<T>, key = ''): Loadable<T> {
  const [result, setResult] = useState<{ key: string; value: Loadable<T> } | null>(null)
  useEffect(() => {
    let active = true
    call()
      .then((data) => active && setResult({ key, value: { state: 'ready', data } }))
      .catch((err: { message?: string; status?: number }) => active && setResult({
        key, value: { state: 'error', message: err.message ?? 'request failed', status: err.status },
      }))
    return () => {
      active = false
    }
    // `key` identifies the request; `call` is recreated on every render by design.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return result && result.key === key ? result.value : LOADING
}
