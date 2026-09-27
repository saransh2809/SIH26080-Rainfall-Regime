import { useEffect, useRef, useState } from 'react'

export type Loadable<T> =
  | { state: 'loading' }
  | { state: 'ready'; data: T }
  | { state: 'error'; message: string; status?: number }

/** Run an API call once on mount and expose loading / ready / error explicitly. */
export function useApi<T>(call: () => Promise<T>): Loadable<T> {
  const [value, setValue] = useState<Loadable<T>>({ state: 'loading' })
  const callRef = useRef(call)
  useEffect(() => {
    let active = true
    callRef.current()
      .then((data) => active && setValue({ state: 'ready', data }))
      .catch((err: { message?: string; status?: number }) =>
        active && setValue({ state: 'error', message: err.message ?? 'request failed', status: err.status }),
      )
    return () => {
      active = false
    }
  }, [])
  return value
}
