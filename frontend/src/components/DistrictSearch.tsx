import { useId, useMemo, useState } from 'react'
import type { District } from '../api'

/** Find a district by name; options carry the state so duplicate names (e.g. two "Raigarh") stay distinct. */
export function DistrictSearch({ districts, onSelect }: { districts: District[]; onSelect: (id: number) => void }) {
  const listId = useId()
  const [text, setText] = useState('')
  const byLabel = useMemo(
    () => new Map(districts.map((d) => [`${d.district_name}, ${d.state_name}`, d.district_id])),
    [districts],
  )
  const labels = useMemo(() => [...byLabel.keys()].sort((a, b) => a.localeCompare(b)), [byLabel])

  const choose = (value: string) => {
    setText(value)
    const id = byLabel.get(value)
    if (id !== undefined) onSelect(id)
  }

  return (
    <label className="control">
      <span>Find district</span>
      <input type="search" list={listId} value={text} placeholder="e.g. Mumbai"
        onChange={(e) => choose(e.target.value)} aria-describedby={`${listId}-hint`} />
      <span id={`${listId}-hint`} className="visually-hidden">Type a district name and choose a suggestion</span>
      <datalist id={listId}>
        {labels.map((label) => <option key={label} value={label} />)}
      </datalist>
    </label>
  )
}
