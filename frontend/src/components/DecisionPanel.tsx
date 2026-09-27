import { useState } from 'react'
import type { Decision, Role } from '../types'
import { titleCase } from '../lib/format'

interface DecisionPanelProps {
  role: Role
  version: number
  status: string
  existingDecision: string | null
  busy: boolean
  error: string
  onSubmit: (decision: Decision, note: string) => Promise<boolean>
}

const options: { value: Decision; label: string; copy: string }[] = [
  { value: 'confirmed_fraud', label: 'Confirm fraud', copy: 'Resolve this synthetic case as confirmed fraud.' },
  { value: 'legitimate', label: 'Mark legitimate', copy: 'Resolve this synthetic case as legitimate.' },
  { value: 'escalate', label: 'Escalate', copy: 'Route for a manual second-look review.' },
]

export function DecisionPanel({ role, version, status, existingDecision, busy, error, onSubmit }: DecisionPanelProps) {
  const [choice, setChoice] = useState<Decision | null>(null)
  const [note, setNote] = useState('')
  const canWrite = role === 'analyst' || role === 'admin'

  if (!canWrite) return <div className="permission-note">Viewer access is read-only. Analyst or admin credentials are required to write a review decision.</div>

  return (
    <div className="decision-panel">
      <div className="decision-state"><span>Current state</span><strong>{titleCase(status)}{existingDecision ? ` · ${titleCase(existingDecision)}` : ''}</strong><small>optimistic-lock version {version}</small></div>
      <div className="decision-options">
        {options.map((option) => <button key={option.value} type="button" className={choice === option.value ? 'selected' : ''} onClick={() => setChoice(option.value)}><strong>{option.label}</strong><span>{option.copy}</span></button>)}
      </div>
      <label className="note-field">Analyst note <span>{note.length}/2000</span><textarea value={note} maxLength={2000} onChange={(event) => setNote(event.target.value)} placeholder="Document the evidence or reason for this review action…" /></label>
      {error && <div className="form-error" role="alert">{error}</div>}
      <button className="button button-primary decision-submit" disabled={!choice || busy} onClick={async () => { if (!choice) return; const done = await onSubmit(choice, note); if (done) { setChoice(null); setNote('') } }}>{busy ? 'Writing decision…' : 'Save review decision'}</button>
      <p className="decision-footnote">Actions are append-only in review history. Concurrent stale writes are rejected by the API.</p>
    </div>
  )
}
