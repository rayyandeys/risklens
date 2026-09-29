import type { Identity, RunSummary } from '../types'
import { compactHash, when } from '../lib/format'
import { DatabaseIcon, LogoutIcon, PulseIcon, QueueIcon, ShieldIcon } from './Icons'

export type ConsoleView = 'queue' | 'monitoring'

interface SidebarProps {
  identity: Identity
  runs: RunSummary[]
  selectedRunId: string
  view: ConsoleView
  onViewChange: (view: ConsoleView) => void
  onRunChange: (runId: string) => void
  onLogout: () => void
}

export function Sidebar({ identity, runs, selectedRunId, view, onViewChange, onRunChange, onLogout }: SidebarProps) {
  const current = runs.find((run) => run.run_id === selectedRunId)
  return (
    <aside className="sidebar">
      <div className="brand-lockup">
        <span className="brand-mark"><ShieldIcon /></span>
        <div><strong>RiskLens</strong><span>Analyst Console</span></div>
      </div>

      <nav className="nav-stack" aria-label="Primary">
        <button className={`nav-item ${view === 'queue' ? 'active' : ''}`} onClick={() => onViewChange('queue')}><QueueIcon /> Review queue <span>01</span></button>
        <button className={`nav-item ${view === 'monitoring' ? 'active' : ''}`} onClick={() => onViewChange('monitoring')}><PulseIcon /> Monitoring <span>02</span></button>
      </nav>

      <div className="sidebar-block">
        <div className="sidebar-label">Workflow run</div>
        <select value={selectedRunId} onChange={(event) => onRunChange(event.target.value)} disabled={runs.length < 2}>
          {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}
        </select>
        {current && (
          <div className="run-meta">
            <span><DatabaseIcon /> Month {current.scored_month} · {(current.capacity * 100).toFixed(0)}% capacity</span>
            <span title={current.model_sha256}>Model {compactHash(current.model_sha256)}</span>
            <span>Imported {when(current.imported_at)}</span>
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <div className="identity-card">
          <div className="avatar">{identity.analyst_id.slice(0, 2).toUpperCase()}</div>
          <div className="identity-copy"><strong>{identity.analyst_id}</strong><span>{identity.expires_at ? `${identity.role} · expires ${when(identity.expires_at)}` : 'Public demo · read-only'}</span></div>
          <button className="icon-button" onClick={onLogout} title="Disconnect"><LogoutIcon /></button>
        </div>
      </div>
    </aside>
  )
}
