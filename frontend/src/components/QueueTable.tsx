import type { CaseListItem } from '../types'
import { pct, score, titleCase } from '../lib/format'
import { ArrowIcon } from './Icons'

interface QueueTableProps {
  cases: CaseListItem[]
  selectedCaseId: string | null
  loading: boolean
  offset: number
  limit: number
  onSelect: (caseId: string) => void
  onPrev: () => void
  onNext: () => void
}

function StatusBadge({ value }: { value: string }) {
  return <span className={`status-badge status-${value}`}>{titleCase(value)}</span>
}

export function QueueTable({ cases, selectedCaseId, loading, offset, limit, onSelect, onPrev, onNext }: QueueTableProps) {
  return (
    <div className="queue-table-wrap">
      <table className="queue-table">
        <thead><tr><th>Rank</th><th>Case</th><th>Model score</th><th>Percentile</th><th>Status</th><th aria-label="Open" /></tr></thead>
        <tbody>
          {loading && Array.from({ length: 6 }).map((_, index) => (
            <tr className="skeleton-row" key={index}><td colSpan={6}><span /></td></tr>
          ))}
          {!loading && cases.map((item) => (
            <tr key={item.case_id} className={selectedCaseId === item.case_id ? 'selected' : ''} onClick={() => onSelect(item.case_id)} tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onSelect(item.case_id) }}>
              <td className="rank-cell">#{item.risk_rank}</td>
              <td><strong>{item.case_id}</strong><small>row {item.source_row_id.toLocaleString()}</small></td>
              <td><span className="score-pill">{score(item.risk_score)}</span></td>
              <td>{pct(item.risk_percentile, 2)}</td>
              <td><StatusBadge value={item.review_status} /></td>
              <td><ArrowIcon /></td>
            </tr>
          ))}
          {!loading && cases.length === 0 && <tr><td colSpan={6}><div className="empty-table">No cases match these filters.</div></td></tr>}
        </tbody>
      </table>
      <div className="pagination">
        <span>{cases.length ? `${offset + 1}–${offset + cases.length}` : '0'} shown</span>
        <div><button className="button button-ghost" onClick={onPrev} disabled={offset === 0 || loading}>Previous</button><button className="button button-ghost" onClick={onNext} disabled={cases.length < limit || loading}>Next</button></div>
      </div>
    </div>
  )
}
