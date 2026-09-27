import type { ReviewEvent } from '../types'
import { titleCase, when } from '../lib/format'
import { HistoryIcon } from './Icons'

export function EventTimeline({ events }: { events: ReviewEvent[] }) {
  if (!events.length) return <div className="soft-empty"><HistoryIcon /><strong>No audit events.</strong></div>
  return <div className="timeline">{events.map((event, index) => (
    <div className="timeline-item" key={event.id}>
      <span className="timeline-dot" />
      {index < events.length - 1 && <span className="timeline-line" />}
      <div className="timeline-body">
        <div><strong>{titleCase(event.event_type)}</strong><span>{when(event.created_at)}</span></div>
        <p>{event.from_status ? `${titleCase(event.from_status)} → ` : ''}{titleCase(event.to_status)}{event.decision ? ` · ${titleCase(event.decision)}` : ''}</p>
        {event.analyst_note && <blockquote>{event.analyst_note}</blockquote>}
        <small>{event.analyst_id ? `by ${event.analyst_id}` : 'system'} · case version {event.case_version}</small>
      </div>
    </div>
  ))}</div>
}
