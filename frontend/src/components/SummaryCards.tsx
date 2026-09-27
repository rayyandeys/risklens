import type { ExplanationSummary, RunSummary } from '../types'
import { pct } from '../lib/format'

interface SummaryCardsProps {
  summary: RunSummary | null
  explanationSummary: ExplanationSummary | null
}

export function SummaryCards({ summary, explanationSummary }: SummaryCardsProps) {
  const pending = summary?.status_counts.pending ?? 0
  const resolved = summary?.status_counts.resolved ?? 0
  const escalated = summary?.status_counts.escalated ?? 0
  const coverage = explanationSummary && explanationSummary.queue_cases > 0
    ? explanationSummary.explained_cases / explanationSummary.queue_cases
    : 0

  return (
    <section className="metric-grid" aria-label="Queue summary">
      <article className="metric-card"><span>Pending review</span><strong>{pending.toLocaleString()}</strong><small>ranked cases awaiting action</small></article>
      <article className="metric-card"><span>Resolved</span><strong>{resolved.toLocaleString()}</strong><small>{summary ? pct(resolved / Math.max(summary.selected_cases, 1)) : '0.0%'} of queue</small></article>
      <article className="metric-card"><span>Escalated</span><strong>{escalated.toLocaleString()}</strong><small>manual second-look workflow</small></article>
      <article className="metric-card accent"><span>Explanation coverage</span><strong>{explanationSummary?.explained_cases ?? 0}<em> / {explanationSummary?.queue_cases ?? summary?.selected_cases ?? 0}</em></strong><small>{pct(coverage)} snapshot coverage</small></article>
    </section>
  )
}
