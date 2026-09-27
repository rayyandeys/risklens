import { useState } from 'react'
import type { CaseDetail, Decision, ExplanationResponse, Identity, ReviewEvent } from '../types'
import { pct, score, titleCase } from '../lib/format'
import { DecisionPanel } from './DecisionPanel'
import { EventTimeline } from './EventTimeline'
import { ExplanationPanel } from './ExplanationPanel'
import { FeaturesPanel } from './FeaturesPanel'

interface CasePanelProps {
  identity: Identity
  detail: CaseDetail | null
  explanation: ExplanationResponse | null
  explanationMissing: boolean
  events: ReviewEvent[]
  loading: boolean
  decisionBusy: boolean
  decisionError: string
  onDecision: (decision: Decision, note: string) => Promise<boolean>
  onClose: () => void
}

type Tab = 'explanation' | 'features' | 'history' | 'decision'

export function CasePanel(props: CasePanelProps) {
  const { detail, loading } = props
  const [tab, setTab] = useState<Tab>('explanation')
  if (!detail && !loading) return <aside className="case-panel empty-case"><div><span className="case-empty-number">01</span><h2>Select a ranked case</h2><p>Open a queue item to inspect its raw feature snapshot, model attribution, immutable audit history and review controls.</p></div></aside>
  return (
    <aside className="case-panel">
      {loading && !detail ? <div className="case-panel-skeleton"><span/><span/><span/><span/></div> : detail && <>
        <header className="case-header">
          <div><span className="eyebrow">Case #{detail.risk_rank}</span><h2>{detail.case_id}</h2><p>Source row {detail.source_row_id.toLocaleString()} · {detail.model_name}</p></div>
          <button className="close-button" onClick={props.onClose} aria-label="Close case">×</button>
        </header>
        <div className="case-score-strip">
          <div><span>Model score</span><strong>{score(detail.risk_score)}</strong></div>
          <div><span>Queue percentile</span><strong>{pct(detail.risk_percentile, 2)}</strong></div>
          <div><span>Status</span><strong>{titleCase(detail.review_status)}</strong></div>
        </div>
        <div className="case-tabs" role="tablist">
          {(['explanation','features','history','decision'] as Tab[]).map((value) => <button key={value} className={tab === value ? 'active' : ''} onClick={() => setTab(value)} role="tab">{titleCase(value)}</button>)}
        </div>
        <div className="case-tab-body">
          {tab === 'explanation' && <ExplanationPanel explanation={props.explanation} missing={props.explanationMissing} loading={loading} />}
          {tab === 'features' && <FeaturesPanel detail={detail} explanation={props.explanation} />}
          {tab === 'history' && <EventTimeline events={props.events} />}
          {tab === 'decision' && <DecisionPanel role={props.identity.role} version={detail.version} status={detail.review_status} existingDecision={detail.decision} busy={props.decisionBusy} error={props.decisionError} onSubmit={props.onDecision} />}
        </div>
      </>}
    </aside>
  )
}
