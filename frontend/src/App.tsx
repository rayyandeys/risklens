import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, api } from './api'
import { CasePanel } from './components/CasePanel'
import { LoginGate } from './components/LoginGate'
import { MonitoringPage } from './components/MonitoringPage'
import { QueueTable } from './components/QueueTable'
import { SearchIcon } from './components/Icons'
import { Sidebar, type ConsoleView } from './components/Sidebar'
import { SummaryCards } from './components/SummaryCards'
import type {
  CaseDetail,
  CaseListItem,
  Decision,
  ExplanationResponse,
  ExplanationSummary,
  Identity,
  MonitoringOverview,
  ReviewEvent,
  RunSummary,
} from './types'

const PAGE_SIZE = 50

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem('risklens_token') ?? '')
  const [identity, setIdentity] = useState<Identity | null>(null)
  const [authReady, setAuthReady] = useState(false)
  const [view, setView] = useState<ConsoleView>('queue')
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runId, setRunId] = useState('')
  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [explanationSummary, setExplanationSummary] = useState<ExplanationSummary | null>(null)
  const [cases, setCases] = useState<CaseListItem[]>([])
  const [queueLoading, setQueueLoading] = useState(false)
  const [statusFilter, setStatusFilter] = useState('pending')
  const [decisionFilter, setDecisionFilter] = useState('')
  const [minScore, setMinScore] = useState('')
  const [offset, setOffset] = useState(0)
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const [detail, setDetail] = useState<CaseDetail | null>(null)
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [explanationMissing, setExplanationMissing] = useState(false)
  const [events, setEvents] = useState<ReviewEvent[]>([])
  const [detailLoading, setDetailLoading] = useState(false)
  const [decisionBusy, setDecisionBusy] = useState(false)
  const [decisionError, setDecisionError] = useState('')
  const [fatalError, setFatalError] = useState('')
  const [queueRefresh, setQueueRefresh] = useState(0)
  const [monitoring, setMonitoring] = useState<MonitoringOverview | null>(null)
  const [monitoringLoading, setMonitoringLoading] = useState(false)
  const [monitoringError, setMonitoringError] = useState('')

  const logout = useCallback(() => {
    sessionStorage.removeItem('risklens_token')
    setToken('')
    setIdentity(null)
    setRuns([])
    setRunId('')
    setSelectedCaseId(null)
    setDetail(null)
    setMonitoring(null)
  }, [])

  useEffect(() => {
    let cancelled = false
    if (!token) { setAuthReady(true); return }
    api.me(token)
      .then((me) => { if (!cancelled) setIdentity(me) })
      .catch(() => { if (!cancelled) logout() })
      .finally(() => { if (!cancelled) setAuthReady(true) })
    return () => { cancelled = true }
  }, [token, logout])

  useEffect(() => {
    if (!identity || !token) return
    let cancelled = false
    setFatalError('')
    api.runs(token).then((items) => {
      if (cancelled) return
      setRuns(items)
      setRunId((current) => current && items.some((run) => run.run_id === current) ? current : (items[0]?.run_id ?? ''))
    }).catch((error) => {
      if (error instanceof ApiError && error.status === 401) logout()
      else if (!cancelled) setFatalError(error instanceof Error ? error.message : 'Could not load workflow runs.')
    })
    return () => { cancelled = true }
  }, [identity, token, logout])

  const loadRunSummary = useCallback(async () => {
    if (!token || !runId) return
    try {
      const [runSummary, explanationCoverage] = await Promise.all([
        api.summary(token, runId),
        api.explanationSummary(token, runId),
      ])
      setSummary(runSummary)
      setExplanationSummary(explanationCoverage)
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) logout()
      else setFatalError(error instanceof Error ? error.message : 'Could not load run summary.')
    }
  }, [token, runId, logout])

  useEffect(() => { void loadRunSummary() }, [loadRunSummary])

  useEffect(() => {
    if (!token || !runId || view !== 'queue') return
    let cancelled = false
    setQueueLoading(true)
    setFatalError('')
    const parsed = minScore.trim() === '' ? undefined : Number(minScore)
    api.cases(token, runId, {
      status: statusFilter || undefined,
      decision: decisionFilter || undefined,
      minScore: parsed !== undefined && Number.isFinite(parsed) ? parsed : undefined,
      limit: PAGE_SIZE,
      offset,
    }).then((items) => {
      if (!cancelled) setCases(items)
    }).catch((error) => {
      if (error instanceof ApiError && error.status === 401) logout()
      else if (!cancelled) setFatalError(error instanceof Error ? error.message : 'Could not load queue.')
    }).finally(() => { if (!cancelled) setQueueLoading(false) })
    return () => { cancelled = true }
  }, [token, runId, statusFilter, decisionFilter, minScore, offset, queueRefresh, view, logout])

  const loadMonitoring = useCallback(async () => {
    if (!token) return
    setMonitoringLoading(true)
    setMonitoringError('')
    try {
      setMonitoring(await api.monitoring(token))
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) logout()
      else setMonitoringError(error instanceof Error ? error.message : 'Could not load monitoring evidence.')
    } finally {
      setMonitoringLoading(false)
    }
  }, [token, logout])

  useEffect(() => {
    if (view === 'monitoring' && !monitoring && !monitoringLoading && !monitoringError) void loadMonitoring()
  }, [view, monitoring, monitoringLoading, monitoringError, loadMonitoring])

  const loadCase = useCallback(async (caseId: string) => {
    if (!token || !runId) return
    setSelectedCaseId(caseId)
    setDetailLoading(true)
    setDecisionError('')
    try {
      const [caseDetail, caseEvents, explanationResult] = await Promise.all([
        api.caseDetail(token, runId, caseId),
        api.events(token, runId, caseId),
        api.explanation(token, runId, caseId).then((value) => ({ value, missing: false })).catch((error) => {
          if (error instanceof ApiError && error.status === 404) return { value: null, missing: true }
          throw error
        }),
      ])
      setDetail(caseDetail)
      setEvents(caseEvents)
      setExplanation(explanationResult.value)
      setExplanationMissing(explanationResult.missing)
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) logout()
      else setFatalError(error instanceof Error ? error.message : 'Could not load case.')
    } finally {
      setDetailLoading(false)
    }
  }, [token, runId, logout])

  async function submitDecision(decision: Decision, note: string): Promise<boolean> {
    if (!token || !runId || !detail) return false
    setDecisionBusy(true)
    setDecisionError('')
    try {
      const updated = await api.decision(token, runId, detail.case_id, { decision, analyst_note: note, expected_version: detail.version })
      setDetail(updated)
      const refreshedEvents = await api.events(token, runId, detail.case_id)
      setEvents(refreshedEvents)
      await loadRunSummary()
      setQueueRefresh((value) => value + 1)
      return true
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setDecisionError('This case changed in another session. The latest version has been reloaded; review it before retrying.')
        await loadCase(detail.case_id)
      } else if (error instanceof ApiError && error.status === 401) {
        logout()
      } else {
        setDecisionError(error instanceof Error ? error.message : 'Could not save review decision.')
      }
      return false
    } finally {
      setDecisionBusy(false)
    }
  }

  const activeRun = useMemo(() => runs.find((run) => run.run_id === runId) ?? null, [runs, runId])

  if (!authReady) return <div className="boot-screen"><div className="boot-mark">RL</div><span>Verifying session…</span></div>
  if (!identity) return <LoginGate onAuthenticated={(newToken, me) => { setToken(newToken); setIdentity(me); setAuthReady(true) }} />

  return (
    <div className="app-shell">
      <Sidebar
        identity={identity}
        runs={runs}
        selectedRunId={runId}
        view={view}
        onViewChange={setView}
        onRunChange={(next) => { setRunId(next); setOffset(0); setSelectedCaseId(null); setDetail(null) }}
        onLogout={logout}
      />
      <main className="workspace">
        <header className="workspace-header">
          {view === 'queue' ? (
            <div><span className="eyebrow">Operational review · validation month {activeRun?.scored_month ?? '—'}</span><h1>Ranked review queue</h1><p>Prioritize scarce analyst capacity, inspect model behaviour, and preserve every review action.</p></div>
          ) : (
            <div><span className="eyebrow">Evidence-backed operations · frozen protocol</span><h1>Model monitoring</h1><p>Separate population drift, model behaviour, operational stability and sealed generalization evidence without silently retuning the system.</p></div>
          )}
          <div className="system-state">
            <span className={view === 'monitoring' && monitoring?.status === 'alert' ? 'status-dot status-dot-alert' : 'status-dot'} />
            <div>
              <strong>{view === 'queue' ? 'API connected' : monitoring?.status === 'alert' ? 'Monitoring alert' : 'Monitoring connected'}</strong>
              <small>{view === 'queue' ? (activeRun?.model_name ?? 'Waiting for workflow run') : (monitoring?.governance.model_name ?? 'Loading immutable reports')}</small>
            </div>
          </div>
        </header>

        {view === 'queue' ? (
          <>
            {fatalError && <div className="global-error" role="alert">{fatalError}</div>}
            <SummaryCards summary={summary} explanationSummary={explanationSummary} />

            <section className="review-layout">
              <div className="queue-card">
                <div className="queue-toolbar">
                  <div><span className="eyebrow">Priority ordered</span><h2>Analyst queue</h2></div>
                  <div className="filters">
                    <label><span>Status</span><select value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setOffset(0) }}><option value="">All statuses</option><option value="pending">Pending</option><option value="resolved">Resolved</option><option value="escalated">Escalated</option></select></label>
                    <label><span>Decision</span><select value={decisionFilter} onChange={(event) => { setDecisionFilter(event.target.value); setOffset(0) }}><option value="">Any decision</option><option value="confirmed_fraud">Confirmed fraud</option><option value="legitimate">Legitimate</option><option value="escalate">Escalate</option></select></label>
                    <label className="score-filter"><span>Min score</span><div><SearchIcon /><input type="number" min="0" max="1" step="0.01" value={minScore} onChange={(event) => { setMinScore(event.target.value); setOffset(0) }} placeholder="0.00" /></div></label>
                  </div>
                </div>
                <QueueTable cases={cases} selectedCaseId={selectedCaseId} loading={queueLoading} offset={offset} limit={PAGE_SIZE} onSelect={(caseId) => void loadCase(caseId)} onPrev={() => setOffset(Math.max(0, offset - PAGE_SIZE))} onNext={() => setOffset(offset + PAGE_SIZE)} />
              </div>
              <CasePanel identity={identity} detail={detail} explanation={explanation} explanationMissing={explanationMissing} events={events} loading={detailLoading} decisionBusy={decisionBusy} decisionError={decisionError} onDecision={submitDecision} onClose={() => { setSelectedCaseId(null); setDetail(null); setExplanation(null); setEvents([]) }} />
            </section>
          </>
        ) : (
          <MonitoringPage data={monitoring} loading={monitoringLoading} error={monitoringError} onRetry={() => void loadMonitoring()} />
        )}
      </main>
    </div>
  )
}
