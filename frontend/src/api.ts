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

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export const PUBLIC_DEMO = 'public-demo'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const demo = token === PUBLIC_DEMO
  if (demo && init?.method && init.method !== 'GET') {
    throw new ApiError(403, 'Public demo is read-only.')
  }
  const route = demo ? path.replace(/^\/api\/v1\//, '/api/demo/') : path
  const response = await fetch(`${API_BASE}${route}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...(!demo && token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const payload = (await response.json()) as { detail?: unknown }
      if (typeof payload.detail === 'string') detail = payload.detail
      else if (payload.detail) detail = JSON.stringify(payload.detail)
    } catch {
      // Keep the HTTP status text when the body is not JSON.
    }
    throw new ApiError(response.status, detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  demoConfig: () => request<{ enabled: boolean }>('/api/demo/config', ''),
  me: (token: string) => request<Identity>('/api/v1/auth/me', token),
  runs: (token: string) => request<RunSummary[]>('/api/v1/runs', token),
  monitoring: (token: string) => request<MonitoringOverview>('/api/v1/monitoring/overview', token),
  summary: (token: string, runId: string) =>
    request<RunSummary>(`/api/v1/runs/${encodeURIComponent(runId)}/summary`, token),
  explanationSummary: (token: string, runId: string) =>
    request<ExplanationSummary>(`/api/v1/runs/${encodeURIComponent(runId)}/explanations/summary`, token),
  cases: (
    token: string,
    runId: string,
    params: { status?: string; decision?: string; minScore?: number; limit: number; offset: number },
  ) => {
    const query = new URLSearchParams({ limit: String(params.limit), offset: String(params.offset) })
    if (params.status) query.set('status', params.status)
    if (params.decision) query.set('decision', params.decision)
    if (params.minScore !== undefined) query.set('min_score', String(params.minScore))
    return request<CaseListItem[]>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cases?${query.toString()}`,
      token,
    )
  },
  caseDetail: (token: string, runId: string, caseId: string) =>
    request<CaseDetail>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}`,
      token,
    ),
  events: (token: string, runId: string, caseId: string) =>
    request<ReviewEvent[]>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/events`,
      token,
    ),
  explanation: (token: string, runId: string, caseId: string) =>
    request<ExplanationResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/explanation`,
      token,
    ),
  decision: (
    token: string,
    runId: string,
    caseId: string,
    payload: { decision: Decision; analyst_note: string; expected_version: number },
  ) =>
    request<CaseDetail>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/decision`,
      token,
      { method: 'POST', body: JSON.stringify(payload) },
    ),
}
