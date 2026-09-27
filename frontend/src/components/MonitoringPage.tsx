import type { MonitoringOverview, MonitoringPerformancePoint, MonitoringSeverity } from '../types'
import { compactHash, pct, titleCase } from '../lib/format'
import { InfoIcon, PulseIcon, ShieldIcon } from './Icons'

interface MonitoringPageProps {
  data: MonitoringOverview | null
  loading: boolean
  error: string
  onRetry: () => void
}

function severityLabel(value: MonitoringSeverity) {
  return value === 'alert' ? 'Alert' : value === 'watch' ? 'Watch' : 'Normal'
}

function severityClass(value: MonitoringSeverity) {
  return `severity-badge severity-${value}`
}

function delta(value: number, digits = 2) {
  const pp = value * 100
  return `${pp >= 0 ? '+' : ''}${pp.toFixed(digits)} pp`
}

function PerformanceChart({ points }: { points: MonitoringPerformancePoint[] }) {
  if (points.length < 2) return null
  const width = 520
  const height = 180
  const left = 40
  const right = 18
  const top = 18
  const bottom = 34
  const innerW = width - left - right
  const innerH = height - top - bottom
  const all = points.flatMap((point) => [point.recall_at_3pct, point.precision_at_3pct])
  const min = Math.max(0, Math.floor((Math.min(...all) - 0.04) * 10) / 10)
  const max = Math.min(1, Math.ceil((Math.max(...all) + 0.04) * 10) / 10)
  const range = Math.max(0.1, max - min)
  const x = (i: number) => left + (innerW * i) / Math.max(1, points.length - 1)
  const y = (value: number) => top + innerH - ((value - min) / range) * innerH
  const line = (key: 'recall_at_3pct' | 'precision_at_3pct') => points.map((point, i) => `${x(i)},${y(point[key])}`).join(' ')
  const ticks = [min, min + range / 2, max]

  return (
    <div className="performance-chart-wrap">
      <svg className="performance-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recall and precision at three percent review capacity for months five through seven">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
            <text x={left - 8} y={y(tick) + 3} textAnchor="end" className="chart-label">{pct(tick, 0)}</text>
          </g>
        ))}
        <polyline points={line('recall_at_3pct')} className="chart-line chart-recall" />
        <polyline points={line('precision_at_3pct')} className="chart-line chart-precision" />
        {points.map((point, i) => (
          <g key={point.month}>
            <circle cx={x(i)} cy={y(point.recall_at_3pct)} r="4" className="chart-dot chart-recall-dot" />
            <circle cx={x(i)} cy={y(point.precision_at_3pct)} r="4" className="chart-dot chart-precision-dot" />
            <text x={x(i)} y={height - 10} textAnchor="middle" className="chart-label">M{point.month}</text>
          </g>
        ))}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-swatch recall" /> Recall@3%</span>
        <span><i className="legend-swatch precision" /> Precision@3%</span>
        <span className="chart-split-note">M5 validation · M6–7 sealed holdout</span>
      </div>
    </div>
  )
}

export function MonitoringPage({ data, loading, error, onRetry }: MonitoringPageProps) {
  if (loading && !data) {
    return <div className="monitoring-loading"><span /><span /><span /><span /></div>
  }
  if (error && !data) {
    return (
      <div className="monitoring-unavailable">
        <PulseIcon />
        <h2>Monitoring evidence unavailable</h2>
        <p>{error}</p>
        <button className="button" onClick={onRetry}>Retry</button>
      </div>
    )
  }
  if (!data) return null

  const drift = data.drift
  const holdout = data.performance.primary_holdout
  const governance = data.governance
  const topMax = Math.max(...drift.feature.top_features.map((row) => row.psi), 0.001)
  const generalization = data.performance.generalization_delta

  return (
    <div className="monitoring-page">
      {error && <div className="global-error" role="alert">{error}</div>}

      <section className="monitoring-hero-grid">
        <article className="monitor-kpi monitor-kpi-alert">
          <span>Feature drift</span>
          <div className="monitor-kpi-value"><strong>{severityLabel(drift.feature.status)}</strong><span className={severityClass(drift.feature.status)}>{drift.feature.counts_by_severity.alert ?? 0} alert</span></div>
          <small>{drift.feature.features_at_watch_or_alert} monitored features at watch/alert · max PSI {drift.feature.max_psi.toFixed(3)}</small>
        </article>
        <article className="monitor-kpi">
          <span>Score drift</span>
          <div className="monitor-kpi-value"><strong>{severityLabel(drift.score.severity)}</strong><span className={severityClass(drift.score.severity)}>PSI {drift.score.psi.toFixed(4)}</span></div>
          <small>Development reference score distribution remained within the engineering normal band.</small>
        </article>
        <article className="monitor-kpi">
          <span>Final Recall@3%</span>
          <div className="monitor-kpi-value"><strong>{pct(holdout.recall_at_3pct, 2)}</strong><span className="monitor-chip">{holdout.fraud_caught.toLocaleString()} caught</span></div>
          <small>{holdout.reviewed.toLocaleString()} reviews across {holdout.rows.toLocaleString()} untouched applications.</small>
        </article>
        <article className="monitor-kpi monitor-kpi-governance">
          <span>Governance state</span>
          <div className="monitor-kpi-value"><strong>Frozen</strong><span className="severity-badge severity-normal">Tested once</span></div>
          <small>{governance.model_name} · direct age input removed · raw model score.</small>
        </article>
      </section>

      <section className="monitoring-callout">
        <InfoIcon />
        <div>
          <strong>Input drift is an investigation signal, not a failure verdict.</strong>
          <p>RiskLens detected material covariate change by month 5 while the development score distribution stayed normal. The sealed holdout is reported separately so drift evidence is not confused with final-model performance.</p>
        </div>
      </section>

      <section className="monitoring-grid monitoring-grid-main">
        <article className="monitor-card drift-card">
          <div className="monitor-card-head">
            <div><span className="eyebrow">Population shift</span><h2>Top drift signals</h2></div>
            <span className={severityClass(drift.feature.status)}>{severityLabel(drift.feature.status)}</span>
          </div>
          <div className="drift-list">
            {drift.feature.top_features.slice(0, 8).map((row) => (
              <div className="drift-row" key={row.feature}>
                <div className="drift-name"><strong>{row.feature}</strong><small>{row.kind} · KS {row.ks_statistic?.toFixed(3) ?? '—'}</small></div>
                <div className="drift-bar-track"><span style={{ width: `${Math.max(2, Math.min(100, (row.psi / topMax) * 100))}%` }} /></div>
                <div className="drift-value"><strong>{row.psi.toFixed(3)}</strong><span className={severityClass(row.severity)}>{severityLabel(row.severity)}</span></div>
              </div>
            ))}
          </div>
          <div className="card-footnote">PSI and missingness cutoffs are engineering triage heuristics, not significance tests.</div>
        </article>

        <article className="monitor-card performance-card">
          <div className="monitor-card-head">
            <div><span className="eyebrow">Frozen operating point</span><h2>Generalization</h2></div>
            <span className="monitor-chip">3% capacity</span>
          </div>
          <PerformanceChart points={data.performance.timeline} />
          <div className="performance-deltas">
            <div><span>Recall Δ</span><strong className={generalization.recall_at_3pct_delta < 0 ? 'delta-negative' : 'delta-positive'}>{delta(generalization.recall_at_3pct_delta)}</strong></div>
            <div><span>Precision Δ</span><strong className={generalization.precision_at_3pct_delta < 0 ? 'delta-negative' : 'delta-positive'}>{delta(generalization.precision_at_3pct_delta)}</strong></div>
            <div><span>Pooled AP</span><strong>{data.performance.pooled_holdout.average_precision.toFixed(4)}</strong></div>
            <div><span>Pooled AUC</span><strong>{data.performance.pooled_holdout.roc_auc.toFixed(4)}</strong></div>
          </div>
        </article>
      </section>

      <section className="monitoring-grid monitoring-grid-secondary">
        <article className="monitor-card">
          <div className="monitor-card-head"><div><span className="eyebrow">Month over month</span><h2>Drift timeline</h2></div></div>
          <div className="month-drift-table">
            <div className="month-drift-head"><span>Window</span><span>Max feature PSI</span><span>Signals</span><span>Score PSI</span><span>Status</span></div>
            {drift.month_over_month.map((row) => (
              <div className="month-drift-row" key={`${row.reference_month}-${row.target_month}`}>
                <strong>M{row.reference_month} → M{row.target_month}</strong>
                <span>{row.max_feature_psi.toFixed(3)}</span>
                <span>{row.features_at_watch_or_alert}</span>
                <span>{row.score_psi.toFixed(4)}</span>
                <span className={severityClass(row.overall_status)}>{severityLabel(row.overall_status)}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="monitor-card stability-card">
          <div className="monitor-card-head"><div><span className="eyebrow">Development sensitivity</span><h2>Review-boundary stability</h2></div></div>
          <div className="stability-metrics">
            <div><span>Queue Jaccard</span><strong>{pct(data.stability.queue_jaccard_mean, 1)}</strong><small>mean across {data.stability.repeats} refits</small></div>
            <div><span>Queue retained</span><strong>{pct(data.stability.reference_retention_mean, 1)}</strong><small>reference cases retained</small></div>
            <div><span>Rank correlation</span><strong>{data.stability.rank_correlation_mean.toFixed(3)}</strong><small>mean Spearman</small></div>
            <div><span>Boundary-sensitive</span><strong>{data.stability.boundary_unstable_cases.toLocaleString()}</strong><small>development cases</small></div>
          </div>
          <p className="monitor-note">{data.stability.note}</p>
        </article>
      </section>

      <section className="monitoring-grid monitoring-grid-bottom">
        <article className="monitor-card governance-card">
          <div className="monitor-card-head"><div><span className="eyebrow">Immutable lineage</span><h2>Model governance</h2></div><ShieldIcon /></div>
          <div className="governance-list">
            <div><span>Model</span><strong>{governance.model_name}</strong></div>
            <div><span>Model SHA-256</span><strong title={governance.model_sha256}>{compactHash(governance.model_sha256)}</strong></div>
            <div><span>Freeze</span><strong>{governance.freeze_id}</strong></div>
            <div><span>Final opening</span><strong>{governance.opening_id}</strong></div>
            <div><span>Score policy</span><strong>{titleCase(governance.score_policy)}</strong></div>
            <div><span>Calibration</span><strong>{titleCase(governance.calibration)}</strong></div>
            <div><span>Review policy</span><strong>{pct(governance.review_capacity, 0)} · tie seed {governance.tie_seed}</strong></div>
            <div><span>Removed direct input</span><strong>{governance.removed_features.join(', ') || 'None'}</strong></div>
          </div>
        </article>

        <article className="monitor-card holdout-card">
          <div className="monitor-card-head"><div><span className="eyebrow">Sealed evidence</span><h2>Final holdout</h2></div><span className="severity-badge severity-normal">Complete</span></div>
          <div className="holdout-summary">
            <div><span>Fraud caught</span><strong>{holdout.fraud_caught.toLocaleString()} <em>/ {holdout.fraud_rows.toLocaleString()}</em></strong></div>
            <div><span>Recall@3%</span><strong>{pct(holdout.recall_at_3pct, 2)}</strong></div>
            <div><span>Precision@3%</span><strong>{pct(holdout.precision_at_3pct, 2)}</strong></div>
            <div><span>Random expected TP</span><strong>{holdout.random_expected_tp.toFixed(1)}</strong></div>
          </div>
          <p className="monitor-note">No model, feature, calibration, capacity or threshold selection is permitted from these final-test results.</p>
        </article>
      </section>

      <details className="monitoring-limitations">
        <summary>Evidence boundaries & limitations</summary>
        <ul>{data.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
      </details>
    </div>
  )
}
