import type { ExplanationResponse } from '../types'
import { compactHash, formatValue, score } from '../lib/format'
import { InfoIcon, SparkIcon } from './Icons'

interface ExplanationPanelProps {
  explanation: ExplanationResponse | null
  missing: boolean
  loading: boolean
}

export function ExplanationPanel({ explanation, missing, loading }: ExplanationPanelProps) {
  if (loading) return <div className="detail-loading"><span /><span /><span /></div>
  if (missing) return (
    <div className="soft-empty">
      <SparkIcon />
      <strong>No immutable explanation snapshot for this case yet.</strong>
      <p>The current build generated snapshots for a bounded subset of the ranked queue. Queue ranking is still available and unchanged.</p>
    </div>
  )
  if (!explanation) return null

  const payload = explanation.explanation
  const top = payload.features.slice(0, 10)
  const maxAbs = Math.max(...top.map((feature) => Math.abs(feature.contribution)), 0.000001)

  return (
    <div className="explanation-stack">
      <div className="explanation-banner"><InfoIcon /><span>Attributions explain <strong>model behaviour</strong>, not causes of fraud. Scores are uncalibrated.</span></div>
      <div className="decomposition">
        <div><span>Reference score</span><strong>{score(payload.reference_score)}</strong></div>
        <span className="decomp-op">+</span>
        <div><span>Net contributions</span><strong>{payload.features.reduce((sum, item) => sum + item.contribution, 0).toFixed(4)}</strong></div>
        <span className="decomp-op">=</span>
        <div className="decomp-final"><span>Model score</span><strong>{score(payload.model_score)}</strong></div>
      </div>
      <div className="section-heading"><div><span className="eyebrow">Top drivers</span><h3>Largest absolute contributions</h3></div><small>positive raises · negative lowers</small></div>
      <div className="contribution-list">
        {top.map((feature) => {
          const width = Math.max(2, Math.abs(feature.contribution) / maxAbs * 48)
          const positive = feature.contribution >= 0
          return (
            <div className="contribution-row" key={feature.feature}>
              <div className="contribution-label"><strong>{feature.feature}</strong><span>{formatValue(feature.value)}{feature.missing_after_cleaning ? ' · treated as missing' : ''}</span></div>
              <div className="bar-track" aria-label={`${feature.feature} contribution ${feature.contribution.toFixed(6)}`}>
                <span className="axis" />
                <span className={`bar ${positive ? 'positive' : 'negative'}`} style={{ width: `${width}%`, [positive ? 'left' : 'right']: '50%' }} />
              </div>
              <strong className={positive ? 'positive-text' : 'negative-text'}>{positive ? '+' : ''}{feature.contribution.toFixed(4)}</strong>
            </div>
          )
        })}
      </div>
      <details className="provenance-card">
        <summary>Explanation provenance and limitations</summary>
        <div className="provenance-grid">
          <span>Method<strong>{payload.method}</strong></span>
          <span>Background<strong>{payload.background_size} training rows</strong></span>
          <span>Paths<strong>{payload.permutation_paths}</strong></span>
          <span>Reconstruction<strong>{payload.reconstruction_error.toExponential(2)}</strong></span>
          <span title={explanation.config_sha256}>Config<strong>{compactHash(explanation.config_sha256)}</strong></span>
          <span title={explanation.background_sha256}>Background hash<strong>{compactHash(explanation.background_sha256)}</strong></span>
        </div>
        <ul>{payload.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
      </details>
    </div>
  )
}
