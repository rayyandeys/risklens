import type { CaseDetail, ExplanationResponse } from '../types'
import { formatValue, titleCase } from '../lib/format'

interface FeaturesPanelProps {
  detail: CaseDetail
  explanation: ExplanationResponse | null
}

export function FeaturesPanel({ detail, explanation }: FeaturesPanelProps) {
  const contributions = new Map(explanation?.explanation.features.map((item) => [item.feature, item]))
  const rows = Object.entries(detail.features).map(([feature, value]) => ({ feature, value, contribution: contributions.get(feature) }))
  return (
    <div className="feature-table-wrap">
      <table className="feature-table">
        <thead><tr><th>Raw feature</th><th>Value</th><th>Contribution</th></tr></thead>
        <tbody>{rows.map(({ feature, value, contribution }) => (
          <tr key={feature}>
            <td><strong>{titleCase(feature)}</strong><small>{feature}</small></td>
            <td>{formatValue(value)}{contribution?.missing_after_cleaning && <span className="missing-tag">cleaned missing</span>}</td>
            <td>{contribution ? <span className={contribution.contribution >= 0 ? 'positive-text' : 'negative-text'}>{contribution.contribution >= 0 ? '+' : ''}{contribution.contribution.toFixed(5)}</span> : '—'}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}
