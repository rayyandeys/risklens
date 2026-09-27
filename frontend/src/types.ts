export type Role = 'viewer' | 'analyst' | 'admin'
export type ReviewStatus = 'pending' | 'resolved' | 'escalated' | string
export type Decision = 'confirmed_fraud' | 'legitimate' | 'escalate'

export interface Identity {
  analyst_id: string
  role: Role
  expires_at: string
}

export interface RunSummary {
  run_id: string
  experiment: string
  model_name: string
  model_sha256: string
  scored_month: number
  capacity: number
  population_cases: number
  selected_cases: number
  imported_at: string
  status_counts: Record<string, number>
  decision_counts: Record<string, number>
}

export interface CaseListItem {
  case_id: string
  source_row_id: number
  risk_score: number
  risk_rank: number
  risk_percentile: number
  review_status: ReviewStatus
  decision: string | null
  version: number
}

export interface CaseDetail extends CaseListItem {
  run_id: string
  model_name: string
  analyst_note: string
  features: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface ReviewEvent {
  id: number
  event_type: string
  from_status: string | null
  to_status: string
  decision: string | null
  analyst_note: string
  analyst_id: string | null
  case_version: number
  created_at: string
}

export interface FeatureContribution {
  feature: string
  value: string | number | null
  missing_after_cleaning: boolean
  contribution: number
  pair_std_dev: number
}

export interface ExplanationPayload {
  method: 'paired-permutation-v1'
  feature_contract: 'baf-base-v1'
  score_scale: 'uncalibrated_model_score'
  reference_score: number
  model_score: number
  reconstruction_error: number
  background_size: number
  permutation_paths: number
  model_rows_evaluated: number
  seed: number
  features: FeatureContribution[]
  limitations: string[]
}

export interface ExplanationResponse {
  explanation_id: number
  run_id: string
  case_id: string
  model_sha256: string
  input_sha256: string
  config_sha256: string
  background_sha256: string
  created_at: string
  explanation: ExplanationPayload
}

export interface ExplanationSummary {
  run_id: string
  queue_cases: number
  explained_cases: number
  remaining_cases: number
}

export type MonitoringSeverity = 'normal' | 'watch' | 'alert' | string

export interface MonitoringFeatureRow {
  feature: string
  kind: string
  psi: number
  ks_statistic: number | null
  reference_missing_rate: number | null
  target_missing_rate: number | null
  missing_rate_delta: number | null
  unseen_category_rate: number | null
  severity: MonitoringSeverity
}

export interface MonitoringMonthDrift {
  reference_month: number
  target_month: number
  feature_status: MonitoringSeverity
  max_feature_psi: number
  features_at_watch_or_alert: number
  score_psi: number
  score_ks_statistic: number
  score_status: MonitoringSeverity
  overall_status: MonitoringSeverity
}

export interface MonitoringPerformancePoint {
  month: number
  split: 'validation' | 'final_holdout' | string
  rows: number
  fraud_rows: number
  average_precision: number
  roc_auc: number
  brier_score: number
  recall_at_3pct: number
  precision_at_3pct: number
  fraud_caught_at_3pct: number
  reviewed_at_3pct: number
}

export interface MonitoringOverview {
  status: MonitoringSeverity
  dataset_sha256: string
  feature_contract: string
  governance: {
    state: string
    freeze_id: string
    opening_id: string
    model_name: string
    model_sha256: string
    removed_features: string[]
    score_policy: string
    calibration: string
    score_semantics: string
    review_capacity: number
    tie_seed: number
    test_evaluated: boolean
    decision_basis: string
    caveat: string
  }
  drift: {
    status: MonitoringSeverity
    as_of_month: number
    reference_population: string
    target_population: string
    feature: {
      status: MonitoringSeverity
      counts_by_severity: Record<string, number>
      max_psi: number
      features_at_watch_or_alert: number
      top_features: MonitoringFeatureRow[]
    }
    score: {
      reference_model: string
      psi: number
      ks_statistic: number
      reference_mean: number
      target_mean: number
      reference_p95: number
      target_p95: number
      severity: MonitoringSeverity
      note: string
    }
    month_over_month: MonitoringMonthDrift[]
    alert_count: number
    threshold_note: string
    interpretation: string
  }
  stability: {
    source_model: string
    repeats: number
    capacity: number
    queue_jaccard_mean: number
    queue_jaccard_min: number
    queue_jaccard_max: number
    reference_retention_mean: number
    rank_correlation_mean: number
    boundary_unstable_cases: number
    selected_calibration: string
    raw_brier: number
    raw_ece: number
    note: string
  }
  performance: {
    timeline: MonitoringPerformancePoint[]
    primary_holdout: {
      rows: number
      fraud_rows: number
      reviewed: number
      fraud_caught: number
      recall_at_3pct: number
      precision_at_3pct: number
      random_expected_tp: number
    }
    pooled_holdout: {
      average_precision: number
      roc_auc: number
      brier_score: number
    }
    generalization_delta: {
      note: string
      recall_at_3pct_delta: number
      precision_at_3pct_delta: number
      average_precision_delta: number
      roc_auc_delta: number
      brier_score_delta: number
    }
    protocol_note: string
  }
  limitations: string[]
}
