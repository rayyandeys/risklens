# Explanation release validation — 27 September 2026

- Python 3.12 Linux local suite: 64 tests; 63 passed, one opt-in PostgreSQL skip.
- Hand-computed additive model contributions, dummy features and grouped missing
  flags match expected values. Forward/reverse orders split a two-feature
  interaction exactly in the chosen test example.
- Training reference loader rejects checksum mismatches, excludes month 5-7 rows
  from reference sampling and never loads fraud_bool.
- Synthetic real-HistGB command-line build/resume passed; model tampering is rejected.
- API tests: 401 without auth, 404 without generated explanation, viewer may read,
  invalid cached provenance returns 409, and coverage counts match stored snapshots.
- v2-to-v3 migration preserved cases, queue state, history and active credentials.
- Synthetic timing only: 50 cases, 16 background rows, 32 paths/case, 200 boosting
  iterations, 4 thread limit: 2.043 seconds; max reconstruction error 5.55e-17.
  This does not establish timing or attribution stability on the actual BAF model.
- Existing ML training/evaluation/scoring source remains unchanged.
- No BAF dataset/model artifact or user operational DB is present in this workspace.
- User passed prior live PostgreSQL gate (1 test, 2.258 seconds). The new v3
  explanation storage/read extension has PostgreSQL SQL compilation coverage but
  its live-server test remains pending.
