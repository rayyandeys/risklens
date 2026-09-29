# Public read-only demo

This adds public viewing to the current RiskLens release. It does not change
training, the frozen model, research dependencies or the completed evaluation.

## Existing Render service

1. Copy the package contents into the existing local RiskLens project.
2. The package includes the rebuilt `frontend/dist`, so no local frontend build
   is required unless you edit the frontend again.
3. Commit and push the changed source, tests, docs, `render.yaml` and frontend build.
4. In Render, open the RiskLens service → Environment. Add
   `RISKLENS_PUBLIC_DEMO` with value `true`, then save and deploy.
   Existing services may not apply edits to `render.yaml` automatically.
5. Wait for the deployment to be live. Open the site in an incognito window.
6. Choose **Explore demo**, browse cases and open Monitoring. No token is needed.
   Review actions remain unavailable. Analyst sign-in returns to the login screen.

If the button is absent, check `/api/demo/config`: it should return
`{"enabled":true}`. If it returns false, check the environment variable and restart
with a deployment. A 404 indicates the new backend is not deployed yet.
If the queue is empty, verify the existing frozen queue import; do not regenerate
research results. Only runs matching the exact BAF dataset hash, frozen model
hash/name, month 5 and 3% capacity are exposed.

## Access boundary

- Anonymous routes are separate GET-only `/api/demo/*` endpoints.
- `/api/v1/*` still requires a valid expiring credential. The browser's
  `public-demo` mode marker is not a credential and cannot authorize those routes.
- No demo credential is minted or stored in the database. Browsing causes no
  review writes and does not consume rows in the credential table.
- Free-text analyst notes and analyst identities are removed from public case
  detail and event responses. Review statuses, decision categories and timestamps
  remain visible for the synthetic benchmark workflow.
- Existing explanation provenance checks and pagination bounds still apply.
- Missing explanation snapshots remain explicit; none are generated on demand.
- Setting `RISKLENS_PUBLIC_DEMO=false` disables all public data routes on restart.
  `/api/demo/config` remains available to tell the login page to hide the button.
- Public browsing uses the existing database and API resources. This release does
  not add a rate limiter or claim protection against denial-of-service traffic.

## Verification

```bat
python -m unittest discover -s tests -p test_public_demo.py -v
python scripts\check_deployment_readiness.py
```

The public-demo tests cover anonymous reads, note/identity redaction, disabled
mode, exclusion of other datasets/runs, rejected writes and continued private-API
authentication. Test fixtures use temporary databases; they do not mutate Neon.
