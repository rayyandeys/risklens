import { FormEvent, useState } from 'react'
import { ApiError, api } from '../api'
import type { Identity } from '../types'
import { ShieldIcon } from './Icons'

interface LoginGateProps {
  onAuthenticated: (token: string, identity: Identity) => void
}

export function LoginGate({ onAuthenticated }: LoginGateProps) {
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    const cleaned = token.trim()
    if (!cleaned) return
    setBusy(true)
    setError('')
    try {
      const identity = await api.me(cleaned)
      sessionStorage.setItem('risklens_token', cleaned)
      onAuthenticated(cleaned, identity)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : 'Could not reach the RiskLens API.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="login-shell">
      <div className="login-orbit login-orbit-one" />
      <div className="login-orbit login-orbit-two" />
      <section className="login-card">
        <div className="brand-lockup brand-lockup-large">
          <span className="brand-mark"><ShieldIcon /></span>
          <div><strong>RiskLens</strong><span>Analyst Console</span></div>
        </div>
        <div className="login-copy">
          <span className="eyebrow">Authenticated review workspace</span>
          <h1>Investigate model-ranked cases with evidence in context.</h1>
          <p>Connect with an opaque RiskLens API credential. The console never asks for a password and keeps the token only for this browser tab.</p>
        </div>
        <form onSubmit={submit} className="token-form">
          <label htmlFor="token">Bearer credential</label>
          <div className="token-row">
            <input
              id="token"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder="rl_••••••••••••••••"
              aria-invalid={Boolean(error)}
            />
            <button className="button button-primary" type="submit" disabled={busy || !token.trim()}>
              {busy ? 'Connecting…' : 'Open console'}
            </button>
          </div>
          {error && <div className="form-error" role="alert">{error}</div>}
        </form>
        <div className="login-footnote">
          <span className="status-dot" />
          Short-lived role-based credential · stored only for this browser tab
        </div>
      </section>
    </main>
  )
}
