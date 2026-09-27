export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function score(value: number): string {
  return value.toFixed(4)
}

export function compactHash(value: string): string {
  return value.length <= 14 ? value : `${value.slice(0, 7)}…${value.slice(-7)}`
}

export function titleCase(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString()
    return value.toLocaleString(undefined, { maximumFractionDigits: 5 })
  }
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  return String(value)
}

export function when(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
