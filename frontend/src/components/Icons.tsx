import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

const base = { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }

export const ShieldIcon = (props: IconProps) => <svg {...base} {...props}><path d="M12 3 4.5 6.2v5.4c0 4.5 3 7.7 7.5 9.4 4.5-1.7 7.5-4.9 7.5-9.4V6.2L12 3Z"/><path d="m9.2 12 1.8 1.8 3.9-4"/></svg>
export const QueueIcon = (props: IconProps) => <svg {...base} {...props}><path d="M5 6h14M5 12h14M5 18h9"/><circle cx="3" cy="6" r=".6" fill="currentColor" stroke="none"/><circle cx="3" cy="12" r=".6" fill="currentColor" stroke="none"/><circle cx="3" cy="18" r=".6" fill="currentColor" stroke="none"/></svg>
export const PulseIcon = (props: IconProps) => <svg {...base} {...props}><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>
export const SearchIcon = (props: IconProps) => <svg {...base} {...props}><circle cx="11" cy="11" r="6"/><path d="m16 16 4 4"/></svg>
export const ArrowIcon = (props: IconProps) => <svg {...base} {...props}><path d="m9 18 6-6-6-6"/></svg>
export const LogoutIcon = (props: IconProps) => <svg {...base} {...props}><path d="M10 5H5v14h5M14 8l4 4-4 4M8 12h10"/></svg>
export const SparkIcon = (props: IconProps) => <svg {...base} {...props}><path d="m12 3 1.2 4.3L17 9l-3.8 1.7L12 15l-1.2-4.3L7 9l3.8-1.7L12 3Z"/><path d="m18.5 14 .7 2.3 2.3.7-2.3.7-.7 2.3-.7-2.3-2.3-.7 2.3-.7.7-2.3Z"/></svg>
export const HistoryIcon = (props: IconProps) => <svg {...base} {...props}><path d="M4 12a8 8 0 1 0 2.3-5.7L4 8.5"/><path d="M4 4v4.5h4.5M12 8v4l3 2"/></svg>
export const DatabaseIcon = (props: IconProps) => <svg {...base} {...props}><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>
export const InfoIcon = (props: IconProps) => <svg {...base} {...props}><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>
