import { Braces, ChevronDown } from 'lucide-react'
import { useState } from 'react'
import type { Json } from './types'

const secretFields = new Set([
  'api_key',
  'access_token',
  'authorization',
  'credential',
  'password',
  'secret',
  'token',
])

function safeValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(safeValue)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value as Json).map(([key, item]) => [
    key,
    secretFields.has(key.toLowerCase()) ? '[redacted]' : safeValue(item),
  ]))
}

export default function DetailsDisclosure({
  value,
  label = '详细信息',
  className = '',
}: {
  value: unknown
  label?: string
  className?: string
}) {
  const [open, setOpen] = useState(false)
  return <div className={`entity-details ${className} ${open ? 'open' : ''}`.trim()}>
    <button type="button" className="secondary details-button" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
      <Braces size={14} />
      <span>{open ? `收起${label.replace(/^展开/, '')}` : label}</span>
      <ChevronDown size={13} className="details-chevron" />
    </button>
    {open && <pre>{JSON.stringify(safeValue(value), null, 2)}</pre>}
  </div>
}
