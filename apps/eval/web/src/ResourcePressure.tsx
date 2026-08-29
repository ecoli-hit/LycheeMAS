import type { Json } from './types'

export type PressureKind = 'vllm' | 'api' | 'gpu'

export interface PressureEntry {
  id: string
  kind: PressureKind
  pressure: Json
  title?: string
}

interface Props {
  title: string
  subtitle: string
  entries: PressureEntry[]
  emptyText?: string
}

const finitePercent = (value: unknown) => {
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, number) : 0
}

function PressureCard({ entry }: { entry: PressureEntry }) {
  const { kind, pressure } = entry
  const devices: Json[] = pressure.devices || []
  const gpuUtil = Math.max(0, ...devices.map((item) => finitePercent(item.utilization_percent)))
  const gpuMemory = Math.max(0, ...devices.map((item) => (
    Number(item.memory_total_mib) > 0
      ? Number(item.memory_used_mib || 0) / Number(item.memory_total_mib)
      : 0
  ))) * 100
  const cacheRatio = finitePercent(pressure.gpu_cache_usage_fraction) * 100
  const waitingRatio = Number(pressure.requests_running) > 0
    ? Number(pressure.requests_waiting || 0) / Number(pressure.requests_running) * 100
    : 0
  const queuePressure = Math.max(
    Number(pressure.queue_time_window_mean_s || 0) / 5,
    Number(pressure.queue_time_window_p95_s || 0) / 10,
  ) * 100
  const ttftPressure = Math.max(
    Number(pressure.time_to_first_token_window_mean_s || 0) / 6,
    Number(pressure.time_to_first_token_window_p95_s || 0) / 12,
  ) * 100
  const gpuTemperaturePressure = finitePercent(pressure.max_temperature_c) / 85 * 100
  const gpuPowerPressure = finitePercent(pressure.max_power_limit_fraction) * 100
  const preemptionPressure = Math.min(100, finitePercent(pressure.preemptions_window) * 100)
  const pressureScoreAvailable = pressure.pressure_score !== null
    && pressure.pressure_score !== undefined
    && Number.isFinite(Number(pressure.pressure_score))
  const meters = kind === 'gpu'
    ? [
        ['GPU 利用率', gpuUtil],
        ['显存分配', gpuMemory],
        ['温度 / 85°C', gpuTemperaturePressure],
        ['功耗 / 上限', gpuPowerPressure],
        ['GPU 综合压力', finitePercent(pressure.pressure_score) * 100],
      ] as [string, number][]
    : kind === 'vllm'
      ? [
          ['KV cache', cacheRatio],
          ['等待请求 / 运行请求', waitingRatio],
          ['Queue 压力', queuePressure],
          ['TTFT 压力', ttftPressure],
          ['Preemption', preemptionPressure],
          ['vLLM 综合压力', finitePercent(pressure.pressure_score) * 100],
        ] as [string, number][]
      : pressureScoreAvailable
        ? [['API 综合压力', finitePercent(pressure.pressure_score) * 100]] as [string, number][]
        : []
  const diagnosis = kind === 'gpu' && gpuUtil >= 95
    ? `计算已饱和，显存仍有 ${Math.max(0, 100 - gpuMemory).toFixed(1)}% 余量`
    : kind === 'vllm' && Number(pressure.requests_waiting || 0) > 0
      ? '服务已有等待请求；继续接纳前观察 Queue 与 TTFT'
      : kind === 'api' && !pressureScoreAvailable
        ? '服务端压力不可直接观测；依据 429、超时和重试反馈'
        : pressure.level === 'unknown'
          ? '尚无活动请求的压力样本'
          : kind === 'vllm'
            ? '当前服务负载处于可接纳范围'
            : '当前资源仍有余量'
  const kindLabel = kind === 'vllm' ? 'vLLM' : kind === 'api' ? 'API' : 'GPU'

  return <article className={`pressure-card ${pressure.level || 'unknown'}`}>
    <div className="pressure-card-title">
      <span>{entry.title || `${kindLabel} · ${entry.id}`}</span>
      <em>{diagnosis}</em>
    </div>
    <strong>{pressure.summary || pressure.reason || '尚无实时压力样本'}</strong>
    <small>{pressure.detail || (
      pressure.effective_capacity != null
        ? `在线准入容量 ${pressure.effective_capacity} / 硬上限 ${pressure.hard_capacity ?? '—'} · 当前观测负载 ${pressure.observed_load ?? '—'}`
        : '服务产生请求后显示实时压力。'
    )}</small>
    <div className="pressure-meters">
      {meters.map(([label, value]) => <div key={label}>
        <span>{label}</span><b>{value.toFixed(1)}%</b>
        <i><u style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></i>
      </div>)}
    </div>
  </article>
}

export default function ResourcePressure({ title, subtitle, entries, emptyText }: Props) {
  return <section className="runtime-pressure-panel">
    <header><strong>{title}</strong><span>{subtitle}</span></header>
    {entries.length
      ? <div className="runtime-pressure-grid">{entries.map((entry) => <PressureCard key={`${entry.kind}:${entry.id}`} entry={entry} />)}</div>
      : <div className="pressure-empty">{emptyText || '尚无压力样本。'}</div>}
  </section>
}
