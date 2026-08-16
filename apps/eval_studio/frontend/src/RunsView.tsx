import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ChevronLeft, ChevronRight, CircleStop, Maximize2, Minimize2, Pause, Play, Radio, RefreshCw, RotateCcw } from 'lucide-react'
import { api } from './api'
import type { Json } from './types'

const EVENT_WINDOW_SIZE = 500

export default function RunsView({ initialRuns }: { initialRuns: Json[] }) {
  const [runs, setRuns] = useState(initialRuns)
  const [selectedId, setSelectedId] = useState(initialRuns[0]?.id || '')
  const [events, setEvents] = useState<Json[]>([])
  const [windowStart, setWindowStart] = useState(0)
  const [totalEvents, setTotalEvents] = useState(0)
  const [activeLine, setActiveLine] = useState(0)
  const [cursor, setCursor] = useState(0)
  const [live, setLive] = useState(false)
  const [replaying, setReplaying] = useState(false)
  const [loadingEvents, setLoadingEvents] = useState(false)
  const [eventError, setEventError] = useState('')
  const [eventView, setEventView] = useState<'group-chat' | 'spans'>('group-chat')
  const [showFullEvent, setShowFullEvent] = useState(false)
  const stream = useRef<EventSource | null>(null)
  const loadSequence = useRef(0)
  const selectedRun = runs.find((run) => run.id === selectedId)

  const requestEvents = (runId: string, view: 'group-chat' | 'spans', startLine: number) => (
    view === 'group-chat'
      ? api.groupChat(runId, startLine, EVENT_WINDOW_SIZE)
      : api.events(runId, startLine, EVENT_WINDOW_SIZE)
  )

  const loadEvents = async (runId: string, view: 'group-chat' | 'spans') => {
    if (!runId) return
    const sequence = ++loadSequence.current
    stream.current?.close()
    stream.current = null
    setLive(false)
    setReplaying(false)
    setLoadingEvents(true)
    setEventError('')
    try {
      let value = await requestEvents(runId, view, 0)
      const total = Number(value.total_lines ?? value.next_line ?? value.events.length)
      if (total > EVENT_WINDOW_SIZE) {
        value = await requestEvents(runId, view, Math.max(0, total - EVENT_WINDOW_SIZE))
      }
      if (sequence !== loadSequence.current) return
      const start = Number(value.start_line ?? Math.max(0, total - value.events.length))
      setEvents(value.events)
      setWindowStart(start)
      setTotalEvents(total)
      setCursor(total)
      setActiveLine(Math.max(0, total - 1))
    } catch (cause) {
      if (sequence === loadSequence.current) {
        setEvents([])
        setWindowStart(0)
        setTotalEvents(0)
        setCursor(0)
        setActiveLine(0)
        setEventError(cause instanceof Error ? cause.message : String(cause))
      }
    } finally {
      if (sequence === loadSequence.current) setLoadingEvents(false)
    }
  }

  const goToEvent = async (requestedLine: number) => {
    if (!selectedId || !totalEvents) return
    const target = Math.max(0, Math.min(requestedLine, totalEvents - 1))
    if (target >= windowStart && target < windowStart + events.length) {
      setActiveLine(target)
      return
    }
    const sequence = ++loadSequence.current
    const maximumStart = Math.max(0, totalEvents - EVENT_WINDOW_SIZE)
    const start = Math.max(0, Math.min(target - Math.floor(EVENT_WINDOW_SIZE / 2), maximumStart))
    setLoadingEvents(true)
    setEventError('')
    try {
      const value = await requestEvents(selectedId, eventView, start)
      if (sequence !== loadSequence.current) return
      const total = Number(value.total_lines ?? totalEvents)
      setEvents(value.events)
      setWindowStart(Number(value.start_line ?? start))
      setTotalEvents(total)
      setCursor(Math.max(cursor, total))
      setActiveLine(Math.max(0, Math.min(target, total - 1)))
    } catch (cause) {
      if (sequence === loadSequence.current) {
        setEventError(cause instanceof Error ? cause.message : String(cause))
      }
    } finally {
      if (sequence === loadSequence.current) setLoadingEvents(false)
    }
  }

  useEffect(() => { void loadEvents(selectedId, eventView) }, [selectedId, eventView])
  useEffect(() => () => stream.current?.close(), [])
  useEffect(() => setShowFullEvent(false), [selectedId, eventView, activeLine])

  useEffect(() => {
    if (!replaying || !totalEvents) return
    if (activeLine >= totalEvents - 1) {
      setReplaying(false)
      return
    }
    const timer = window.setTimeout(() => { void goToEvent(activeLine + 1) }, 700)
    return () => window.clearTimeout(timer)
  }, [activeLine, replaying, totalEvents])

  const startLive = async () => {
    if (!selectedId) return
    if (totalEvents) await goToEvent(totalEvents - 1)
    stream.current?.close()
    const streamPath = eventView === 'group-chat' ? 'group-chat/stream' : 'events/stream'
    const source = new EventSource(`/api/runs/${selectedId}/${streamPath}?start_line=${cursor}`)
    source.onmessage = (message) => {
      const event = JSON.parse(message.data)
      setEvents((current) => {
        const next = [...current, event].slice(-EVENT_WINDOW_SIZE)
        setWindowStart((start) => start + Math.max(0, current.length + 1 - EVENT_WINDOW_SIZE))
        return next
      })
      setTotalEvents((current) => {
        const next = current + 1
        setActiveLine(next - 1)
        return next
      })
    }
    source.addEventListener('cursor', (message) => setCursor(Number((message as MessageEvent).data)))
    source.onerror = () => setLive(false)
    stream.current = source
    setLive(true)
  }

  const stopLive = () => {
    stream.current?.close()
    stream.current = null
    setLive(false)
  }

  const startReplay = async () => {
    setReplaying(false)
    if (totalEvents) await goToEvent(0)
    setReplaying(totalEvents > 1)
  }

  const activeLocalIndex = activeLine - windowStart
  const activeEvent = activeLocalIndex >= 0 && activeLocalIndex < events.length ? events[activeLocalIndex] : undefined
  const activeRole = displayRole(activeEvent)
  const rawActiveRole = rawRole(activeEvent)
  const roles = useMemo(
    () => Array.from(new Set(events.map(displayRole).filter((role) => role !== 'Runtime'))),
    [events],
  )
  const timelineStart = Math.max(0, Math.min(events.length - 80, activeLocalIndex - 40))
  const timelineEvents = events.slice(timelineStart, timelineStart + 80)
  const loadedRange = events.length ? `${windowStart + 1}-${windowStart + events.length}` : '—'
  const activeEventText = eventContent(activeEvent, eventView)
  const activeEventIsTruncated = activeEventText.length > 3000

  return (
    <section className="runs-view">
      <aside className="runs-list">
        <div className="panel-heading"><h2>Runs</h2><button className="icon-button" title="刷新运行记录" onClick={() => api.refreshRuns().then(setRuns)}><RefreshCw size={16} /></button></div>
        {runs.map((run) => (
          <button key={run.id} className={run.id === selectedId ? 'selected' : ''} onClick={() => setSelectedId(run.id)}>
            <span className={`run-dot status-${run.status?.status || 'unknown'}`} />
            <span><strong>{run.status?.task || run.relative_path.split('/').at(-1)}</strong><small>{run.relative_path}</small></span>
            <em>{run.status?.successful_predictions || 0}/{run.status?.expected_predictions || '?'}</em>
          </button>
        ))}
      </aside>

      <div className="run-monitor">
        <div className="monitor-header">
          <div><h1>{selectedRun?.status?.task || '选择运行记录'}</h1><span>{selectedRun?.run_dir}</span></div>
          <div className="monitor-actions">
            <div className="segmented" aria-label="运行记录类型">
              <button className={eventView === 'group-chat' ? 'active' : ''} onClick={() => setEventView('group-chat')}>GroupChat</button>
              <button className={eventView === 'spans' ? 'active' : ''} onClick={() => setEventView('spans')}>Spans</button>
            </div>
            <button className={live ? 'primary live' : 'secondary'} onClick={() => { void (live ? Promise.resolve(stopLive()) : startLive()) }}>{live ? <CircleStop size={15} /> : <Radio size={15} />}{live ? '停止实时显示' : '实时显示'}</button>
            <button className="secondary" onClick={() => { void startReplay() }}><RotateCcw size={15} />从头回放</button>
            <button className="icon-button" title={replaying ? '暂停回放' : '继续回放'} disabled={!totalEvents} onClick={() => setReplaying(!replaying)}>{replaying ? <Pause size={16} /> : <Play size={16} />}</button>
          </div>
        </div>

        <div className="run-summary-strip">
          <Metric label="状态" value={selectedRun?.status?.status || '—'} />
          <Metric label="Cases" value={`${selectedRun?.status?.successful_predictions || 0}/${selectedRun?.status?.expected_predictions || '—'}`} />
          <Metric label="Score" value={selectedRun?.metrics?.score_mean ?? selectedRun?.metrics?.accuracy ?? '—'} />
          <Metric label="模型实际成本" value={formatCost(selectedRun?.metrics?.costing?.actual_cost)} />
          <Metric label="API 等价成本" value={formatCost(selectedRun?.metrics?.costing?.api_equivalent_cost)} />
          <Metric label="污染审计" value={selectedRun?.metrics?.contamination_audit?.num_suspected_cases ?? selectedRun?.contamination_summary?.num_suspected_cases ?? '—'} />
          <Metric label="审计覆盖率" value={formatRate(selectedRun?.metrics?.contamination_audit?.audit_coverage ?? selectedRun?.contamination_summary?.audit_coverage)} />
          <Metric label="事件总数" value={totalEvents} />
          <Metric label="已加载窗口" value={loadedRange} />
        </div>

        <div className="communication-board">
          <div className="role-lanes">
            {roles.map((role) => <div key={role} className={role === activeRole ? 'active' : ''}><span /><strong>{role}</strong></div>)}
            {!roles.length && <div className="empty-state">暂无角色事件</div>}
          </div>
          <div className="active-event">
            <div className="event-title">
              <Activity size={16} />
              <strong>{activeRole}</strong>
              <span className={`event-kind kind-${eventCategory(eventType(activeEvent))}`}>{eventType(activeEvent) || 'idle'}</span>
              <div className="event-title-actions">
                {rawActiveRole !== activeRole && <small>原始来源：{rawActiveRole}</small>}
                {activeEventIsTruncated && <button className="icon-button" title={showFullEvent ? '折叠为前后摘要' : '展开完整事件'} aria-label={showFullEvent ? '折叠完整事件' : '展开完整事件'} onClick={() => setShowFullEvent(!showFullEvent)}>{showFullEvent ? <Minimize2 size={15} /> : <Maximize2 size={15} />}</button>}
              </div>
            </div>
            <pre>{loadingEvents ? '正在读取事件…' : eventError || formatEventText(activeEventText, showFullEvent)}</pre>
            <div className="event-meta">
              <span>event {totalEvents ? activeLine + 1 : 0}/{totalEvents}</span>
              <span>input {activeEvent?.input_total_positions ?? activeEvent?.input_positions ?? activeEvent?.prompt_tokens ?? '—'}</span>
              <span>output {activeEvent?.output_text_tokens ?? activeEvent?.output_tokens ?? activeEvent?.completion_tokens ?? '—'}</span>
              <span>latency {activeEvent?.model_latency_s ?? activeEvent?.latency_s ?? activeEvent?.model_generation_latency_s ?? '—'}</span>
              <span>{activeEventIsTruncated && !showFullEvent ? `摘要 3000/${activeEventText.length} 字符` : `完整 ${activeEventText.length} 字符`}</span>
            </div>
          </div>
        </div>

        <div className="replay-control">
          <button className="secondary step-button" disabled={loadingEvents || !totalEvents || activeLine <= 0} onClick={() => { setReplaying(false); void goToEvent(activeLine - 1) }}><ChevronLeft size={15} />上一条</button>
          <input aria-label="事件位置" type="range" min={0} max={Math.max(0, totalEvents - 1)} value={Math.min(activeLine, Math.max(0, totalEvents - 1))} disabled={!totalEvents} onChange={(event) => { setReplaying(false); void goToEvent(Number(event.target.value)) }} />
          <span>{totalEvents ? activeLine + 1 : 0} / {totalEvents}</span>
          <button className="secondary step-button" disabled={loadingEvents || !totalEvents || activeLine >= totalEvents - 1} onClick={() => { setReplaying(false); void goToEvent(activeLine + 1) }}>下一条<ChevronRight size={15} /></button>
        </div>

        <div className="event-timeline">
          {timelineEvents.map((event, index) => {
            const globalLine = windowStart + timelineStart + index
            const kind = eventType(event)
            return <article key={`${event.event_id || event.span_id || globalLine}-${globalLine}`} className={`event-row ${globalLine === activeLine ? 'active' : ''}`} onClick={() => setActiveLine(globalLine)}>
              <time>{event.timestamp_utc?.slice(11, 19) || `#${globalLine + 1}`}</time>
              <span className={`event-kind kind-${eventCategory(kind)}`}>{kind || 'event'}</span>
              <strong>{displayRole(event)}</strong>
              <p>{formatTimelineContent(event)}</p>
            </article>
          })}
        </div>
      </div>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: any }) {
  return <div><span>{label}</span><strong>{String(value)}</strong></div>
}

function eventType(event?: Json) {
  return String(event?.event_type || event?.span_type || '')
}

function rawRole(event?: Json) {
  return String(event?.source || event?.role || event?.sender_role || 'Runtime')
}

function displayRole(event?: Json) {
  const role = rawRole(event)
  if (role === 'MagenticOneOrchestrator') return 'Orchestrator'
  if (role === 'None' || role === 'undefined' || role === 'null') return 'Runtime'
  return role
}

function eventCategory(kind: string) {
  if (!kind) return 'neutral'
  if (kind.includes('error') || kind === 'invalid_jsonl') return 'error'
  if (kind === 'model_call_start') return 'model-start'
  if (kind === 'model_call_end') return 'model-end'
  if (kind === 'case_start' || kind === 'case_attempt_start') return 'case-start'
  if (kind === 'case_end') return 'case-end'
  if (kind === 'run_start' || kind === 'runtime_start' || kind === 'group_chat_start') return 'runtime-start'
  if (kind === 'run_end' || kind === 'runtime_end' || kind === 'group_chat_end') return 'runtime-end'
  if (kind === 'autogen_message' || kind === 'message') return 'message'
  if (kind.includes('tool')) return 'tool'
  if (kind === 'group_chat_selected') return 'selection'
  return 'neutral'
}

function formatRate(value: any) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
}

function formatCost(value?: Json) {
  const totals = value?.totals_by_currency || {}
  const parts = Object.entries(totals).map(
    ([currency, amount]) => `${currency} ${Number(amount).toFixed(4)}`,
  )
  if (!parts.length) return '—'
  return String(value?.status || '').includes('lower_bound')
    ? `≥ ${parts.join(' / ')}`
    : parts.join(' / ')
}

function primaryContent(event?: Json) {
  return event?.content ?? event?.output_preview ?? event?.final_answer ?? event?.error_message ?? event?.reason
}

function eventContent(event: Json | undefined, view: 'group-chat' | 'spans') {
  if (!event) return '等待事件'
  if (view === 'spans') return JSON.stringify(event, null, 2)
  const value = primaryContent(event)
  return value === undefined || value === ''
    ? JSON.stringify(event, null, 2)
    : typeof value === 'string' ? value : JSON.stringify(value, null, 2)
}

function formatEventText(text: string, showFull: boolean, limit = 3000) {
  if (showFull || text.length <= limit) return text
  return `${text.slice(0, Math.floor(limit / 2))}\n…\n${text.slice(-Math.floor(limit / 2))}`
}

function formatTimelineContent(event: Json) {
  const value = primaryContent(event)
  if (value !== undefined && value !== '') {
    const text = typeof value === 'string' ? value : JSON.stringify(value)
    return text.length > 220 ? `${text.slice(0, 108)} … ${text.slice(-108)}` : text
  }
  return [
    event.case_id ? `case=${event.case_id}` : '',
    event.turn !== undefined ? `turn=${event.turn}` : '',
    event.model ? `model=${event.model}` : '',
    event.status ? `status=${event.status}` : '',
  ].filter(Boolean).join(' · ') || '结构化运行事件'
}
