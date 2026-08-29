import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, BarChart3, ChevronLeft, ChevronRight, Maximize2, Minimize2, RefreshCw, RotateCcw, Search } from 'lucide-react'
import { api } from './api'
import type { Json } from './types'

const TRIAL_WINDOW_SIZE = 100
const TRACE_WINDOW_SIZE = 120
const EVENT_WINDOW_SIZE = 200
const RUN_PAGE_SIZE = 100
type RunView = 'metrics' | 'results' | 'trace' | 'events' | 'evidence'

const METRIC_CATEGORIES = [
  { id: 'task', label: '任务结果', englishLabel: 'Task Outcome', description: '是否完成 benchmark 目标' },
  { id: 'coordination', label: '协作过程', englishLabel: 'Coordination', description: '角色如何参与和交换信息' },
  { id: 'efficiency', label: '运行效率', englishLabel: 'Efficiency', description: '每个 Trial 消耗的 token 和时间' },
  { id: 'reliability', label: '运行可靠性', englishLabel: 'Reliability', description: '工具与执行过程是否稳定' },
]

const METRIC_LABELS: Record<string, [string, string]> = {
  'task.official_score': ['官方任务得分', 'Official Task Score'],
  'coordination.message_repetition_rate': ['消息重复率', 'Message Repetition Rate'],
  'coordination.role_participation_balance': ['角色参与均衡度', 'Role Participation Balance'],
  'coordination.model_call_participation_balance': ['模型调用参与均衡度', 'Model-call Participation Balance'],
  'coordination.controller_model_call_ratio': ['控制节点调用占比', 'Controller Model-call Ratio'],
  'coordination.controller_output_token_ratio': ['控制节点输出 Token 占比', 'Controller Output-token Ratio'],
  'coordination.coordination_operation_count': ['协调操作数', 'Coordination Operation Count'],
  'coordination.replan_count': ['重规划次数', 'Replan Count'],
  'coordination.stall_rate': ['停滞判定率', 'Stall Detection Rate'],
  'coordination.role_switch_rate': ['角色切换率', 'Role Switch Rate'],
  'coordination.active_role_count': ['活跃角色数', 'Active Role Count'],
  'coordination.agent_message_count': ['智能体消息数', 'Agent Message Count'],
  'efficiency.input_tokens': ['输入 Token 数', 'Input Tokens'],
  'efficiency.output_tokens': ['输出 Token 数', 'Output Tokens'],
  'efficiency.model_call_latency': ['模型调用总延迟', 'Model-call Latency'],
  'efficiency.provider_queue_time': ['服务端排队时间', 'Provider Queue Time'],
  'efficiency.time_to_first_token': ['首 Token 延迟', 'Time to First Token'],
  'efficiency.generation_throughput': ['生成吞吐率', 'Generation Throughput'],
  'efficiency.model_call_count': ['模型调用数', 'Model-call Count'],
  'efficiency.tool_execution_count': ['工具执行数', 'Tool Execution Count'],
  'efficiency.tool_execution_latency': ['工具执行总延迟', 'Tool Execution Latency'],
  'efficiency.trial_wall_time': ['独立运行墙钟时间', 'Trial Wall Time'],
  'reliability.tool_error_rate': ['工具错误率', 'Tool Error Rate'],
  'reliability.model_call_error_rate': ['模型调用错误率', 'Model-call Error Rate'],
  'reliability.trial_retry_count': ['独立运行重试数', 'Trial Retry Count'],
  'reliability.trial_runtime_success': ['独立运行成功率', 'Trial Runtime Success'],
  'reliability.result_contract_valid': ['结果合同有效率', 'Result Contract Validity'],
  'reliability.post_tool_failure_trial_completion': ['工具失败后运行完成率', 'Post-tool-failure Trial Completion'],
}

export default function RunsView({ initialRuns }: { initialRuns: Json[] }) {
  const [runs, setRuns] = useState(initialRuns)
  const [runQuery, setRunQuery] = useState('')
  const [runOffset, setRunOffset] = useState(0)
  const [runTotal, setRunTotal] = useState(initialRuns.length)
  const [selectedId, setSelectedId] = useState<string>(String(initialRuns[0]?.id || ''))
  const [view, setView] = useState<RunView>('metrics')
  const [items, setItems] = useState<Json[]>([])
  const [windowStart, setWindowStart] = useState(0)
  const [totalItems, setTotalItems] = useState(0)
  const [activeIndex, setActiveIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showFull, setShowFull] = useState(false)
  const [metricReport, setMetricReport] = useState<Json | undefined>()
  const [analyzing, setAnalyzing] = useState(false)
  const [replaying, setReplaying] = useState(false)
  const [traceFilters, setTraceFilters] = useState<string[]>([])
  const [traceScope, setTraceScope] = useState<{ caseId?: string; trialIndex?: number }>({})
  const loadSequence = useRef(0)
  const selectedRun = runs.find((run) => run.id === selectedId)
  const activeRun = ['starting', 'running', 'queued'].includes(String(selectedRun?.status?.status || ''))

  const requestView = async (runId: string, selectedView: RunView, start: number) => {
    if (selectedView === 'metrics') {
      const value = await api.metricTrials(runId, start, TRIAL_WINDOW_SIZE)
      return { items: value.trials || [], start: value.start_trial || 0, next: value.next_trial || 0, total: value.total_trials || 0 }
    }
    if (selectedView === 'results') {
      const value = await api.results(runId, start, TRIAL_WINDOW_SIZE)
      return { items: value.trials || [], start: value.start_trial || 0, next: value.next_trial || 0, total: value.total_trials || 0 }
    }
    if (selectedView === 'trace') {
      const value = await api.executionTrace(runId, start, TRACE_WINDOW_SIZE, traceScope.caseId || '', traceScope.trialIndex, traceFilters)
      return { items: value.nodes || [], start: value.start_node || 0, next: value.next_node || 0, total: value.total_nodes || 0 }
    }
    const value = selectedView === 'events'
      ? await api.runEvents(runId, start, EVENT_WINDOW_SIZE)
      : await api.evidence(runId, start, EVENT_WINDOW_SIZE)
    return { items: value.events || [], start: value.start_line || 0, next: value.next_line || 0, total: value.total_lines || 0 }
  }

  const loadView = async (runId: string, selectedView: RunView, start = 0, preserveSelection = false) => {
    if (!runId) return
    const sequence = ++loadSequence.current
    setLoading(true)
    setError('')
    try {
      const value = await requestView(runId, selectedView, start)
      if (sequence !== loadSequence.current) return
      setItems(value.items)
      setWindowStart(value.start)
      setTotalItems(value.total)
      if (!preserveSelection) setActiveIndex(value.total ? value.start : 0)
      else setActiveIndex((current) => Math.min(current, Math.max(0, value.total - 1)))
    } catch (cause) {
      if (sequence !== loadSequence.current) return
      setItems([])
      setTotalItems(0)
      setActiveIndex(0)
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      if (sequence === loadSequence.current) setLoading(false)
    }
  }

  const goToItem = async (targetIndex: number) => {
    if (!selectedId || !totalItems) return
    const target = Math.max(0, Math.min(targetIndex, totalItems - 1))
    if (target >= windowStart && target < windowStart + items.length) {
      setActiveIndex(target)
      return
    }
    const pageSize = view === 'trace'
      ? TRACE_WINDOW_SIZE
      : view === 'metrics' || view === 'results'
        ? TRIAL_WINDOW_SIZE
        : EVENT_WINDOW_SIZE
    const start = Math.max(0, Math.min(target - Math.floor(pageSize / 2), Math.max(0, totalItems - pageSize)))
    await loadView(selectedId, view, start)
    setActiveIndex(target)
  }

  useEffect(() => { void loadView(selectedId, view) }, [selectedId, view, traceFilters.join(','), traceScope.caseId, traceScope.trialIndex])
  useEffect(() => setShowFull(false), [selectedId, view, activeIndex])
  useEffect(() => {
    setMetricReport(undefined)
    if (!selectedId || !selectedRun?.has_metric_observations) return
    api.metricEvaluation(selectedId).then(setMetricReport).catch(() => setMetricReport(undefined))
  }, [selectedId, selectedRun?.has_metric_observations])
  useEffect(() => {
    if (!activeRun || !selectedId) return
    const timer = window.setInterval(() => { void loadView(selectedId, view, windowStart, true) }, 3000)
    return () => window.clearInterval(timer)
  }, [activeRun, selectedId, view, windowStart, traceFilters.join(','), traceScope.caseId, traceScope.trialIndex])
  useEffect(() => {
    if (!replaying || !totalItems) return
    if (activeIndex >= totalItems - 1) {
      setReplaying(false)
      return
    }
    const timer = window.setTimeout(() => { void goToItem(activeIndex + 1) }, 700)
    return () => window.clearTimeout(timer)
  }, [activeIndex, replaying, totalItems])

  const refreshRuns = async (query = runQuery, offset = runOffset) => {
    const value = await api.runPage(query, offset, RUN_PAGE_SIZE)
    const nextRuns = Array.isArray(value.items) ? value.items : []
    setRuns(nextRuns)
    setRunOffset(Number(value.offset || 0))
    setRunTotal(Number(value.total || 0))
    setSelectedId((current) => nextRuns.some((run: Json) => run.id === current) ? current : String(nextRuns[0]?.id || ''))
  }

  useEffect(() => {
    const timer = window.setTimeout(() => { void refreshRuns(runQuery, 0) }, 250)
    return () => window.clearTimeout(timer)
  }, [runQuery])

  const analyzeRun = async () => {
    if (!selectedId) return
    setAnalyzing(true)
    setError('')
    try {
      const result = await api.analyzeRun(selectedId)
      setMetricReport(result.metric_evaluation)
      await refreshRuns()
      await loadView(selectedId, 'metrics')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setAnalyzing(false)
    }
  }

  const localIndex = activeIndex - windowStart
  const activeItem = localIndex >= 0 && localIndex < items.length ? items[localIndex] : undefined
  const roles = useMemo(() => roleLanes(items), [items])
  const activeText = itemContent(activeItem)
  const activeTextLong = activeText.length > 3000

  const openTrialTrace = (trial: Json) => {
    setTraceScope({ caseId: String(trial.case_id), trialIndex: Number(trial.trial_index || 0) })
    setView('trace')
  }

  return <section className="runs-view">
    <aside className="runs-list">
      <div className="panel-heading"><h2><Term chinese="运行记录" english="Runs" /></h2><button className="icon-button" title="刷新运行记录" onClick={() => { void refreshRuns() }}><RefreshCw size={16} /></button></div>
      <label className="runs-search"><Search size={14} /><input value={runQuery} onChange={(event) => setRunQuery(event.target.value)} placeholder="搜索 task、状态或路径" /></label>
      <div className="runs-list-items">
        {runs.map((run) => <button key={run.id} className={run.id === selectedId ? 'selected' : ''} onClick={() => setSelectedId(run.id)}>
          <span className={`run-dot status-${run.status?.status || 'unknown'}`} />
          <span className="run-list-main"><strong>{run.status?.task || run.relative_path.split('/').at(-1)}</strong><small title={run.relative_path}>{run.relative_path}</small></span>
          <em>{run.status?.successful_trials || 0}/{run.status?.expected_trials || '?'}</em>
        </button>)}
        {!runs.length && <small className="runs-empty">没有匹配的运行记录</small>}
      </div>
      <div className="runs-pagination"><button className="icon-button" title="上一页" disabled={runOffset <= 0} onClick={() => { void refreshRuns(runQuery, Math.max(0, runOffset - RUN_PAGE_SIZE)) }}><ChevronLeft size={15} /></button><span>{runTotal ? `${runOffset + 1}-${Math.min(runOffset + runs.length, runTotal)} / ${runTotal}` : '0 / 0'}</span><button className="icon-button" title="下一页" disabled={runOffset + runs.length >= runTotal} onClick={() => { void refreshRuns(runQuery, runOffset + RUN_PAGE_SIZE) }}><ChevronRight size={15} /></button></div>
    </aside>

    <div className={`run-monitor ${view === 'metrics' || view === 'results' ? 'metrics-mode' : ''} ${view === 'trace' ? 'has-trace-toolbar' : ''}`}>
      <div className="monitor-header">
        <div><h1>{selectedRun?.status?.task || '选择运行记录'}</h1><span>{selectedRun?.run_dir}</span></div>
        <div className="monitor-actions">
          <div className="segmented run-view-tabs" aria-label="运行记录视图">
            <button className={view === 'metrics' ? 'active' : ''} onClick={() => setView('metrics')}><Term chinese="四类指标" english="Metrics" /></button>
            <button className={view === 'results' ? 'active' : ''} onClick={() => setView('results')}><Term chinese="作答结果" english="Result Projection" /></button>
            <button className={view === 'trace' ? 'active' : ''} onClick={() => setView('trace')}><Term chinese="执行轨迹" english="Execution Trace" /></button>
            <button className={view === 'events' ? 'active' : ''} title="唯一、无损的原始事件日志" onClick={() => setView('events')}><Term chinese="原始事件" english="EventLog" /></button>
            <button className={view === 'evidence' ? 'active' : ''} title="由 EventLog 派生的指标证据" onClick={() => setView('evidence')}><Term chinese="分析证据" english="Evidence" /></button>
          </div>
        </div>
      </div>

      <div className="run-summary-strip">
        <Metric label="运行状态" englishLabel="Run Status" value={runStatusLabel(selectedRun?.status?.status)} />
        <Metric label="Trial 完成度" englishLabel="Trial Progress" value={`${selectedRun?.status?.successful_trials || 0}/${selectedRun?.status?.expected_trials || '—'}`} />
        <Metric label="官方任务得分" englishLabel="Official Score" value={formatMetricNumber(selectedRun?.metrics?.score_mean ?? selectedRun?.metrics?.accuracy)} />
        <Metric label="证据完整性" englishLabel="Evidence Coverage" value={coverageStatusLabel(selectedRun?.evidence_coverage?.overall_status ?? selectedRun?.metrics?.evidence_coverage?.overall_status)} />
        <Metric label="已计算观测" englishLabel="Measured" value={metricEvaluationStatusCount(selectedRun, 'measured')} />
        <Metric label="不适用观测" englishLabel="Not Applicable" value={metricEvaluationStatusCount(selectedRun, 'not_applicable')} />
        <Metric label="缺证据观测" englishLabel="Missing Evidence" value={metricEvaluationStatusCount(selectedRun, 'missing_evidence')} />
        <Metric label="指标计算错误" englishLabel="Evaluator Errors" value={metricEvaluationStatusCount(selectedRun, 'evaluator_error')} />
      </div>

      {view === 'trace' && <div className="trace-toolbar">
        <span><Term chinese="轨迹筛选" english="Trace Filters" /></span>
        {[
          ['messages', '消息'], ['model_calls', '模型调用'], ['tools', '工具'], ['errors', '错误'],
        ].map(([id, label]) => <button key={id} className={traceFilters.includes(id) ? 'active' : ''} onClick={() => setTraceFilters((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])}>{label}</button>)}
        {(traceScope.caseId !== undefined) && <button className="secondary" onClick={() => setTraceScope({})}>清除 Trial 范围 · {traceScope.caseId} · Trial {Number(traceScope.trialIndex || 0) + 1}</button>}
        {activeRun && <small>运行中，每 3 秒重新投影</small>}
      </div>}

      {view === 'metrics' && <MetricBoard loading={loading} error={error} report={metricReport} trial={activeItem} analyzing={analyzing} analyzeDisabled={!selectedId || analyzing || activeRun} onAnalyze={() => { void analyzeRun() }} />}
      {view === 'results' && <ResultBoard loading={loading} error={error} trials={items} onOpenTrace={openTrialTrace} />}
      {(view === 'trace' || view === 'events' || view === 'evidence') && <TraceBoard loading={loading} error={error} items={items} activeItem={activeItem} roles={roles} showFull={showFull} setShowFull={setShowFull} view={view} activeText={activeText} activeTextLong={activeTextLong} activeIndex={activeIndex} windowStart={windowStart} setActiveIndex={setActiveIndex} />}

      <Navigator view={view} loading={loading} activeIndex={activeIndex} total={totalItems} activeItem={activeItem} onMove={goToItem} replaying={replaying} setReplaying={setReplaying} />
    </div>
  </section>
}

function MetricBoard({ loading, error, report, trial, analyzing, analyzeDisabled, onAnalyze }: { loading: boolean; error: string; report?: Json; trial?: Json; analyzing: boolean; analyzeDisabled: boolean; onAnalyze: () => void }) {
  return <div className="metric-board">
    <header className="metric-board-header"><div><Term chinese="四类标准化指标" english="Four-Dimension Standardized Metrics" /><small>每个指标一行，横向对照整次 Run 与当前 Trial</small></div><button className="secondary" disabled={analyzeDisabled} onClick={onAnalyze}><BarChart3 size={15} />{analyzing ? '分析中…' : report?.metrics?.length ? '重新分析' : '补充分析'}</button></header>
    {loading ? <Empty text="正在读取 Trial 指标…" /> : error ? <Empty text={error} error /> : report?.metrics?.length ? <MetricComparisonTable report={report} trial={trial} /> : <div className="metric-empty-state metric-board-content"><BarChart3 size={24} /><strong>这次 Run 尚未生成离线指标</strong><button className="secondary" onClick={onAnalyze}>补充分析</button></div>}
  </div>
}

function ResultBoard({ loading, error, trials, onOpenTrace }: { loading: boolean; error: string; trials: Json[]; onOpenTrace: (trial: Json) => void }) {
  return <div className="metric-board result-board">
    <header className="metric-board-header"><div><Term chinese="作答结果" english="Result Projection" /></div><small>每个 Trial 一行；Prediction 与 Evaluation 在读取时关联</small></header>
    {loading ? <Empty text="正在生成作答结果…" /> : error ? <Empty text={error} error /> : <div className="metric-comparison-wrap metric-board-content"><table className="result-table"><thead><tr><th>Case / Trial</th><th>推理状态</th><th>Prediction</th><th>评分</th><th>模型调用</th><th>输入 Token</th><th>输出 Token</th><th>模型延迟</th><th /></tr></thead><tbody>
      {trials.map((trial) => <tr key={`${trial.case_id}:${trial.trial_index}`}><td><strong>{String(trial.case_id)}</strong><small>Trial {Number(trial.trial_index || 0) + 1} · dataset {trial.dataset_index ?? '—'}</small></td><td><Status value={trial.trial_status} /><small>{trial.evaluation_status}</small></td><td className="result-answer" title={String(trial.prediction || '')}>{compact(String(trial.prediction || '—'), 260)}</td><td>{formatMetricNumber(trial.score)}</td><td>{trial.model_call_count ?? 0}</td><td>{trial.input_tokens ?? 0}</td><td>{trial.output_tokens ?? 0}</td><td>{formatSeconds(trial.model_latency_s)}</td><td><button className="secondary" onClick={() => onOpenTrace(trial)}>查看轨迹</button></td></tr>)}
      {!trials.length && <tr><td colSpan={9}>尚未观察到 Trial</td></tr>}
    </tbody></table></div>}
  </div>
}

function TraceBoard({ loading, error, items, activeItem, roles, showFull, setShowFull, view, activeText, activeTextLong, activeIndex, windowStart, setActiveIndex }: any) {
  return <div className="trace-board">
    <div className="communication-board">
      <div className="role-lanes">{roles.map((role: string) => <div key={role} className={role === displayRole(activeItem) ? 'active' : ''}><span /><strong>{role}</strong></div>)}{!roles.length && <div className="empty-state">暂无角色事件</div>}</div>
      <div className="active-event"><div className="event-title"><Activity size={16} /><strong>{displayRole(activeItem)}</strong><span className={`event-kind kind-${eventCategory(eventType(activeItem))}`}>{eventType(activeItem) || 'idle'}</span><div className="event-title-actions">{activeTextLong && <button className="icon-button" title={showFull ? '折叠' : '展开完整内容'} onClick={() => setShowFull(!showFull)}>{showFull ? <Minimize2 size={15} /> : <Maximize2 size={15} />}</button>}</div></div><pre>{loading ? '正在读取…' : error || formatEventText(activeText, showFull)}</pre><div className="event-meta"><span>{view === 'trace' ? 'node' : 'event'} {activeIndex + 1}</span><span>status {activeItem?.status || '—'}</span><span>duration {formatSeconds(activeItem?.duration_s)}</span><span>Trial {Number(activeItem?.trial_index || 0) + 1}</span></div></div>
    </div>
    <div className="event-timeline">{items.map((item: Json, index: number) => { const globalIndex = windowStart + index; const kind = eventType(item); return <article key={`${item.node_id || item.event_id || globalIndex}`} className={`event-row ${globalIndex === activeIndex ? 'active' : ''}`} onClick={() => setActiveIndex(globalIndex)}><time>{String(item.timestamp_start_utc || item.timestamp_utc || '').slice(11, 19) || `#${globalIndex + 1}`}</time><span className={`event-kind kind-${eventCategory(kind)}`}>{kind || 'event'}</span><strong>{displayRole(item)}</strong><p>{formatTimelineContent(item)}</p></article> })}</div>
  </div>
}

function Navigator({ view, loading, activeIndex, total, activeItem, onMove, replaying, setReplaying }: any) {
  const label = view === 'metrics' ? 'Trial 指标' : view === 'results' ? 'Trial 结果' : view === 'trace' ? '轨迹节点' : '事件'
  const canReplay = ['trace', 'events', 'evidence'].includes(view)
  return <div className={`prediction-control ${canReplay ? 'has-replay' : ''}`}>{canReplay && <button className="secondary step-button" disabled={loading || !total} onClick={() => { setReplaying(false); void onMove(0); setReplaying(total > 1) }}><RotateCcw size={15} />从头回放</button>}<button className="secondary step-button" disabled={loading || !total || activeIndex <= 0} onClick={() => { setReplaying(false); void onMove(activeIndex - 1) }}><ChevronLeft size={15} />上一个</button><input aria-label="当前位置" type="range" min={0} max={Math.max(0, total - 1)} value={Math.min(activeIndex, Math.max(0, total - 1))} disabled={!total} onChange={(event) => { setReplaying(false); void onMove(Number(event.target.value)) }} /><span><strong>{label} {total ? activeIndex + 1 : 0} / {total}</strong><small>{activeItem?.case_id ? `Case ${activeItem.case_id} · Trial ${Number(activeItem.trial_index || 0) + 1}` : '暂无 Trial 范围'}</small></span><button className="secondary step-button" disabled={loading || !total || activeIndex >= total - 1} onClick={() => { setReplaying(false); void onMove(activeIndex + 1) }}>{replaying ? '暂停' : '下一个'}<ChevronRight size={15} /></button></div>
}

function MetricComparisonTable({ trial, report }: { trial?: Json; report?: Json }) {
  const observations = Array.isArray(trial?.observations) ? trial.observations : []
  return <div className="metric-comparison-wrap metric-board-content"><table className="metric-comparison-table"><colgroup><col className="metric-col" /><col className="unit-col" /><col className="run-status-col" /><col className="stat-col" /><col className="stat-col" /><col className="stat-col" /><col className="count-col" /><col className="prediction-status-col" /><col className="prediction-value-col" /></colgroup><thead><tr className="metric-scope-row"><th rowSpan={2}><Term chinese="具体指标" english="Metric" /></th><th rowSpan={2}><Term chinese="单位" english="Unit" /></th><th className="scope-divider" colSpan={5}><Term chinese="整次运行" english="Run" /></th><th className="scope-divider" colSpan={2}><Term chinese="当前 Trial" english="Trial" />{trial && <small>Case {trial.case_id} · Trial {Number(trial.trial_index || 0) + 1}</small>}</th></tr><tr><th className="scope-divider">状态</th><th>均值</th><th>中位数</th><th>P95</th><th>观测数</th><th className="scope-divider">状态</th><th>数值</th></tr></thead>
    {metricGroups(report, observations).map((group) => <tbody key={group.id}><tr className={`metric-category-row category-${group.id}`}><th colSpan={9}><Term chinese={group.label} english={group.englishLabel} /><span>{group.description}</span></th></tr>{group.metrics.map((metric: Json) => { const observation = observations.find((item: Json) => item.metric_id === metric.metric_id); return <tr key={String(metric.metric_id)} title={String(observation?.reason || '')}><td className="metric-identity"><strong>{metricLabel(metric.metric_id)[0]}</strong><small>{metricLabel(metric.metric_id)[1]}</small></td><td>{unitLabel(metric.unit)}</td><td className="scope-divider"><MetricStatuses counts={metric.status_counts} /></td><td>{formatMetricNumber(metric.measured_mean)}</td><td>{formatMetricNumber(metric.measured_p50)}</td><td>{formatMetricNumber(metric.measured_p95)}</td><td>{metric.observation_count ?? '—'}</td><td className="scope-divider">{observation ? <MetricStatusBadge status={observation.status} /> : <MetricStatusBadge status="missing_evidence" />}</td><td><strong>{formatMetricNumber(observation?.value)}</strong></td></tr> })}</tbody>)}
  </table></div>
}

function Term({ chinese, english }: { chinese: string; english: string }) { return <span className="bilingual-term"><strong>{chinese}</strong><small>{english}</small></span> }
function Metric({ label, englishLabel, value }: { label: string; englishLabel: string; value: any }) { return <div><span>{label}<small>{englishLabel}</small></span><strong>{String(value)}</strong></div> }
function Empty({ text, error = false }: { text: string; error?: boolean }) { return <div className={`metric-empty-state metric-board-content ${error ? 'error-text' : ''}`}>{text}</div> }
function Status({ value }: { value: any }) { return <span className={`metric-status status-${String(value || 'unknown')}`}><strong>{String(value || 'unknown')}</strong></span> }
function MetricStatuses({ counts }: { counts?: Json }) { return !counts || !Object.keys(counts).length ? <span>—</span> : <div className="metric-statuses">{Object.entries(counts).map(([status, count]) => <MetricStatusBadge key={status} status={status} count={count} />)}</div> }
function MetricStatusBadge({ status, count }: { status: any; count?: any }) { const value = String(status || 'missing_evidence'); return <span className={`metric-status status-${value}`}><strong>{metricStatusLabel(value)}{count === undefined ? '' : ` ${String(count)}`}</strong><small>{metricStatusEnglishLabel(value)}</small></span> }

function metricGroups(report?: Json, fallback: Json[] = []) { const metrics = Array.isArray(report?.metrics) && report.metrics.length ? report.metrics : fallback; return METRIC_CATEGORIES.map((category) => ({ ...category, metrics: metrics.filter((metric: Json) => String(metric.category || metric.metric_id || '').split('.')[0] === category.id) })) }
function metricLabel(id: any): [string, string] { return METRIC_LABELS[String(id || '')] || [String(id || '指标观测'), String(id || 'Metric')] }
function metricStatusLabel(status: any) { return ({ measured: '已计算', not_applicable: '不适用', missing_evidence: '缺少证据', evaluator_error: '计算错误', unsupported: '不支持' } as Record<string, string>)[String(status)] || String(status || '未知') }
function metricStatusEnglishLabel(status: any) { return ({ measured: 'Measured', not_applicable: 'N/A', missing_evidence: 'Missing Evidence', evaluator_error: 'Evaluator Error', unsupported: 'Unsupported' } as Record<string, string>)[String(status)] || String(status || 'Unknown') }
function runStatusLabel(status: any) { return ({ completed: '已完成', complete: '已完成', complete_with_errors: '完成但有错误', running: '运行中', starting: '启动中', queued: '排队中', draining: '优雅停止中', paused: '已暂停', stopped: '已停止', failed: '运行失败' } as Record<string, string>)[String(status)] || String(status || '—') }
function coverageStatusLabel(status: any) { return ({ complete: '完整', partial: '部分缺失', missing: '尚未生成', not_analyzed: '尚未分析' } as Record<string, string>)[String(status)] || String(status || '尚未分析') }
function unitLabel(unit: any) { return ({ benchmark_score: '任务分数', ratio: '比例', normalized_index: '归一化指数', tokens: 'Token', seconds: '秒' } as Record<string, string>)[String(unit)] || String(unit || '—') }
function formatMetricNumber(value: any) { if (typeof value !== 'number') return '—'; return Number.isInteger(value) ? String(value) : value.toLocaleString(undefined, { maximumFractionDigits: 4 }) }
function formatSeconds(value: any) { return typeof value === 'number' ? `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} s` : '—' }
function metricEvaluationStatusCount(run: Json | undefined, status: string) { const summary = run?.metric_evaluation ?? run?.metrics?.metric_evaluation; return summary?.status_counts ? summary.status_counts[status] ?? 0 : '未分析' }
function eventType(item?: Json) { return String(item?.event_type || item?.metric_id || '') }
function rawRole(item?: Json) { const payload = item?.payload || item?.events?.at?.(-1)?.payload || {}; return String(item?.actor || payload.role || payload.source || payload.controller || item?.source || item?.role || 'Runtime') }
function displayRole(item?: Json) { const role = rawRole(item); return role === 'MagenticOneOrchestrator' ? 'Orchestrator' : ['None', 'undefined', 'null'].includes(role) ? 'Runtime' : role }
function roleLanes(items: Json[]) {
  const systemActors = new Set(['Runtime', 'runtime', 'user', 'system', 'autogen', 'langgraph', 'crewai'])
  return Array.from(new Set(items.flatMap((item) => {
    const kind = eventType(item)
    if (!(kind.startsWith('model_call') || kind.startsWith('agent.message') || kind.includes('selection'))) return []
    const role = displayRole(item)
    return systemActors.has(role) ? [] : [role]
  })))
}
function eventCategory(kind: string) { if (!kind) return 'neutral'; if (kind.includes('failed') || kind.includes('error')) return 'error'; if (kind.startsWith('model_call')) return kind.endsWith('started') ? 'model-start' : 'model-end'; if (kind.startsWith('trial') || kind.startsWith('attempt')) return kind.endsWith('started') ? 'case-start' : 'case-end'; if (kind.startsWith('run') || kind.startsWith('runtime') || kind.startsWith('group_chat')) return kind.endsWith('started') ? 'runtime-start' : 'runtime-end'; if (kind.startsWith('agent.message')) return 'message'; if (kind.includes('tool') || kind.includes('web')) return 'tool'; return 'selection' }
function itemContent(item?: Json) { if (!item) return '等待记录'; return JSON.stringify(item, null, 2) }
function formatEventText(text: string, full: boolean, limit = 3000) { return full || text.length <= limit ? text : `${text.slice(0, limit / 2)}\n…\n${text.slice(-limit / 2)}` }
function formatTimelineContent(item: Json) { const value = item.summary || item.payload?.content || item.payload?.final_content || item.content || item.payload?.error_message; return value ? compact(typeof value === 'string' ? value : JSON.stringify(value), 220) : [item.case_id ? `case=${item.case_id}` : '', item.status ? `status=${item.status}` : ''].filter(Boolean).join(' · ') || '结构化运行事实' }
function compact(text: string, limit: number) { return text.length <= limit ? text : `${text.slice(0, Math.floor(limit / 2) - 2)} … ${text.slice(-Math.floor(limit / 2) + 2)}` }
