import { Check, CircleAlert, CirclePlay, Clipboard, Folder, Infinity, ListMinus, ListPlus, Pause, Play, Plus, RefreshCw, RotateCcw, Save, Search, Square, Trash2, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import ResourcePressure from './ResourcePressure'
import SpecSelector from './SpecSelector'
import { deriveCoordinationSummary, memberNodeIds } from './teamGraph'
import type { Bootstrap, Json, PlanResponse } from './types'

interface Props {
  bootstrap: Bootstrap
  environment: Json
  notify: (message: string, tone?: 'success' | 'error') => void
  onRefresh: () => void
}

const statusReady = (value: Json) => Boolean(value.available)

const slug = (value: unknown, fallback: string) => {
  const normalized = String(value || '').replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^[.-]+|[.-]+$/g, '')
  return normalized.slice(0, 64) || fallback
}

const utcPathStamp = () => new Date().toISOString().replace(/[-:]/g, '').replace('.', '').replace(/Z$/, 'Z')

const randomIdSuffix = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID().replaceAll('-', '').slice(0, 8)
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`.slice(-8)
}

const defaultExperimentInstanceId = (specId: unknown) =>
  `${slug(specId, 'experiment')}-run-${randomIdSuffix()}`

const TRIAL_SEQUENCE_COLORS = [
  '#2f7d6d', '#396fa4', '#b06c2f', '#845e9f', '#b04e62', '#667c35',
  '#7d6045', '#4c7d8f', '#a35f91', '#887629', '#4f749f', '#9a5c3e',
]
const MAX_RENDERED_TRIALS_PER_SEQUENCE = 240

const sequenceAlias = (index: number) => {
  let value = index + 1
  let result = ''
  while (value > 0) {
    value -= 1
    result = String.fromCharCode(65 + (value % 26)) + result
    value = Math.floor(value / 26)
  }
  return result
}

const duration = (seconds: unknown) => {
  if (seconds === null || seconds === undefined || seconds === '') return '—'
  const value = Math.max(0, Number(seconds || 0))
  if (!Number.isFinite(value)) return '—'
  const hours = Math.floor(value / 3600)
  const minutes = Math.floor((value % 3600) / 60)
  const secs = Math.floor(value % 60)
  return hours ? `${hours}h ${minutes}m ${secs}s` : minutes ? `${minutes}m ${secs}s` : `${secs}s`
}

export default function ExperimentView({ bootstrap, environment, notify, onRefresh }: Props) {
  const defaultRunsRoot = bootstrap.paths?.runs_root || 'runs/benchmarks'
  const initialSpec = bootstrap.experiments?.specs?.[0] || bootstrap.default_experiment_spec
  const [catalog, setCatalog] = useState<Json>(bootstrap.experiments || { specs: [], instances: [], queue: {} })
  const [draft, setDraft] = useState<Json>(() => ({
    ...structuredClone(initialSpec),
    environment: {
      ...structuredClone(initialSpec.environment || {}),
      python: environment.python,
      conda_env: environment.conda_env || '',
      conda_sh: environment.conda_sh || '',
    },
  }))
  const [benchmarkInstanceId, setBenchmarkInstanceId] = useState('')
  const [teamInstanceId, setTeamInstanceId] = useState('')
  const [instanceId, setInstanceId] = useState(() => defaultExperimentInstanceId(initialSpec.id))
  const [runDir, setRunDir] = useState('')
  const [runStamp, setRunStamp] = useState(utcPathStamp)
  const [priority, setPriority] = useState(100)
  const [segmentCaseLimit, setSegmentCaseLimit] = useState<number | null>(null)
  const [caseCompletionTarget, setCaseCompletionTarget] = useState<number | null>(null)
  const [maxParallelInstances, setMaxParallelInstances] = useState(
    Number(bootstrap.experiments?.queue?.max_parallel_instances || 16),
  )
  const [maxRunningTrials, setMaxRunningTrials] = useState(
    Number(bootstrap.experiments?.queue?.max_running_trials || 64),
  )
  const [launcherType, setLauncherType] = useState('subprocess')
  const [tmuxSession, setTmuxSession] = useState('lychee-eval')
  const [tmuxWindow, setTmuxWindow] = useState('benchmark')
  const [plan, setPlan] = useState<PlanResponse | null>(null)
  const [planInstanceId, setPlanInstanceId] = useState('')
  const [busy, setBusy] = useState(false)
  const [workspace, setWorkspace] = useState<'spec' | 'assemble' | 'runs'>('runs')
  const [instanceFilter, setInstanceFilter] = useState<'active' | 'ready' | 'attention' | 'finished' | 'all'>('active')
  const [instanceSearch, setInstanceSearch] = useState('')
  const pollInFlight = useRef(false)
  const experimentScheduling: Json[] = catalog.queue?.experiment_scheduling || []
  const notifyError = (cause: unknown) => notify(
    cause instanceof Error ? cause.message : String(cause),
    'error',
  )

  const benchmarkSpecs: Json[] = bootstrap.benchmark_registry?.specs || []
  const benchmarkInstances: Json[] = bootstrap.benchmark_registry?.instances || []
  const teamSpecs: Json[] = bootstrap.teams?.specs || []
  const teamInstances: Json[] = bootstrap.team_instances?.instances || []
  const benchmarkSpec = benchmarkSpecs.find((item) => item.id === draft.benchmark?.benchmark_spec_id)
  const teamSpec = teamSpecs.find((item) => item.id === draft.team_spec_id)
  const sandboxProfiles = Object.entries(benchmarkSpec?.sandbox_profiles || {}) as [string, Json][]
  const selectedSandboxProfile = sandboxProfiles.find(
    ([, profile]) => profile.docker_image === draft.runtime?.docker_image,
  )
  const coordinationSummary = deriveCoordinationSummary(teamSpec)
  const participantCount = Math.max(
    1,
    Number(memberNodeIds(teamSpec).length || 1),
  )
  const isFixedSchedule = participantCount === 1 || coordinationSummary.isDependencyChain
  const effectiveMaxTurns = isFixedSchedule
    ? participantCount * Math.max(1, Number(draft.runtime?.max_rounds || 4))
    : Math.max(1, Number(draft.runtime?.max_turns || 20))
  const dynamicScheduleDescription = coordinationSummary.operationNames.includes('plan')
    && coordinationSummary.operationNames.includes('delegate')
    ? '绑定规划、委派与进度操作的 Node 动态组织工作'
    : coordinationSummary.operationNames.includes('select_next')
      ? '绑定 select_next 操作的 Node 动态选择下一执行者'
      : coordinationSummary.hasHandoff
        ? '执行 Node 通过显式 handoff Relation 移交工作'
        : 'RuntimeAdapter 按显式 Control Relation 激活执行 Node'
  const modelCallsUnlimited = draft.runtime?.max_model_calls_per_case == null
  const caseWallTimeUnlimited = draft.runtime?.max_case_wall_time_s == null
  const benchmarkNetworkDefaults: Json = {
    mode: 'direct',
    access: 'optional',
    no_proxy: '127.0.0.1,localhost,::1',
    targets: { downloads: false, web_surfer: false, code_executor: false, model_backend: false },
    docker_bridge_host: '172.17.0.1',
    container_proxy_port: 17897,
    probe_url: 'https://www.google.com/generate_204',
    ...(benchmarkSpec?.network_defaults || {}),
  }
  const enabledDefaultNetworkTargets = Object.entries(benchmarkNetworkDefaults.targets || {})
    .filter(([, enabled]) => Boolean(enabled))
    .map(([name]) => name)
  const launcherLabels: Record<string, string> = {
    subprocess: 'Subprocess',
    new_tmux_session: '新建 tmux 会话',
    existing_tmux_session: '现有 tmux 会话',
  }
  const taskContract = benchmarkSpec?.task_contracts?.[draft.benchmark?.runnable_task] || {}
  const scoring = taskContract.scoring || {}
  const scoringProfiles = Object.keys(scoring.profiles || {})
  const candidateBenchmarks = useMemo(
    () => benchmarkInstances.filter((item) => item.benchmark_spec_id === benchmarkSpec?.id),
    [benchmarkInstances, benchmarkSpec?.id],
  )
  const candidateTeams = useMemo(
    () => teamInstances.filter((item) => item.team_spec_id === teamSpec?.id),
    [teamInstances, teamSpec?.id],
  )
  const selectedBenchmarkId = benchmarkInstanceId
    || candidateBenchmarks.find(statusReady)?.id
    || candidateBenchmarks[0]?.id
    || ''
  const selectedTeamId = teamInstanceId
    || candidateTeams.find(statusReady)?.id
    || candidateTeams[0]?.id
    || ''
  const selectedBenchmark = candidateBenchmarks.find((item) => item.id === selectedBenchmarkId)
  const selectedTeam = candidateTeams.find((item) => item.id === selectedTeamId)
  const automaticRunDir = useMemo(() => {
    const benchmarkId = slug(draft.benchmark?.benchmark_spec_id, 'benchmark')
    const task = slug(draft.benchmark?.runnable_task, 'task')
    const teamId = slug(teamSpec?.id || draft.team_spec_id, 'studio-team')
    const method = slug(draft.runtime?.method || 'none', 'none')
    const experimentId = slug(draft.id, 'experiment')
    const root = String(draft.environment?.runs_root || defaultRunsRoot).replace(/\/+$/, '')
    return `${root}/${benchmarkId}/${task}/${teamId}/${method}/${experimentId}/${runStamp}`
  }, [defaultRunsRoot, draft.benchmark?.benchmark_spec_id, draft.benchmark?.runnable_task, draft.environment?.runs_root, draft.id, draft.runtime?.method, draft.team_spec_id, runStamp, teamSpec?.id])
  const effectiveRunDir = runDir || automaticRunDir
  const queuedCount = (catalog.queue?.queued_instance_ids || []).length
  const runningCount = (catalog.queue?.running_instance_ids || []).length
  const trialSequencePresentation = useMemo(() => {
    const colorByInstance = new Map<string, string>()
    const aliasByInstance = new Map<string, string>()
    experimentScheduling.forEach((item, index) => {
      const instanceId = String(item.instance_id)
      colorByInstance.set(instanceId, TRIAL_SEQUENCE_COLORS[index % TRIAL_SEQUENCE_COLORS.length])
      aliasByInstance.set(instanceId, sequenceAlias(index))
    })
    const runningCounters = new Map<string, number>()
    const runningSource: Json[] = catalog.queue?.running_trial_sequence || []
    const running = runningSource.slice(0, MAX_RENDERED_TRIALS_PER_SEQUENCE).map((item) => {
      const instanceId = String(item.instance_id)
      const ordinal = (runningCounters.get(instanceId) || 0) + 1
      runningCounters.set(instanceId, ordinal)
      return {
        ...item,
        color: colorByInstance.get(instanceId) || '#71817d',
        label: `${aliasByInstance.get(instanceId) || '?'}${item.case_order ?? ordinal}`,
      }
    })
    const waiting: Json[] = []
    for (const segment of (catalog.queue?.waiting_trial_segments || [])) {
      const instanceId = String(segment.instance_id)
      const count = Math.max(0, Number(segment.count || 0))
      for (let index = 0; index < count && waiting.length < MAX_RENDERED_TRIALS_PER_SEQUENCE; index += 1) {
        waiting.push({
          instance_id: instanceId,
          priority: segment.priority,
          color: colorByInstance.get(instanceId) || '#71817d',
          label: `${aliasByInstance.get(instanceId) || '?'}${index + 1}`,
        })
      }
      if (waiting.length >= MAX_RENDERED_TRIALS_PER_SEQUENCE) break
    }
    const waitingTotal = (catalog.queue?.waiting_trial_segments || []).reduce(
      (sum: number, item: Json) => sum + Math.max(0, Number(item.count || 0)),
      0,
    )
    return {
      aliasByInstance,
      colorByInstance,
      running,
      runningHidden: Math.max(0, runningSource.length - running.length),
      waiting,
      waitingHidden: Math.max(0, waitingTotal - waiting.length),
    }
  }, [catalog.queue?.running_trial_sequence, catalog.queue?.waiting_trial_segments, experimentScheduling])
  const orphanedLaunches: Json[] = catalog.orphans || []
  const admissionPressureEntries = (['vllm', 'api', 'gpu'] as const).map((kind) => ({
    id: `experiment-${kind}`,
    kind,
    pressure: catalog.queue?.pressure?.[kind] || {},
    title: kind === 'vllm' ? 'vLLM 聚合压力' : kind === 'api' ? 'API 聚合压力' : 'GPU 聚合压力',
  }))
  const queueIsActive = Boolean(
    catalog.queue?.running
    || (catalog.instances || []).some((item: Json) => item.status === 'running')
    || orphanedLaunches.length > 0
  )
  const environmentUnavailable = environment.exists === false
  const planCommand = plan?.files['run_all.sh'] || plan?.commands?.subprocess || ''
  const experimentInstances: Json[] = catalog.instances || []
  const instanceGroup = (item: Json) => {
    if (item.configuration_status && item.configuration_status !== 'current') return 'attention'
    if (item.status === 'running' || item.status === 'queued') return 'active'
    if (item.status === 'ready') return 'ready'
    if (item.status === 'completed') return 'finished'
    return 'attention'
  }
  const instanceCounts = experimentInstances.reduce((counts: Record<string, number>, item: Json) => {
    const group = instanceGroup(item)
    counts[group] = (counts[group] || 0) + 1
    counts.all = (counts.all || 0) + 1
    return counts
  }, { active: 0, ready: 0, attention: 0, finished: 0, all: 0 })
  const normalizedInstanceSearch = instanceSearch.trim().toLowerCase()
  const visibleInstances = experimentInstances.filter((item) => {
    if (instanceFilter !== 'all' && instanceGroup(item) !== instanceFilter) return false
    if (!normalizedInstanceSearch) return true
    return [
      item.id,
      item.experiment_spec_id,
      item.benchmark_instance_id,
      item.team_instance_id,
      item.run_dir,
    ].some((value) => String(value || '').toLowerCase().includes(normalizedInstanceSearch))
  })

  useEffect(() => {
    if (!queueIsActive) return undefined
    let cancelled = false
    const poll = async () => {
      if (pollInFlight.current || document.hidden) return
      pollInFlight.current = true
      try {
        const snapshot = await api.experimentDashboard()
        if (!cancelled) setCatalog((current: Json) => ({ ...current, ...snapshot }))
      } catch {
        // The next manual refresh will surface persistent connectivity errors.
      } finally {
        pollInFlight.current = false
      }
    }
    const timer = window.setInterval(() => void poll(), 2500)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [queueIsActive])

  const refresh = async () => {
    const snapshot = await api.experimentDashboard(true)
    setCatalog(snapshot)
    onRefresh()
  }

  const load = (spec: Json) => {
    setDraft({
      ...structuredClone(spec),
      environment: {
        ...structuredClone(spec.environment || {}),
        python: environment.python,
        conda_env: environment.conda_env || '',
        conda_sh: environment.conda_sh || '',
      },
    })
    setBenchmarkInstanceId('')
    setTeamInstanceId('')
    setInstanceId(defaultExperimentInstanceId(spec.id))
    setRunDir('')
    setRunStamp(utcPathStamp())
    setLauncherType('subprocess')
    setTmuxSession('lychee-eval')
    setTmuxWindow(slug(spec.id, 'benchmark'))
    setPlan(null)
    setPlanInstanceId('')
    setSegmentCaseLimit(null)
  }

  const newSpec = () => {
    const next = structuredClone(bootstrap.default_experiment_spec)
    next.id = `experiment-${Date.now().toString(36)}`
    load(next)
  }

  const updateSection = (section: string, patch: Json) =>
    setDraft((current: Json) => ({
      ...current,
      [section]: { ...(current[section] || {}), ...patch },
    }))

  const chooseNetworkMode = (mode: string) => setDraft((current: Json) => {
    const previous = current.network || {}
    const targets = previous.mode === 'benchmark_default'
      ? structuredClone(benchmarkNetworkDefaults.targets || {})
      : structuredClone(previous.targets || {})
    return { ...current, network: { ...previous, mode, ...(mode === 'proxy' ? { targets } : {}) } }
  })

  const patchNetworkTarget = (target: string, enabled: boolean) => updateSection('network', {
    targets: { ...(draft.network?.targets || {}), [target]: enabled },
  })

  const chooseBenchmarkSpec = (id: string) => {
    const spec = benchmarkSpecs.find((item) => item.id === id)
    const task = spec?.runnable_tasks?.[0] || ''
    const defaultProfile = spec?.task_contracts?.[task]?.scoring?.default_profile
    setDraft((current: Json) => ({
      ...current,
      benchmark: {
        ...(current.benchmark || {}),
        benchmark_spec_id: id,
        runnable_task: task,
        ...(defaultProfile ? { scoring_profile: defaultProfile } : {}),
      },
      runtime: {
        ...(current.runtime || {}),
        docker_image: spec?.runtime_defaults?.docker_image || null,
      },
    }))
    setBenchmarkInstanceId('')
  }

  const chooseTask = (task: string) => {
    const defaultProfile = benchmarkSpec?.task_contracts?.[task]?.scoring?.default_profile
    updateSection('benchmark', {
      runnable_task: task,
      ...(defaultProfile ? { scoring_profile: defaultProfile } : {}),
    })
  }

  const save = async () => {
    setBusy(true)
    try {
      const saved = await api.saveExperimentSpec(draft.id, draft)
      setDraft(saved)
      await refresh()
      notify(`ExperimentSpec ${saved.id} 已保存`)
    } catch (cause) {
      notifyError(cause)
    } finally {
      setBusy(false)
    }
  }

  const instantiate = async () => {
    setBusy(true)
    try {
      await api.saveExperimentSpec(draft.id, draft)
      const saved = await api.createExperimentInstance({
        id: instanceId || defaultExperimentInstanceId(draft.id),
        experiment_spec_id: draft.id,
        benchmark_instance_id: selectedBenchmarkId,
        team_instance_id: selectedTeamId,
        run_dir: effectiveRunDir,
        priority,
        next_segment_case_limit: segmentCaseLimit,
        case_completion_target: caseCompletionTarget,
        launcher: {
          type: launcherType,
          ...(
            launcherType === 'new_tmux_session' || launcherType === 'existing_tmux_session'
              ? { tmux_session: tmuxSession, tmux_window: tmuxWindow }
              : {}
          ),
        },
      })
      await refresh()
      setInstanceId(defaultExperimentInstanceId(draft.id))
      setRunDir('')
      setRunStamp(utcPathStamp())
      notify(`ExperimentInstance ${saved.id} 已实例化，状态为 ready`)
    } catch (cause) {
      notifyError(cause)
    } finally {
      setBusy(false)
    }
  }

  const previewInstance = async (item: Json) => {
    if (plan && planInstanceId === item.id) {
      setPlan(null)
      setPlanInstanceId('')
      return
    }
    setBusy(true)
    try {
      const nextPlan = await api.planExperimentInstance(item.id)
      setPlan(nextPlan)
      setPlanInstanceId(item.id)
      notify(`ExperimentInstance ${item.id} 的完整命令已生成`)
    } catch (cause) {
      notifyError(cause)
    } finally {
      setBusy(false)
    }
  }

  const launchInstance = async (item: Json) => {
    setBusy(true)
    try {
      const result = await api.launchExperimentInstance(item.id)
      setPlan(result.plan)
      setPlanInstanceId(item.id)
      await refresh()
      notify(`Launch ${result.job.launch_id}: ${result.job.status}`)
    } catch (cause) {
      notifyError(cause)
    } finally {
      setBusy(false)
    }
  }

  const enqueueInstance = async (item: Json) => {
    try {
      await api.enqueueExperimentInstance(item.id)
      await refresh()
      notify(`ExperimentInstance ${item.id} 已加入队列`)
    } catch (cause) { notifyError(cause) }
  }

  const dequeueInstance = async (item: Json) => {
    try {
      await api.dequeueExperimentInstance(item.id)
      await refresh()
      notify(`ExperimentInstance ${item.id} 已移出队列`)
    } catch (cause) { notifyError(cause) }
  }

  const updateInstanceControls = async (
    item: Json,
    nextPriority: number,
    nextSegmentCaseLimit: number | null,
    nextCaseCompletionTarget: number | null,
  ) => {
    try {
      await api.updateExperimentInstance(item.id, {
        priority: nextPriority,
        next_segment_case_limit: nextSegmentCaseLimit,
        case_completion_target: nextCaseCompletionTarget,
      })
      await refresh()
      notify(`ExperimentInstance ${item.id} 的优先级与 Case 调度目标已更新`)
    } catch (cause) { notifyError(cause) }
  }

  const stopInstance = async (item: Json) => {
    if (!window.confirm(`确认立即停止 ${item.id}？当前正在运行的 Trial 也会被中断。`)) return
    try {
      await api.stopExperimentInstance(item.id)
      await refresh()
      notify(`ExperimentInstance ${item.id} 已停止`)
    } catch (cause) { notifyError(cause) }
  }

  const resumeInstance = async (item: Json) => {
    const successful = item.progress?.successful_trials ?? '已完成'
    const remaining = item.progress?.remaining_trials ?? '剩余'
    if (!window.confirm(`确认继续 ${item.id} 的未完成 Case？将跳过 ${successful} 个成功 Trial，并继续处理剩余 ${remaining} 个 Trial。`)) return
    setBusy(true)
    try {
      await api.resumeExperimentInstance(item.id)
      await refresh()
      notify(`ExperimentInstance ${item.id} 已继续原 Run 的未完成 Case`)
    } catch (cause) { notifyError(cause) } finally { setBusy(false) }
  }

  const stopOrphan = async (item: Json) => {
    if (!window.confirm(`确认停止孤立 Launch ${item.launch_id}？`)) return
    try {
      await api.stopExperimentOrphan(item.launch_id)
      await refresh()
      notify(`孤立 Launch ${item.launch_id} 已停止`)
    } catch (cause) { notifyError(cause) }
  }

  const recoverOrphan = async (item: Json) => {
    try {
      await api.recoverExperimentOrphan(item.launch_id)
      await refresh()
      notify(`Launch ${item.launch_id} 已恢复为 ExperimentInstance`)
    } catch (cause) { notifyError(cause) }
  }

  const startQueue = async () => {
    setBusy(true)
    try {
      const result = await api.startExperimentQueue(maxParallelInstances, maxRunningTrials)
      await refresh()
      const accepted = result.accepted_instance_ids || []
      notify(`调度器已接收 ${accepted.length} 个 ExperimentInstance`)
    } catch (cause) {
      notifyError(cause)
    } finally {
      setBusy(false)
    }
  }

  const drainInstance = async (item: Json) => {
    try {
      await api.drainExperimentInstance(item.id)
      await refresh()
      notify(`ExperimentInstance ${item.id} 将在当前执行中 Trial 完成后停止`)
    } catch (cause) { notifyError(cause) }
  }

  const pauseQueue = async () => {
    try {
      await api.stopExperimentQueueAfterCurrent()
      await refresh()
      notify('调度器已停止接纳新实例；当前并行运行的实例会继续完成')
    } catch (cause) { notifyError(cause) }
  }

  const removeSpec = async () => {
    if (!window.confirm(`确认删除 ExperimentSpec ${draft.id}？`)) return
    try {
      await api.deleteExperimentSpec(draft.id)
      await refresh()
      notify(`ExperimentSpec ${draft.id} 已删除`)
    } catch (cause) { notifyError(cause) }
  }

  const removeInstance = async (item: Json) => {
    if (!window.confirm(`确认删除 ExperimentInstance ${item.id}？`)) return
    try {
      await api.deleteExperimentInstance(item.id)
      if (planInstanceId === item.id) {
        setPlan(null)
        setPlanInstanceId('')
      }
      await refresh()
    } catch (cause) { notifyError(cause) }
  }

  const copyPlanCommand = async () => {
    try {
      await navigator.clipboard.writeText(planCommand)
      notify('完整运行命令已复制')
    } catch (cause) {
      notifyError(cause)
    }
  }

  const closePlan = () => {
    setPlan(null)
    setPlanInstanceId('')
  }

  const canAssemble = Boolean(
    selectedBenchmarkId
    && selectedTeamId
    && candidateBenchmarks.find((item) => item.id === selectedBenchmarkId)?.available
    && candidateTeams.find((item) => item.id === selectedTeamId)?.available
    && !environmentUnavailable,
  )

  return <section className="studio-view experiment-management">
    <nav className="experiment-workspace-tabs" aria-label="实验管理工作区">
      <button className={workspace === 'spec' ? 'active' : ''} onClick={() => setWorkspace('spec')}><span>1</span><strong>实验定义</strong><small>ExperimentSpec</small></button>
      <button className={workspace === 'assemble' ? 'active' : ''} onClick={() => setWorkspace('assemble')}><span>2</span><strong>实例化</strong><small>ExperimentInstance</small></button>
      <button className={workspace === 'runs' ? 'active' : ''} onClick={() => setWorkspace('runs')}><span>3</span><strong>调度与运行</strong><small>Scheduler</small><em>{runningCount + queuedCount}</em></button>
    </nav>
    <div className={`experiment-workspace management-workspace stage-${workspace}`}>
      <section className="experiment-spec-pane management-spec-pane">
        <header className="management-spec-header">
          <div className="section-toolbar">
            <div><h1>ExperimentSpec</h1><span>组合 BenchmarkSpec、TeamSpec 与运行参数；这里只引用 Spec</span></div>
            <div className="toolbar-actions">
              <button className="icon-button" title="新建 ExperimentSpec" onClick={newSpec}><Plus size={16} /></button>
              <button className="secondary" disabled={busy} onClick={save}><Save size={15} />保存</button>
              <button className="icon-button danger" title="删除 ExperimentSpec" onClick={removeSpec}><Trash2 size={15} /></button>
            </div>
          </div>
          <SpecSelector kind="ExperimentSpec" specs={catalog.specs || []} draftId={draft.id || ''} optionLabel={(item) => `${item.id} · ${item.benchmark?.benchmark_spec_id || 'BenchmarkSpec 未设置'}`} onLoad={load} />
        </header>
        <div className="experiment-spec-body">
          <section className="management-editor-section experiment-definition-details">
            <header className="experiment-column-heading"><strong>实验定义</strong><small>选择 BenchmarkSpec、TeamSpec、任务范围与评分合同</small></header>
        <label><span>ExperimentSpec ID</span><input value={draft.id || ''} onChange={(event) => setDraft({ ...draft, id: event.target.value })} /></label>
        <label><span>BenchmarkSpec</span><select value={draft.benchmark?.benchmark_spec_id || ''} onChange={(event) => chooseBenchmarkSpec(event.target.value)}>{benchmarkSpecs.map((item) => <option key={item.id} value={item.id}>{item.category} · {item.name}</option>)}</select></label>
        <label><span>Runnable task</span><select value={draft.benchmark?.runnable_task || ''} onChange={(event) => chooseTask(event.target.value)}>{(benchmarkSpec?.runnable_tasks || []).map((task: string) => <option key={task} value={task}>{task}</option>)}</select></label>
        <div className="two-fields">
          <label><span>Cases</span><div className="input-with-action"><input value={Number.isNaN(draft.benchmark?.cases) ? '' : (draft.benchmark?.cases ?? 10)} placeholder="正整数或 all" spellCheck={false} onChange={(event) => updateSection('benchmark', { cases: event.target.value.toLowerCase() })} /><button className="icon-button" title="运行当前 task 的全部 case" onClick={() => updateSection('benchmark', { cases: 'all' })}><Infinity size={15} /></button></div><small>输入正整数，或输入 all；右侧按钮可直接设为全部。</small></label>
          <label><span>Start index</span><input type="number" min="0" value={draft.benchmark?.start_index || 0} onChange={(event) => updateSection('benchmark', { start_index: Number(event.target.value) })} /></label>
          <label><span>Case selection</span><select value={draft.benchmark?.case_selection || 'head'} onChange={(event) => updateSection('benchmark', { case_selection: event.target.value })}><option value="head">Head · 从起点连续选择</option><option value="uniform">Uniform · 全数据均匀选择</option><option value="stratified">Stratified · 按字段分层选择</option></select><small>小规模子集运行推荐 uniform 或 stratified，避免只验证数据集开头。</small></label>
          {draft.benchmark?.case_selection === 'stratified' && <label><span>Strata field</span><input value={draft.benchmark?.case_strata_field || ''} placeholder="留空自动检测 level/task/domain/repo" onChange={(event) => updateSection('benchmark', { case_strata_field: event.target.value || null })} /></label>}
        </div>
        <label><span>Scoring profile</span><select value={draft.benchmark?.scoring_profile || scoring.default_profile || ''} onChange={(event) => updateSection('benchmark', { scoring_profile: event.target.value })}>{scoringProfiles.map((profile) => <option key={profile} value={profile}>{profile} · {scoring.profiles[profile].scorer_id}</option>)}</select></label>
        <label><span>TeamSpec</span><select value={draft.team_spec_id || ''} onChange={(event) => { setDraft({ ...draft, team_spec_id: event.target.value }); setTeamInstanceId('') }}>{teamSpecs.map((item) => <option key={item.id} value={item.id}>{item.id} · {deriveCoordinationSummary(item).label}</option>)}</select></label>
        <DetailsDisclosure value={draft} label="展开 ExperimentSpec" />
          </section>
          <section className="management-editor-section experiment-runtime-details">
            <header className="experiment-column-heading"><strong>运行参数与输出</strong><small>按作用层级组织 Trial、并发、上下文、采样、工具、网络与观测参数</small></header>
            <div className="two-fields">
              <div className="runtime-category-heading"><strong>执行与调度</strong><span>Trial、错误策略、终止条件与 Trial 并发</span></div>
              <label><span>Method</span><select value={draft.runtime?.method || 'none'} onChange={(event) => updateSection('runtime', { method: event.target.value })}><option value="none">none</option><option value="nl_only">nl_only</option><option value="latent_only">latent_only</option><option value="both">both</option></select></label>
              <label><span>Trials / case</span><input type="number" min="1" step="1" value={draft.runtime?.trials_per_case ?? 1} onChange={(event) => updateSection('runtime', { trials_per_case: Number(event.target.value) })} /><small>每个 Case 的独立完整执行次数；用于 pass@K、自一致性或重复试验，常规 accuracy 保持 1。</small></label>
              <label><span>Trial concurrency mode</span><select value={draft.runtime?.concurrency_policy?.mode || 'fixed'} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), mode: event.target.value } })}><option value="fixed">Fixed · 固定并发目标</option><option value="auto">Dynamic · 动态弹性准入</option></select><small>第一阶段只补足运行池内已有的固定实例，不查看待调度固定实例；第二阶段才在压力正常时按待调度序列逐个接纳。一个 Trial 是一个 (case_id, trial_index)。</small></label>
              {draft.runtime?.concurrency_policy?.mode !== 'auto'
                ? <label><span>固定并发 Trial 数</span><input type="number" min="1" value={draft.runtime?.trial_concurrency || 1} onChange={(event) => updateSection('runtime', { trial_concurrency: Number(event.target.value), concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), mode: 'fixed', initial: Number(event.target.value), minimum: Number(event.target.value), maximum: Number(event.target.value) } })} /><small>待调度时按普通序列等待；一旦运行池中已有该实例的 Trial，每轮就在压力判断前逐个补足到 min(剩余 Trial 数, n)。</small></label>
                : <label><span>最大并发 Trial 数</span><input type="number" min="1" value={draft.runtime?.concurrency_policy?.maximum || draft.runtime?.trial_concurrency || 1} onChange={(event) => updateSection('runtime', { trial_concurrency: Number(event.target.value), concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), mode: 'auto', minimum: 1, initial: Math.min(Number(event.target.value), Number(draft.runtime?.concurrency_policy?.initial || 1)), maximum: Number(event.target.value) } })} /><small>Scheduler 在该上限内依据优先级和实时资源压力逐个接纳 Trial。</small></label>}
              <label><span>On Trial error</span><select value={draft.runtime?.on_trial_error || 'continue'} onChange={(event) => updateSection('runtime', { on_trial_error: event.target.value })}><option value="continue">continue · 记录失败并继续</option><option value="fail-fast">fail-fast · 立即停止 Run</option></select><small>同一 Trial 的全部 Attempt 都失败后，决定是否继续后续 Trial。</small></label>
              <label><span>Max Attempts / Trial</span><input type="number" min="1" step="1" value={draft.runtime?.max_attempts_per_trial ?? 1} onChange={(event) => updateSection('runtime', { max_attempts_per_trial: Math.max(1, Number(event.target.value || 1)) })} /><small>一次 Trial 最多执行多少次；1 表示不重试。错误答案不会触发 Attempt，只有运行时异常才会。</small></label>
              {isFixedSchedule
                ? <><label><span>Max rounds</span><input type="number" min="1" value={draft.runtime?.max_rounds || 4} onChange={(event) => updateSection('runtime', { max_rounds: Number(event.target.value) })} /><small>固定执行顺序的完整轮数；每轮最多包含 {participantCount} 个 participant turn。</small></label><label><span>Max turns（自动换算）</span><input value={effectiveMaxTurns} readOnly /><small>{participantCount} 个角色 × {draft.runtime?.max_rounds || 4} 轮；由所选 RuntimeAdapter 编译。</small></label></>
                : <><label><span>Max turns</span><input type="number" min="1" value={draft.runtime?.max_turns || 20} onChange={(event) => updateSection('runtime', { max_turns: Number(event.target.value) })} /><small>{dynamicScheduleDescription}时的 participant 工作步数上限。</small></label><label><span>Max rounds</span><input value="不适用" readOnly /><small>当前显式关系图不存在“所有角色依次发言一遍”的固定轮次。</small></label></>}
              <div className="runtime-limit-field"><span>Max model calls / case</span><div className="runtime-limit-control"><input type="number" min="1" disabled={modelCallsUnlimited} value={modelCallsUnlimited ? '' : draft.runtime?.max_model_calls_per_case} placeholder="不限制" onChange={(event) => updateSection('runtime', { max_model_calls_per_case: Math.max(1, Number(event.target.value || 1)) })} /><label className="check-field"><input type="checkbox" checked={modelCallsUnlimited} onChange={(event) => updateSection('runtime', { max_model_calls_per_case: event.target.checked ? null : 40 })} /><span>不限制</span></label></div><small>统计 participant、Selector 和 Orchestrator 发出的所有真实模型请求；达到后终止当前 case。</small></div>
              <div className="runtime-limit-field"><span>Max wall time / Trial</span><div className="runtime-limit-control"><input type="number" min="1" step="1" disabled={caseWallTimeUnlimited} value={caseWallTimeUnlimited ? '' : draft.runtime?.max_case_wall_time_s} placeholder="不限制" onChange={(event) => updateSection('runtime', { max_case_wall_time_s: Math.max(1, Number(event.target.value || 1)) })} /><label className="check-field"><input type="checkbox" checked={caseWallTimeUnlimited} onChange={(event) => updateSection('runtime', { max_case_wall_time_s: event.target.checked ? null : 1800 })} /><span>不限制</span></label></div><small>整个 Trial 的墙钟秒数上限，包含框架调度、模型请求和工具执行；超时按运行错误记录。</small></div>
              <div className="runtime-category-heading"><strong>上下文与思考预算</strong><span>每次模型调用的输入窗口、输出上限与保留量</span></div>
              <label><span>Max new tokens</span><input type="number" min="1" value={draft.runtime?.max_new_tokens || 16384} onChange={(event) => updateSection('runtime', { max_new_tokens: Number(event.target.value) })} /></label>
              <label><span>Max input tokens</span><input type="number" min="1" value={draft.runtime?.max_input_tokens ?? ''} placeholder="自动按模型窗口分配" onChange={(event) => updateSection('runtime', { max_input_tokens: event.target.value ? Number(event.target.value) : null })} /></label>
              <label><span>Min output reserve</span><input type="number" min="0" value={draft.runtime?.min_output_reserve_tokens ?? 2048} onChange={(event) => updateSection('runtime', { min_output_reserve_tokens: Number(event.target.value) })} /><small>自适应裁剪输入时至少为本次模型输出保留的 token。</small></label>
              <label><span>Min thinking reserve</span><input type="number" min="0" value={draft.runtime?.min_thinking_reserve_tokens ?? 0} onChange={(event) => updateSection('runtime', { min_thinking_reserve_tokens: Number(event.target.value) })} /></label>
              <label><span>Max thinking budget</span><input type="number" min="1" value={draft.runtime?.max_thinking_budget_tokens ?? ''} placeholder="不单独限制" onChange={(event) => updateSection('runtime', { max_thinking_budget_tokens: event.target.value ? Number(event.target.value) : null })} /></label>
              <label><span>Min final reserve</span><input type="number" min="0" value={draft.runtime?.min_final_reserve_tokens ?? 1024} onChange={(event) => updateSection('runtime', { min_final_reserve_tokens: Number(event.target.value) })} /></label>
              <label><span>Safety margin</span><input type="number" min="0" value={draft.runtime?.safety_margin_tokens ?? 256} onChange={(event) => updateSection('runtime', { safety_margin_tokens: Number(event.target.value) })} /></label>
              <div className="runtime-category-heading"><strong>生成与采样</strong><span>随机性、候选过滤、重复惩罚与可复现种子</span></div>
              <label><span>Sampling</span><select value={draft.runtime?.do_sample ? 'enabled' : 'disabled'} onChange={(event) => updateSection('runtime', { do_sample: event.target.value === 'enabled' })}><option value="disabled">Disabled · deterministic</option><option value="enabled">Enabled · stochastic</option></select><small>关闭表示不随机采样；当前 vLLM/API 映射为 temperature=0、top_p=1，其他确定性搜索仍取决于后端能力。</small></label>
              <label><span>Temperature</span><input type="number" min="0" step="0.05" value={draft.runtime?.temperature ?? 0.7} onChange={(event) => updateSection('runtime', { temperature: Number(event.target.value) })} /></label>
              <label><span>Top P</span><input type="number" min="0.01" max="1" step="0.05" value={draft.runtime?.top_p ?? 0.8} onChange={(event) => updateSection('runtime', { top_p: Number(event.target.value) })} /></label>
              <label><span>Top K</span><input type="number" min="1" step="1" value={draft.runtime?.top_k ?? ''} placeholder="Provider 默认" onChange={(event) => updateSection('runtime', { top_k: event.target.value ? Number(event.target.value) : null })} /><small>采样时最多保留概率最高的 K 个候选 token。</small></label>
              <label><span>Min P</span><input type="number" min="0" max="1" step="0.05" value={draft.runtime?.min_p ?? ''} placeholder="Provider 默认" onChange={(event) => updateSection('runtime', { min_p: event.target.value ? Number(event.target.value) : null })} /><small>过滤相对概率过低的候选；0 表示不额外过滤。</small></label>
              <label><span>Presence penalty</span><input type="number" min="-2" max="2" step="0.1" value={draft.runtime?.presence_penalty ?? ''} placeholder="Provider 默认" onChange={(event) => updateSection('runtime', { presence_penalty: event.target.value ? Number(event.target.value) : null })} /><small>对已经生成过的 token 施加固定惩罚；0 表示关闭。</small></label>
              <label><span>Repetition penalty</span><input type="number" min="0.01" step="0.05" value={draft.runtime?.repetition_penalty ?? 1} onChange={(event) => updateSection('runtime', { repetition_penalty: Number(event.target.value) })} /><small>乘法式重复惩罚；1 表示关闭。</small></label>
              <label><span>Base seed</span><input type="number" step="1" value={draft.runtime?.seed ?? 0} onChange={(event) => updateSection('runtime', { seed: Number(event.target.value) })} /><small>实际 Trial seed 由该值、Benchmark/Task、稳定 Case ID 和 trial_index 派生；数据重排、并发、恢复和异常重试不会改变它。</small></label>
              <label><span>Model call console trace</span><select value={draft.runtime?.trace_model_calls === false ? 'disabled' : 'enabled'} onChange={(event) => updateSection('runtime', { trace_model_calls: event.target.value === 'enabled' })}><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select><small>只控制终端 start/done 摘要；无损 EventLog 始终持续记录。</small></label>
              <div className="runtime-category-heading"><strong>工具与沙盒</strong><span>代码执行、浏览器与 Benchmark 专用镜像</span></div>
              <label><span>Code executor</span><select value={draft.runtime?.code_executor || 'docker'} onChange={(event) => updateSection('runtime', { code_executor: event.target.value })}><option value="docker">Docker sandbox</option><option value="local">Local process</option></select><small>正式工具评测建议使用 Docker；local 会直接在宿主环境执行代码。</small></label>
              <label><span>Code timeout (s)</span><input type="number" min="1" value={draft.runtime?.code_timeout ?? 60} onChange={(event) => updateSection('runtime', { code_timeout: Number(event.target.value) })} /><small>单个代码块的执行上限；60 秒与 AgBench GAIA 模板使用的 AutoGen 默认值一致。</small></label>
              <label><span>Tool workspace root</span><input value={draft.runtime?.work_root || 'runs/lychee_tool_workspaces'} onChange={(event) => updateSection('runtime', { work_root: event.target.value })} /></label>
              <label><span>Browser mode</span><select value={draft.runtime?.web_headless === false ? 'visible' : 'headless'} onChange={(event) => updateSection('runtime', { web_headless: event.target.value === 'headless' })}><option value="headless">Headless</option><option value="visible">Visible</option></select></label>
              <label><span>Browser screenshots</span><select value={draft.runtime?.save_screenshots ? 'enabled' : 'disabled'} onChange={(event) => updateSection('runtime', { save_screenshots: event.target.value === 'enabled' })}><option value="disabled">Disabled</option><option value="enabled">Enabled</option></select></label>
              <div className="runtime-category-heading"><strong>观测与事件日志</strong><span>服务压力采样、控制台摘要与无损 EventLog 分片</span></div>
              <label><span>vLLM service metrics</span><select value={draft.observability?.collect_vllm_metrics === false ? 'disabled' : 'enabled'} onChange={(event) => updateSection('observability', { collect_vllm_metrics: event.target.value === 'enabled' })}><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select><small>定时读取每个 vLLM DeploymentInstance 的 /metrics；API 与 Local HF 没有对应目标时不会伪造数据。</small></label>
              <label><span>vLLM metrics interval (s)</span><input type="number" min="0.5" step="0.5" value={draft.observability?.vllm_metrics_interval_s ?? 5} onChange={(event) => updateSection('observability', { vllm_metrics_interval_s: Number(event.target.value) })} /><small>服务级时间序列采样间隔；通常 5 秒足够观察排队和 KV cache 压力。</small></label>
              <label><span>EventLog events / file</span><input type="number" min="1" step="1" value={draft.observability?.event_log_max_events_per_file ?? 10000} onChange={(event) => updateSection('observability', { event_log_max_events_per_file: Number(event.target.value) })} /><small>单个 JSONL 分片的 Event 数上限；达到后自动滚动到下一分片。</small></label>
              <label><span>EventLog MiB / file</span><input type="number" min="1" step="1" value={draft.observability?.event_log_max_mib_per_file ?? 64} onChange={(event) => updateSection('observability', { event_log_max_mib_per_file: Number(event.target.value) })} /><small>单个 JSONL 分片的近似大小上限；事件数或大小任一达到即滚动。</small></label>
              {sandboxProfiles.length > 0 && <label><span>Sandbox profile</span><select value={draft.runtime?.docker_image || benchmarkSpec?.runtime_defaults?.docker_image || ''} onChange={(event) => updateSection('runtime', { docker_image: event.target.value })}>{sandboxProfiles.map(([profileId, profile]) => <option key={profileId} value={profile.docker_image}>{profileId} · {profile.label}</option>)}</select><small>{selectedSandboxProfile?.[1]?.description || '选择该 benchmark 的代码执行环境；具体镜像会写入运行配置和追踪记录。'}</small></label>}
            </div>
            <div className="runtime-category-heading runtime-category-wide"><strong>网络与输出</strong><span>Benchmark 网络策略和本次 Run 的保存根目录</span></div>
            <label><span>Network</span><select value={draft.network?.mode || 'benchmark_default'} onChange={(event) => chooseNetworkMode(event.target.value)}><option value="benchmark_default">Benchmark 默认 · {benchmarkNetworkDefaults.mode}</option><option value="direct">Direct</option><option value="proxy">Proxy</option><option value="offline" disabled>Offline · 尚未实现强制隔离</option></select></label>
            {(draft.network?.mode || 'benchmark_default') === 'benchmark_default' && <div className="network-default-summary">
              <div><strong>当前 Benchmark 默认配置</strong><span>{benchmarkSpec?.id || '未选择 BenchmarkSpec'}</span></div>
              <dl>
                <dt>Mode</dt><dd>{benchmarkNetworkDefaults.mode}</dd>
                <dt>Access</dt><dd>{benchmarkNetworkDefaults.access}</dd>
                <dt>Targets</dt><dd>{enabledDefaultNetworkTargets.join(', ') || 'none'}</dd>
                {benchmarkNetworkDefaults.proxy_url && <><dt>Proxy</dt><dd>{benchmarkNetworkDefaults.proxy_url}</dd></>}
                <dt>Probe</dt><dd>{benchmarkNetworkDefaults.probe_url}</dd>
              </dl>
              <DetailsDisclosure value={benchmarkNetworkDefaults} label="展开 Network 默认配置" />
            </div>}
            {draft.network?.mode === 'proxy' && <>
              <label><span>Proxy URL</span><input value={draft.network?.proxy_url || ''} onChange={(event) => updateSection('network', { proxy_url: event.target.value })} /></label>
              <div className="network-target-controls"><strong>代理应用目标</strong>{[['downloads', 'Benchmark downloads'], ['web_surfer', 'WebSurfer'], ['code_executor', 'Docker code executor'], ['model_backend', 'Model API backend']].map(([target, label]) => <label key={target} className="check-field"><input type="checkbox" checked={Boolean(draft.network?.targets?.[target])} onChange={(event) => patchNetworkTarget(target, event.target.checked)} /><span>{label}</span></label>)}</div>
              <label><span>No proxy</span><input value={draft.network?.no_proxy || '127.0.0.1,localhost,::1'} onChange={(event) => updateSection('network', { no_proxy: event.target.value })} /><small>逗号分隔；这些地址不经过代理，通常包括本机 vLLM endpoint。</small></label>
              <label><span>Docker bridge host</span><input value={draft.network?.docker_bridge_host || '172.17.0.1'} onChange={(event) => updateSection('network', { docker_bridge_host: event.target.value })} /><small>容器访问宿主代理时使用的 bridge 地址。</small></label>
              <label><span>Container proxy port</span><input type="number" min="1" max="65535" value={draft.network?.container_proxy_port || 17897} onChange={(event) => updateSection('network', { container_proxy_port: Number(event.target.value) })} /><small>LycheeMAS 为 Docker 工具暴露宿主代理的端口。</small></label>
              <label><span>Probe URL</span><input value={draft.network?.probe_url || 'https://www.google.com/generate_204'} onChange={(event) => updateSection('network', { probe_url: event.target.value })} /><small>启动 Trial 前验证所选网络路径，不参与 Benchmark 作答。</small></label>
            </>}
            {draft.network?.mode === 'offline' && <small className="field-warning">当前版本尚未强制隔离宿主、Playwright 和 Docker 网络；该历史配置不能作为严格离线实验条件。</small>}
            <label><span>运行结果根目录</span><div className="input-with-action"><div className="input-with-icon"><Folder size={15} /><input value={draft.environment?.runs_root || defaultRunsRoot} onChange={(event) => updateSection('environment', { runs_root: event.target.value })} /></div><button className="icon-button" title="恢复默认结果目录" onClick={() => updateSection('environment', { runs_root: defaultRunsRoot })}><RotateCcw size={14} /></button></div><small>默认保存到 {defaultRunsRoot}；可填写服务器上的绝对路径或相对仓库的路径。</small></label>
          </section>
        </div>
      </section>

      <aside className="experiment-instance-pane management-instance-pane">
        <section className="management-instance-builder">
          <div className="section-toolbar"><div><h1>ExperimentInstance</h1><span>绑定具体资源与启动器；实例化后再选择运行或排队</span></div><button className="icon-button" title="刷新实例与队列" onClick={refresh}><RefreshCw size={15} /></button></div>
        <label><span>BenchmarkInstance</span><select value={selectedBenchmarkId} onChange={(event) => setBenchmarkInstanceId(event.target.value)}><option value="">没有匹配实例</option>{candidateBenchmarks.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{item.id} · {item.status}</option>)}</select></label>
        <label><span>TeamInstance</span><select value={selectedTeamId} onChange={(event) => setTeamInstanceId(event.target.value)}><option value="">没有匹配实例</option>{candidateTeams.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{item.id} · {item.observed_status}</option>)}</select></label>
        <div className="contract-summary">
          <strong>{benchmarkSpec?.name || '未选择 BenchmarkSpec'} + {teamSpec?.id || '未选择 TeamSpec'}</strong>
          <span>{draft.benchmark?.runnable_task} · scorer={scoring.profiles?.[draft.benchmark?.scoring_profile || scoring.default_profile]?.scorer_id || 'unknown'}</span>
          <span>Spec 只规定类型；这里选择的两个 Instance 只写入 ExperimentInstance</span>
        </div>
        {selectedBenchmark && <DetailsDisclosure value={selectedBenchmark} label="展开 BenchmarkInstance" />}
        {selectedTeam && <DetailsDisclosure value={selectedTeam} label="展开 TeamInstance" />}
        <label><span>ExperimentInstance ID</span><input value={instanceId} placeholder={`${slug(draft.id, 'experiment')}-run-xxxxxxxx`} onChange={(event) => setInstanceId(event.target.value)} /><small>默认由 ExperimentSpec ID、run 和 8 位随机标识组成；可在实例化前手动修改。</small></label>
        <label><span>本次运行数据目录</span><div className="input-with-action"><div className="input-with-icon"><Folder size={15} /><input value={effectiveRunDir} onChange={(event) => setRunDir(event.target.value)} /></div><button className="icon-button" title="生成新的自动运行目录" onClick={() => { setRunDir(''); setRunStamp(utcPathStamp()) }}><RotateCcw size={14} /></button></div><small>{runDir ? '使用该 ExperimentInstance 的自定义精确目录。' : '自动按运行根目录 / BenchmarkSpec / task / TeamSpec / Method / ExperimentSpec / UTC 时间戳生成。'}</small></label>
        <label><span>Launcher</span><select value={launcherType} onChange={(event) => setLauncherType(event.target.value)}><option value="subprocess">Studio 后台 Subprocess</option><option value="new_tmux_session">新建 tmux 会话</option><option value="existing_tmux_session">使用现有 tmux 会话</option></select><small>立即运行和队列运行都会使用这里保存的同一个 Launcher。</small></label>
        {(launcherType === 'new_tmux_session' || launcherType === 'existing_tmux_session') && <div className="two-fields"><label><span>tmux session</span>{launcherType === 'existing_tmux_session' && (bootstrap.tmux_sessions || []).length > 0 ? <select value={tmuxSession} onChange={(event) => setTmuxSession(event.target.value)}>{(bootstrap.tmux_sessions || []).map((item: Json) => <option key={item.name} value={item.name}>{item.name} · {item.windows} windows</option>)}</select> : <input value={tmuxSession} onChange={(event) => setTmuxSession(event.target.value)} />}</label><label><span>tmux window</span><input value={tmuxWindow} onChange={(event) => setTmuxWindow(event.target.value)} /></label></div>}
        <label><span>队列优先级</span><input type="number" value={priority} onChange={(event) => setPriority(Number(event.target.value))} /></label>
        <div className="runtime-limit-field"><span>下一执行段新增 Case</span><div className="runtime-limit-control"><input type="number" min="1" disabled={segmentCaseLimit == null} value={segmentCaseLimit ?? ''} placeholder="不限制" onChange={(event) => setSegmentCaseLimit(Math.max(1, Number(event.target.value || 1)))} /><label className="check-field"><input type="checkbox" checked={segmentCaseLimit == null} onChange={(event) => setSegmentCaseLimit(event.target.checked ? null : 50)} /><span>不限制</span></label></div><small>只限制下一次启动或续跑额外选择的新 Case 数；当前 draining 阶段的执行中 Case 不计入这里。</small></div>
        <div className="runtime-limit-field"><span>累计完成 Case 上限</span><div className="runtime-limit-control"><input type="number" min="1" disabled={caseCompletionTarget == null} value={caseCompletionTarget ?? ''} placeholder="不限制" onChange={(event) => setCaseCompletionTarget(Math.max(1, Number(event.target.value || 1)))} /><label className="check-field"><input type="checkbox" checked={caseCompletionTarget == null} onChange={(event) => setCaseCompletionTarget(event.target.checked ? null : 50)} /><span>不限制</span></label></div><small>可选硬上限。Scheduler 会取“下一执行段数量”和“距离累计上限的余量”中的较小值。</small></div>
        {environmentUnavailable && <div className="experiment-environment-warning"><CircleAlert size={15} /><span>当前实验环境不可用：{environment.python}。请先到“运行环境”选择可用环境并设为实验环境。</span></div>}
        <button className="primary" disabled={!canAssemble || busy} onClick={instantiate}><Check size={15} />实例化</button>
        </section>
        <section className="management-instance-registry">
        <div className="experiment-run-layout">
        <section className="experiment-run-pool">
        <div className="experiment-queue-toolbar"><div><h2>运行与调度</h2><span>Scheduler · 逐 Trial 准入、资源压力与优先级顺序</span></div></div>
        <div className="scheduler-control-bar">
          <label className="queue-parallel-limit"><span><strong>运行器上限</strong><small>同时存活的实验进程数</small></span><input type="number" min="1" max="64" value={maxParallelInstances} onChange={(event) => setMaxParallelInstances(Math.max(1, Math.min(64, Number(event.target.value) || 1)))} /></label>
          <label className="queue-parallel-limit"><span><strong>运行中 Trial 硬上限</strong><small>仅作监控滞后和工具资源的安全熔断</small></span><input type="number" min="1" max="4096" value={maxRunningTrials} onChange={(event) => setMaxRunningTrials(Math.max(1, Math.min(4096, Number(event.target.value) || 1)))} /></label>
          <div className="toolbar-actions"><button className="secondary" title={catalog.queue?.running ? '热更新运行器和运行中 Trial 安全上限' : (queuedCount ? '按压力、优先级和每个实验的并发规则逐个接纳 Trial' : '请先将 ready 实例加入队列')} disabled={(!catalog.queue?.running && queuedCount === 0) || busy} onClick={startQueue}><Play size={14} />{catalog.queue?.running ? '应用上限' : '启动调度器'}</button><button className="secondary" title="不再接纳新的 Trial；已经运行和正在启动的 Trial 继续完成" disabled={!catalog.queue?.running || busy} onClick={pauseQueue}><Square size={14} />停止接纳</button></div>
        </div>
        <div className="queue-overview">
          <div><span>调度状态</span><strong>{catalog.queue?.running ? '接纳中' : '已停止接纳'}</strong></div>
          <div><span>运行实例</span><strong>{runningCount}</strong></div>
          <div><span>排队实例</span><strong>{queuedCount}</strong></div>
          <div title="这是防止监控滞后和工具资源过载的最终安全上限，不是正常调度目标"><span>运行池 Trial</span><strong>{catalog.queue?.running_trial_pool?.occupied || 0} / {catalog.queue?.running_trial_pool?.hard_limit || maxRunningTrials}</strong><small>{catalog.queue?.running_trial_pool?.running || 0} 运行中 · {catalog.queue?.running_trial_pool?.launching || 0} 启动中</small></div>
          {Object.entries(catalog.queue?.resource_usage || {}).map(([deploymentId, usage]: [string, any]) => <div key={deploymentId} title={deploymentId}><span>{deploymentId}</span><strong>{usage.associated_running_trials || 0} 个关联运行 Trial</strong><small>{usage.associated_launching_trials || 0} 启动中 · 模型请求并发上限 {usage.request_concurrency_limit ?? '—'}</small></div>)}
        </div>
        <ResourcePressure
          title="运行池准入压力"
          subtitle="只汇总当前运行池涉及的 Deployment 与 GPU，用于 Scheduler 是否接纳新 Trial；后端自身压力请在部署管理查看"
          entries={admissionPressureEntries}
        />
        {(runningCount > 0 || queuedCount > 0) && <details className="queue-instance-identities"><summary>查看调度中的实例 ID</summary><div><span>active={(catalog.queue?.active_instance_ids || []).join(', ') || 'none'}</span><span>queued={(catalog.queue?.queued_instance_ids || []).join(', ') || 'none'}</span></div></details>}
        {(catalog.queue?.invalid_queued_instance_ids || []).length > 0 && <div className="inline-warning">有 {(catalog.queue?.invalid_queued_instance_ids || []).length} 个实例因 Spec 已变化而不能进入运行池，已归入“需处理”。</div>}
        {experimentScheduling.length > 0 && <section className="trial-sequence-board">
          <article>
            <header><strong>运行池内 Trial 序列</strong><span>{catalog.queue?.running_trial_pool?.running || 0} 运行中 · {catalog.queue?.running_trial_pool?.launching || 0} 启动中</span></header>
            <div className="trial-token-lane">
              {trialSequencePresentation.running.map((item: Json, index: number) => <span
                className={`trial-token ${item.state}`}
                key={`${item.instance_id}-${item.case_order ?? 'launching'}-${item.trial_index ?? index}-${index}`}
                style={{ '--trial-color': item.color } as CSSProperties}
                title={`${item.instance_id} · ${item.state === 'running' ? `case=${item.case_id || item.case_order || 'unknown'}` : '已获准入许可，Runner 正在启动'}`}
              >{item.label}</span>)}
              {trialSequencePresentation.running.length === 0 && <em>暂无运行中或启动中的 Trial</em>}
              {trialSequencePresentation.runningHidden > 0 && <b>+{trialSequencePresentation.runningHidden}</b>}
            </div>
          </article>
          <article>
            <header><strong>待调度 Trial 序列</strong><span>按优先级、入队时间排列，Scheduler 从队首逐个检查</span></header>
            <div className="trial-token-lane waiting">
              {trialSequencePresentation.waiting.map((item: Json, index: number) => <span
                className="trial-token waiting"
                key={`${item.instance_id}-${index}`}
                style={{ '--trial-color': item.color } as CSSProperties}
                title={`${item.instance_id} · priority=P${item.priority}`}
              >{item.label}</span>)}
              {trialSequencePresentation.waiting.length === 0 && <em>待调度序列为空</em>}
              {trialSequencePresentation.waitingHidden > 0 && <b>+{trialSequencePresentation.waitingHidden}</b>}
            </div>
          </article>
        </section>}
        {experimentScheduling.length > 0 && <section className="case-pool-queues">
          <header><strong>ExperimentInstance 调度状态</strong><span>{experimentScheduling.length} 个实验 · 颜色与上方 Trial 序列一致</span></header>
          <div className="case-pool-queue-list">{experimentScheduling.map((queue: Json) => {
            const fixedConcurrency = queue.trial_concurrency_mode === 'fixed'
            const waitLabel = queue.admission_wait_reason === 'dynamic_paused_by_pressure'
              ? '动态并发已因 vLLM 或 GPU 压力暂停接纳'
              : queue.admission_wait_reason === 'pressure_paused'
                ? '运行池资源压力过载，本轮不再接纳新 Trial'
              : queue.admission_wait_reason === 'running_trial_hard_limit'
                ? '已达到运行中 Trial 安全硬上限'
                : queue.admission_wait_reason === 'next_admission_tick'
                  ? '等待下一轮压力采样后继续逐个接纳'
                  : null
            const instanceColor = trialSequencePresentation.colorByInstance.get(String(queue.instance_id)) || '#71817d'
            const instanceAlias = trialSequencePresentation.aliasByInstance.get(String(queue.instance_id)) || '?'
            return <div className={`case-pool-queue ${queue.state}`} key={queue.instance_id} style={{ '--trial-color': instanceColor } as CSSProperties}>
              <span className="case-pool-rank">{instanceAlias}</span>
              <strong title={queue.instance_id}>{queue.instance_id}</strong>
              <span className="case-pool-priority">优先级 P{queue.priority}</span>
              <div className="case-pool-queue-details">
                <span title="固定并发会补足设定值；动态并发会在压力升高时暂停新增 Trial">
                  <b>Trial 并发 · {fixedConcurrency ? '固定' : '动态'}</b>
                  {fixedConcurrency ? '固定目标' : '动态上限'} {queue.max_trial_concurrency}
                  <i>→</i>
                  运行中 {queue.running_trial_count}
                  <i>·</i>
                  启动中 {queue.launching_trial_count}
                </span>
                <span>
                  <b>Trial 工作量</b>
                  剩余 {queue.remaining_trial_count ?? '待读取'}
                  <i>·</i>
                  待调度 {queue.waiting_trial_count ?? '待读取'}
                </span>
                {waitLabel && <span className="queue-admission-wait"><b>当前状态</b>{waitLabel}</span>}
              </div>
            </div>
          })}</div>
        </section>}
        {orphanedLaunches.length > 0 && <section className="orphaned-launches">
          <header><div><CircleAlert size={15} /><strong>孤立 Launch</strong></div><span>进程仍在运行，但缺少 ExperimentInstance；可恢复登记或停止。</span></header>
          {orphanedLaunches.map((item: Json) => <article className="experiment-instance-card orphaned-launch-card" key={item.launch_id}>
            <div className="experiment-instance-card-main">
              <header><span className="status-pill orphaned">orphaned</span><strong title={item.launch_id}>{item.launch_id}</strong></header>
              <dl className="experiment-instance-meta"><dt>PID</dt><dd>{item.job?.pid || '—'}</dd><dt>Launcher</dt><dd>{launcherLabels[item.job?.mode] || item.job?.mode}</dd><dt>可恢复</dt><dd>{item.recoverable ? 'yes' : 'no'}</dd><dt>Run directory</dt><dd>{item.run_dir || '—'}</dd></dl>
              {item.progress && <ExperimentProgress progress={item.progress} />}
              <DetailsDisclosure value={item} label="展开孤立 Launch" className="card-details" />
            </div>
            <div className="row-actions experiment-instance-actions">{item.recoverable && <button className="icon-button" title="恢复为 ExperimentInstance" onClick={() => recoverOrphan(item)}><RotateCcw size={14} /></button>}<button className="icon-button danger" title="停止孤立 Launch" onClick={() => stopOrphan(item)}><Square size={14} /></button></div>
          </article>)}
        </section>}
        </section>
        <section className="experiment-run-instances">
        <div className="experiment-instance-filter">
          <div className="segmented" aria-label="筛选 ExperimentInstance">
            {([
              ['active', '运行中', instanceCounts.active],
              ['ready', '待运行', instanceCounts.ready],
              ['attention', '需处理', instanceCounts.attention],
              ['finished', '已完成', instanceCounts.finished],
              ['all', '全部', instanceCounts.all],
            ] as const).map(([id, label, count]) => <button key={id} className={instanceFilter === id ? 'active' : ''} onClick={() => setInstanceFilter(id)}>{label}<span>{count}</span></button>)}
          </div>
          <label className="experiment-instance-search"><Search size={14} /><input value={instanceSearch} placeholder="搜索实例、Spec、Team 或路径" onChange={(event) => setInstanceSearch(event.target.value)} /></label>
          <span>显示 {visibleInstances.length} / {experimentInstances.length}</span>
        </div>
        <div className="experiment-instance-list">{visibleInstances.map((item: Json) => <article className={`experiment-instance-card ${plan && planInstanceId === item.id ? 'command-open' : ''}`} key={item.id}>
          <div className="experiment-instance-card-main">
            <header><span className={`status-pill ${item.progress?.run_status === 'complete_with_errors' ? 'failed' : item.status}`}>{item.progress?.run_status === 'complete_with_errors' ? 'complete_with_errors' : item.status}</span><strong title={item.id}>{item.id}</strong></header>
            <dl className="experiment-instance-meta"><dt>ExperimentSpec</dt><dd>{item.experiment_spec_id}</dd><dt>Launcher</dt><dd>{launcherLabels[item.launcher?.type] || item.launcher?.type}</dd><dt>BenchmarkInstance</dt><dd>{item.benchmark_instance_id}</dd><dt>TeamInstance</dt><dd>{item.team_instance_id}</dd></dl>
            {item.run_dir && <code className="experiment-instance-path" title={item.run_dir}>{item.run_dir}</code>}
            {item.progress && <ExperimentProgress progress={item.progress} />}
            {item.status !== 'completed' && <section className="instance-scheduling-controls"><header><strong>调度设置</strong><span>P{item.queue?.priority ?? 100} · 下一执行段 {item.execution?.next_segment_case_limit ?? item.execution?.segment_case_limit ?? '不限制'}</span></header><InstanceExecutionControls item={item} busy={busy} onSave={updateInstanceControls} /></section>}
            {item.error && <div className="experiment-instance-error"><CircleAlert size={14} /><span>{item.error}</span></div>}
            <DetailsDisclosure value={item} label="展开 ExperimentInstance" className="card-details" />
          </div>
          <div className="row-actions experiment-instance-actions"><button className="icon-button" title={plan && planInstanceId === item.id ? '收起完整命令' : '生成完整命令'} disabled={busy} onClick={() => previewInstance(item)}><Clipboard size={14} /></button>{item.status === 'ready' && <><button className="icon-button" title="立即运行" disabled={busy} onClick={() => launchInstance(item)}><Play size={14} /></button><button className="icon-button" title="加入队列" onClick={() => enqueueInstance(item)}><ListPlus size={14} /></button></>}{item.status === 'queued' && <button className="icon-button" title="移出队列" onClick={() => dequeueInstance(item)}><ListMinus size={14} /></button>}{item.status === 'running' && <>{item.progress?.drain_requested && (item.queue?.enabled ? <button className="icon-button" title="取消本次安全停止后的自动续队" onClick={() => dequeueInstance(item)}><ListMinus size={14} /></button> : <button className="icon-button" title="当前执行中 Trial 完成后，按下一执行段配置自动进入队列" onClick={() => enqueueInstance(item)}><ListPlus size={14} /></button>)}<button className="icon-button" disabled={item.progress?.drain_available === false || Boolean(item.progress?.drain_requested)} title={item.progress?.drain_available === false ? '该 Run 由旧版 runner 启动，不支持在 Trial 边界优雅停止' : item.progress?.drain_requested ? '已请求优雅停止，正在等待当前执行中 Trial 完成' : '完成当前执行中 Trial 后停止；不再启动新 Trial'} onClick={() => drainInstance(item)}><Pause size={14} /></button><button className="icon-button danger" title="立即强制停止；当前执行中 Trial 也会中断" onClick={() => stopInstance(item)}><Square size={14} /></button></>}{(item.status === 'paused' || item.status === 'stopped' || item.status === 'failed') && item.progress?.resume_available !== false && <><button className="icon-button" title="继续未完成 Case：复用原 Run 并跳过已成功 Trial" disabled={busy} onClick={() => resumeInstance(item)}><CirclePlay size={14} /></button><button className="icon-button" title="加入队列并继续未完成 Case" disabled={busy} onClick={() => enqueueInstance(item)}><ListPlus size={14} /></button></>}{item.status !== 'running' && item.status !== 'queued' && <button className="icon-button danger" title="删除 ExperimentInstance" onClick={() => removeInstance(item)}><Trash2 size={14} /></button>}</div>
          {plan && planInstanceId === item.id && <section className="experiment-instance-command">
            <header>
              <div><strong>完整运行命令</strong><span title={plan.run_dir}>运行数据：{plan.run_dir}</span></div>
              <div className="row-actions"><button className="icon-button" title="复制完整运行命令" disabled={!planCommand} onClick={copyPlanCommand}><Clipboard size={14} /></button><button className="icon-button" title="收起完整运行命令" onClick={closePlan}><X size={15} /></button></div>
            </header>
            <pre><code>{planCommand}</code></pre>
          </section>}
        </article>)}{visibleInstances.length === 0 && <div className="experiment-empty-state"><strong>没有匹配的 ExperimentInstance</strong><span>调整状态筛选或搜索关键词。</span></div>}</div>
        </section>
        </div>
        </section>
      </aside>
    </div>

  </section>
}

function InstanceExecutionControls({
  item,
  busy,
  onSave,
}: {
  item: Json
  busy: boolean
  onSave: (
    item: Json,
    priority: number,
    segmentCaseLimit: number | null,
    caseCompletionTarget: number | null,
  ) => void
}) {
  const storedLimit = item.execution?.next_segment_case_limit
    ?? item.execution?.segment_case_limit
    ?? null
  const storedTarget = item.execution?.case_completion_target ?? null
  const [priority, setPriority] = useState(Number(item.queue?.priority ?? 100))
  const [segmentCaseLimit, setSegmentCaseLimit] = useState<number | null>(storedLimit)
  const [caseCompletionTarget, setCaseCompletionTarget] = useState<number | null>(storedTarget)
  useEffect(() => {
    setPriority(Number(item.queue?.priority ?? 100))
    setSegmentCaseLimit(
      item.execution?.next_segment_case_limit ?? item.execution?.segment_case_limit ?? null,
    )
    setCaseCompletionTarget(item.execution?.case_completion_target ?? null)
  }, [
    item.queue?.priority,
    item.execution?.next_segment_case_limit,
    item.execution?.segment_case_limit,
    item.execution?.case_completion_target,
  ])
  const dirty = priority !== Number(item.queue?.priority ?? 100)
    || segmentCaseLimit !== storedLimit
    || caseCompletionTarget !== storedTarget
  return <div className="experiment-instance-control-editor">
    <label><span>优先级</span><input type="number" value={priority} onChange={(event) => setPriority(Number(event.target.value))} /></label>
    <label><span>下一执行段新增 Case</span><div className="runtime-limit-control"><input type="number" min="1" disabled={segmentCaseLimit == null} value={segmentCaseLimit ?? ''} placeholder="不限制" onChange={(event) => setSegmentCaseLimit(Math.max(1, Number(event.target.value || 1)))} /><label className="check-field"><input type="checkbox" checked={segmentCaseLimit == null} onChange={(event) => setSegmentCaseLimit(event.target.checked ? null : 50)} /><span>不限制</span></label></div></label>
    <label><span>累计完成 Case 上限</span><div className="runtime-limit-control"><input type="number" min="1" disabled={caseCompletionTarget == null} value={caseCompletionTarget ?? ''} placeholder="不限制" onChange={(event) => setCaseCompletionTarget(Math.max(1, Number(event.target.value || 1)))} /><label className="check-field"><input type="checkbox" checked={caseCompletionTarget == null} onChange={(event) => setCaseCompletionTarget(event.target.checked ? null : 50)} /><span>不限制</span></label></div></label>
    <button className="icon-button" title="保存队列优先级与 Case 调度目标" disabled={busy || !dirty} onClick={() => onSave(item, priority, segmentCaseLimit, caseCompletionTarget)}><Save size={14} /></button>
  </div>
}

function ExperimentProgress({ progress }: { progress: Json }) {
  const fullPercent = Math.max(0, Math.min(100, Number(progress.percent || 0)))
  const completed = progress.completed_trials
  const expected = progress.expected_trials
  const completedCases = Number(progress.completed_distinct_cases || 0)
  const caseTarget = progress.case_completion_target == null
    ? null
    : Number(progress.case_completion_target)
  const percent = caseTarget && caseTarget > 0
    ? Math.max(0, Math.min(100, (completedCases / caseTarget) * 100))
    : fullPercent
  const currentCases: Json[] = progress.current_cases || []
  const activity = progress.current_activity
  const lines: string[] = progress.lines || []
  const stage = activity
    ? `${activity.role} · ${activity.phase} · turn ${activity.turn}`
    : progress.run_status || progress.state?.status || 'pending'
  return <section className="experiment-progress" aria-label="运行进度">
    {progress.observation_status === 'unavailable' && <div className="experiment-instance-error"><CircleAlert size={14} /><span>{progress.message || '暂时无法读取运行进度；实验生命周期状态未改变。'}</span></div>}
    <div className="experiment-progress-heading">
      <strong>{caseTarget ? `累计 Case ${completedCases} / ${caseTarget}` : (completed !== null && completed !== undefined && expected ? `${completed} / ${expected}` : stage)}</strong>
      <span>{caseTarget && expected ? `全量 Trial ${completed ?? 0} / ${expected} · ` : ''}{percent.toFixed(percent < 10 && percent > 0 ? 1 : 0)}%</span>
    </div>
    <div className="experiment-progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div>
    <div className="experiment-progress-metrics">
      <span>成功 {progress.successful_trials ?? '—'}</span>
      <span>失败 {progress.failed_trials ?? '—'}</span>
      <span>剩余 {progress.remaining_trials ?? '—'}</span>
      <span>全量范围 {completed !== null && completed !== undefined && expected ? `${completed} / ${expected} Trial` : '—'}</span>
      <span>当前 Trial 并发上限 {progress.trial_concurrency ?? progress.case_concurrency ?? '—'}</span>
      <span>耗时 {duration(progress.elapsed_s)}</span>
      <span>预计总计 {duration(progress.estimated_total_active_s)}</span>
      <span>预计剩余 {duration(progress.estimated_remaining_active_s)}</span>
    </div>
    {progress.drain_requested && <div className="experiment-drain-status"><Pause size={14} /><span>正在安全停止：不再启动新 Trial，等待当前 {currentCases.length} 个执行中 Trial 完成</span></div>}
    <div className="experiment-current-work">
      <strong>{stage}</strong>
      {currentCases.length > 0
        ? <div className="experiment-active-trials">{currentCases.map((item) => <span key={`${item.worker_id ?? 'log'}-${item.case_order}-${item.trial_index ?? 0}`}>#{item.case_order}/{item.num_cases} · {item.case_id}{item.trials_per_case > 1 ? ` · k=${Number(item.trial_index || 0) + 1}/${item.trials_per_case}` : ''}</span>)}</div>
        : <span>{progress.state?.status === 'running' ? '正在等待下一条可观测事件' : progress.run_status || progress.state?.status}</span>}
    </div>
    {progress.message && <small className="experiment-progress-message" title={progress.message}>{progress.message}</small>}
    {lines.length > 0 && <details className="experiment-log-tail"><summary>最近日志 · {lines.length} 行</summary><pre>{lines.join('\n')}</pre></details>}
  </section>
}
