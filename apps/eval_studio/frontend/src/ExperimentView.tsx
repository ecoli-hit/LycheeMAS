import { Check, CircleAlert, Clipboard, Folder, Infinity, ListMinus, ListPlus, Play, Plus, RefreshCw, RotateCcw, Save, Square, Trash2, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import SpecSelector from './SpecSelector'
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

const duration = (seconds: unknown) => {
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
  const [maxParallelInstances, setMaxParallelInstances] = useState(
    Number(bootstrap.experiments?.queue?.max_parallel_instances || 8),
  )
  const [launcherType, setLauncherType] = useState('subprocess')
  const [tmuxSession, setTmuxSession] = useState('lychee-eval')
  const [tmuxWindow, setTmuxWindow] = useState('benchmark')
  const [plan, setPlan] = useState<PlanResponse | null>(null)
  const [planInstanceId, setPlanInstanceId] = useState('')
  const [busy, setBusy] = useState(false)
  const pollInFlight = useRef(false)
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
  const groupChatType = String(teamSpec?.group_chat?.type || 'round_robin')
  const participantCount = Math.max(1, Number(teamSpec?.participants?.length || 1))
  const isRoundRobin = groupChatType === 'round_robin'
  const effectiveMaxTurns = isRoundRobin
    ? participantCount * Math.max(1, Number(draft.runtime?.max_rounds || 4))
    : Math.max(1, Number(draft.runtime?.max_turns || 20))
  const modelCallsUnlimited = draft.runtime?.max_model_calls_per_case == null
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
  const orphanedLaunches: Json[] = catalog.orphans || []
  const queueIsActive = Boolean(
    catalog.queue?.running
    || (catalog.instances || []).some((item: Json) => item.status === 'running')
    || orphanedLaunches.length > 0
  )
  const environmentUnavailable = environment.exists === false
  const planCommand = plan?.files['run_all.sh'] || plan?.commands?.subprocess || ''

  useEffect(() => {
    if (!queueIsActive) return undefined
    let cancelled = false
    const poll = async () => {
      if (pollInFlight.current) return
      pollInFlight.current = true
      try {
        const [instances, queue, orphans] = await Promise.all([
          api.experimentInstances(),
          api.experimentQueue(),
          api.experimentOrphans(),
        ])
        if (!cancelled) setCatalog((current: Json) => ({ ...current, instances, queue, orphans }))
      } catch {
        // The next manual refresh will surface persistent connectivity errors.
      } finally {
        pollInFlight.current = false
      }
    }
    const timer = window.setInterval(() => void poll(), 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [queueIsActive])

  const refresh = async () => {
    const [specs, instances, queue, orphans] = await Promise.all([
      api.experimentSpecs(),
      api.experimentInstances(),
      api.experimentQueue(),
      api.experimentOrphans(),
    ])
    setCatalog({ specs, instances, queue, orphans })
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

  const stopInstance = async (item: Json) => {
    try {
      await api.stopExperimentInstance(item.id)
      await refresh()
      notify(`ExperimentInstance ${item.id} 已停止`)
    } catch (cause) { notifyError(cause) }
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
      const result = await api.startExperimentQueue(maxParallelInstances)
      await refresh()
      const accepted = result.accepted_instance_ids || []
      notify(`调度器已接收 ${accepted.length} 个 ExperimentInstance`)
    } catch (cause) {
      notifyError(cause)
    } finally {
      setBusy(false)
    }
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
    <div className="experiment-workspace management-workspace">
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
          <section className="management-editor-section">
            <div className="panel-heading"><h2>实验定义</h2></div>
        <label><span>ExperimentSpec ID</span><input value={draft.id || ''} onChange={(event) => setDraft({ ...draft, id: event.target.value })} /></label>
        <label><span>BenchmarkSpec</span><select value={draft.benchmark?.benchmark_spec_id || ''} onChange={(event) => chooseBenchmarkSpec(event.target.value)}>{benchmarkSpecs.map((item) => <option key={item.id} value={item.id}>{item.category} · {item.name}</option>)}</select></label>
        <label><span>Runnable task</span><select value={draft.benchmark?.runnable_task || ''} onChange={(event) => chooseTask(event.target.value)}>{(benchmarkSpec?.runnable_tasks || []).map((task: string) => <option key={task} value={task}>{task}</option>)}</select></label>
        <div className="two-fields">
          <label><span>Cases</span><div className="input-with-action"><input value={Number.isNaN(draft.benchmark?.cases) ? '' : (draft.benchmark?.cases ?? 10)} placeholder="正整数或 all" spellCheck={false} onChange={(event) => updateSection('benchmark', { cases: event.target.value.toLowerCase() })} /><button className="icon-button" title="运行当前 task 的全部 case" onClick={() => updateSection('benchmark', { cases: 'all' })}><Infinity size={15} /></button></div><small>输入正整数，或输入 all；右侧按钮可直接设为全部。</small></label>
          <label><span>Start index</span><input type="number" min="0" value={draft.benchmark?.start_index || 0} onChange={(event) => updateSection('benchmark', { start_index: Number(event.target.value) })} /></label>
        </div>
        <label><span>Scoring profile</span><select value={draft.benchmark?.scoring_profile || scoring.default_profile || ''} onChange={(event) => updateSection('benchmark', { scoring_profile: event.target.value })}>{scoringProfiles.map((profile) => <option key={profile} value={profile}>{profile} · {scoring.profiles[profile].scorer_id}</option>)}</select></label>
        <label><span>TeamSpec</span><select value={draft.team_spec_id || ''} onChange={(event) => { setDraft({ ...draft, team_spec_id: event.target.value }); setTeamInstanceId('') }}>{teamSpecs.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.group_chat?.type}</option>)}</select></label>
        <DetailsDisclosure value={draft} label="展开 ExperimentSpec" />
          </section>
          <section className="management-editor-section">
            <div className="panel-heading"><h2>运行参数与输出</h2></div>
            <div className="two-fields">
              <label><span>Method</span><select value={draft.runtime?.method || 'none'} onChange={(event) => updateSection('runtime', { method: event.target.value })}><option value="none">none</option><option value="nl_only">nl_only</option><option value="latent_only">latent_only</option><option value="both">both</option></select></label>
              <label><span>Samples / case</span><input type="number" min="1" step="1" value={draft.runtime?.samples ?? 1} onChange={(event) => updateSection('runtime', { samples: Number(event.target.value) })} /><small>每个 case 的独立 prediction 数；用于 pass@K、自一致性或重复采样，常规 accuracy 保持 1。</small></label>
              <label><span>Case concurrency mode</span><select value={draft.runtime?.concurrency_policy?.mode || 'fixed'} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), mode: event.target.value } })}><option value="fixed">Fixed · 固定</option><option value="auto">Auto · AIMD 自适应准入</option></select><small>只调节同时进入 case 推理的数量，不会运行中重启或修改 vLLM max_num_seqs。</small></label>
              {draft.runtime?.concurrency_policy?.mode !== 'auto'
                ? <label><span>Case concurrency</span><input type="number" min="1" value={draft.runtime?.case_concurrency || 1} onChange={(event) => updateSection('runtime', { case_concurrency: Number(event.target.value), concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), mode: 'fixed', initial: Number(event.target.value), minimum: Number(event.target.value), maximum: Number(event.target.value) } })} /></label>
                : <><label><span>Initial case concurrency</span><input type="number" min="1" value={draft.runtime?.concurrency_policy?.initial || 1} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), initial: Number(event.target.value) } })} /></label><label><span>Minimum concurrency</span><input type="number" min="1" value={draft.runtime?.concurrency_policy?.minimum || 1} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), minimum: Number(event.target.value) } })} /></label><label><span>Maximum concurrency</span><input type="number" min="1" value={draft.runtime?.concurrency_policy?.maximum || draft.runtime?.case_concurrency || 1} onChange={(event) => updateSection('runtime', { case_concurrency: Number(event.target.value), concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), maximum: Number(event.target.value) } })} /></label><label><span>Increase step</span><input type="number" min="1" value={draft.runtime?.concurrency_policy?.increase_step || 1} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), increase_step: Number(event.target.value) } })} /></label><label><span>Decrease factor</span><input type="number" min="0.1" max="0.9" step="0.1" value={draft.runtime?.concurrency_policy?.decrease_factor || 0.5} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), decrease_factor: Number(event.target.value) } })} /></label><label><span>Control window (cases)</span><input type="number" min="1" value={draft.runtime?.concurrency_policy?.control_window_cases || 8} onChange={(event) => updateSection('runtime', { concurrency_policy: { ...(draft.runtime?.concurrency_policy || {}), control_window_cases: Number(event.target.value) } })} /></label></>}
              <label><span>On case error</span><select value={draft.runtime?.on_case_error || 'continue'} onChange={(event) => updateSection('runtime', { on_case_error: event.target.value })}><option value="continue">continue · 记录错误并继续</option><option value="fail-fast">fail-fast · 立即停止 run</option></select><small>同一 (case_id, k_index) 的全部 attempt 都失败后，决定是否继续后续 prediction。</small></label>
              <label><span>Max case retries</span><input type="number" min="0" step="1" value={draft.runtime?.max_case_retries ?? 0} onChange={(event) => updateSection('runtime', { max_case_retries: Math.max(0, Number(event.target.value || 0)) })} /><small>runtime.run() 抛异常后的额外整轨迹重试次数；0 表示不重试，错误答案不会触发重试。</small></label>
              {isRoundRobin
                ? <><label><span>Max rounds</span><input type="number" min="1" value={draft.runtime?.max_rounds || 4} onChange={(event) => updateSection('runtime', { max_rounds: Number(event.target.value) })} /><small>RoundRobin 的完整轮数；每轮包含 {participantCount} 个 participant turn。</small></label><label><span>Max turns（自动换算）</span><input value={effectiveMaxTurns} readOnly /><small>{participantCount} 个角色 × {draft.runtime?.max_rounds || 4} 轮；实际传给 AutoGen。</small></label></>
                : <><label><span>Max turns</span><input type="number" min="1" value={draft.runtime?.max_turns || 20} onChange={(event) => updateSection('runtime', { max_turns: Number(event.target.value) })} /><small>{groupChatType === 'magentic_one' ? 'Orchestrator' : 'Selector'} 动态选人时的 participant 工作步数上限。</small></label><label><span>Max rounds</span><input value="不适用" readOnly /><small>{groupChatType} 不存在“所有角色依次发言一遍”的固定轮次。</small></label></>}
              <div className="runtime-limit-field"><span>Max model calls / case</span><div className="runtime-limit-control"><input type="number" min="1" disabled={modelCallsUnlimited} value={modelCallsUnlimited ? '' : draft.runtime?.max_model_calls_per_case} placeholder="不限制" onChange={(event) => updateSection('runtime', { max_model_calls_per_case: Math.max(1, Number(event.target.value || 1)) })} /><label className="check-field"><input type="checkbox" checked={modelCallsUnlimited} onChange={(event) => updateSection('runtime', { max_model_calls_per_case: event.target.checked ? null : 40 })} /><span>不限制</span></label></div><small>统计 participant、Selector 和 Orchestrator 发出的所有真实模型请求；达到后终止当前 case。</small></div>
              <label><span>Max new tokens</span><input type="number" min="1" value={draft.runtime?.max_new_tokens || 16384} onChange={(event) => updateSection('runtime', { max_new_tokens: Number(event.target.value) })} /></label>
              <label><span>Max input tokens</span><input type="number" min="1" value={draft.runtime?.max_input_tokens ?? ''} placeholder="自动按模型窗口分配" onChange={(event) => updateSection('runtime', { max_input_tokens: event.target.value ? Number(event.target.value) : null })} /></label>
              <label><span>Min output reserve</span><input type="number" min="0" value={draft.runtime?.min_output_reserve_tokens ?? 2048} onChange={(event) => updateSection('runtime', { min_output_reserve_tokens: Number(event.target.value) })} /><small>自适应裁剪输入时至少为本次模型输出保留的 token。</small></label>
              <label><span>Min thinking reserve</span><input type="number" min="0" value={draft.runtime?.min_thinking_reserve_tokens ?? 0} onChange={(event) => updateSection('runtime', { min_thinking_reserve_tokens: Number(event.target.value) })} /></label>
              <label><span>Max thinking budget</span><input type="number" min="1" value={draft.runtime?.max_thinking_budget_tokens ?? ''} placeholder="不单独限制" onChange={(event) => updateSection('runtime', { max_thinking_budget_tokens: event.target.value ? Number(event.target.value) : null })} /></label>
              <label><span>Min final reserve</span><input type="number" min="0" value={draft.runtime?.min_final_reserve_tokens ?? 1024} onChange={(event) => updateSection('runtime', { min_final_reserve_tokens: Number(event.target.value) })} /></label>
              <label><span>Safety margin</span><input type="number" min="0" value={draft.runtime?.safety_margin_tokens ?? 256} onChange={(event) => updateSection('runtime', { safety_margin_tokens: Number(event.target.value) })} /></label>
              <label><span>Sampling</span><select value={draft.runtime?.do_sample ? 'enabled' : 'disabled'} onChange={(event) => updateSection('runtime', { do_sample: event.target.value === 'enabled' })}><option value="disabled">Disabled · deterministic</option><option value="enabled">Enabled · stochastic</option></select><small>关闭表示不随机采样；当前 vLLM/API 映射为 temperature=0、top_p=1，其他确定性搜索仍取决于后端能力。</small></label>
              <label><span>Temperature</span><input type="number" min="0" step="0.05" value={draft.runtime?.temperature ?? 0.7} onChange={(event) => updateSection('runtime', { temperature: Number(event.target.value) })} /></label>
              <label><span>Top P</span><input type="number" min="0.01" max="1" step="0.05" value={draft.runtime?.top_p ?? 0.8} onChange={(event) => updateSection('runtime', { top_p: Number(event.target.value) })} /></label>
              <label><span>Top K</span><input type="number" min="1" step="1" value={draft.runtime?.top_k ?? ''} placeholder="Provider 默认" onChange={(event) => updateSection('runtime', { top_k: event.target.value ? Number(event.target.value) : null })} /><small>采样时最多保留概率最高的 K 个候选 token。</small></label>
              <label><span>Min P</span><input type="number" min="0" max="1" step="0.05" value={draft.runtime?.min_p ?? ''} placeholder="Provider 默认" onChange={(event) => updateSection('runtime', { min_p: event.target.value ? Number(event.target.value) : null })} /><small>过滤相对概率过低的候选；0 表示不额外过滤。</small></label>
              <label><span>Presence penalty</span><input type="number" min="-2" max="2" step="0.1" value={draft.runtime?.presence_penalty ?? ''} placeholder="Provider 默认" onChange={(event) => updateSection('runtime', { presence_penalty: event.target.value ? Number(event.target.value) : null })} /><small>对已经生成过的 token 施加固定惩罚；0 表示关闭。</small></label>
              <label><span>Repetition penalty</span><input type="number" min="0.01" step="0.05" value={draft.runtime?.repetition_penalty ?? 1} onChange={(event) => updateSection('runtime', { repetition_penalty: Number(event.target.value) })} /><small>乘法式重复惩罚；1 表示关闭。</small></label>
              <label><span>Base seed</span><input type="number" step="1" value={draft.runtime?.seed ?? 0} onChange={(event) => updateSection('runtime', { seed: Number(event.target.value) })} /><small>实际 prediction seed 由该值、Benchmark/Task、稳定 Case ID 和 k_index 派生；数据重排、并发、恢复和重试不会改变它。</small></label>
              <label><span>Model call console trace</span><select value={draft.runtime?.trace_model_calls === false ? 'disabled' : 'enabled'} onChange={(event) => updateSection('runtime', { trace_model_calls: event.target.value === 'enabled' })}><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select><small>只控制终端 start/done 摘要；spans 仍持续记录。</small></label>
              <label><span>Trace detail</span><select value={draft.runtime?.trace_detail_level || 'compact'} onChange={(event) => updateSection('runtime', { trace_detail_level: event.target.value })}><option value="compact">compact</option><option value="full">full</option></select></label>
              <label><span>Code executor</span><select value={draft.runtime?.code_executor || 'docker'} onChange={(event) => updateSection('runtime', { code_executor: event.target.value })}><option value="docker">Docker sandbox</option><option value="local">Local process</option></select><small>正式工具评测建议使用 Docker；local 会直接在宿主环境执行代码。</small></label>
              <label><span>Code timeout (s)</span><input type="number" min="1" value={draft.runtime?.code_timeout ?? 60} onChange={(event) => updateSection('runtime', { code_timeout: Number(event.target.value) })} /><small>单个代码块的执行上限；60 秒与 AgBench GAIA 模板使用的 AutoGen 默认值一致。</small></label>
              <label><span>Tool workspace root</span><input value={draft.runtime?.work_root || 'runs/lychee_tool_workspaces'} onChange={(event) => updateSection('runtime', { work_root: event.target.value })} /></label>
              <label><span>Browser mode</span><select value={draft.runtime?.web_headless === false ? 'visible' : 'headless'} onChange={(event) => updateSection('runtime', { web_headless: event.target.value === 'headless' })}><option value="headless">Headless</option><option value="visible">Visible</option></select></label>
              <label><span>Browser screenshots</span><select value={draft.runtime?.save_screenshots ? 'enabled' : 'disabled'} onChange={(event) => updateSection('runtime', { save_screenshots: event.target.value === 'enabled' })}><option value="disabled">Disabled</option><option value="enabled">Enabled</option></select></label>
              <label><span>vLLM service metrics</span><select value={draft.observability?.collect_vllm_metrics === false ? 'disabled' : 'enabled'} onChange={(event) => updateSection('observability', { collect_vllm_metrics: event.target.value === 'enabled' })}><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select><small>定时读取每个 vLLM DeploymentInstance 的 /metrics；API 与 Local HF 没有对应目标时不会伪造数据。</small></label>
              <label><span>vLLM metrics interval (s)</span><input type="number" min="0.5" step="0.5" value={draft.observability?.vllm_metrics_interval_s ?? 5} onChange={(event) => updateSection('observability', { vllm_metrics_interval_s: Number(event.target.value) })} /><small>服务级时间序列采样间隔；通常 5 秒足够观察排队和 KV cache 压力。</small></label>
              {sandboxProfiles.length > 0 && <label><span>Sandbox profile</span><select value={draft.runtime?.docker_image || benchmarkSpec?.runtime_defaults?.docker_image || ''} onChange={(event) => updateSection('runtime', { docker_image: event.target.value })}>{sandboxProfiles.map(([profileId, profile]) => <option key={profileId} value={profile.docker_image}>{profileId} · {profile.label}</option>)}</select><small>{selectedSandboxProfile?.[1]?.description || '选择该 benchmark 的代码执行环境；具体镜像会写入运行配置和追踪记录。'}</small></label>}
            </div>
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
            {draft.network?.mode === 'proxy' && <><label><span>Proxy URL</span><input value={draft.network?.proxy_url || ''} onChange={(event) => updateSection('network', { proxy_url: event.target.value })} /></label><div className="network-target-controls"><strong>代理应用目标</strong>{[['web_surfer', 'WebSurfer'], ['code_executor', 'Docker code executor'], ['model_backend', 'Model API backend']].map(([target, label]) => <label key={target} className="check-field"><input type="checkbox" checked={Boolean(draft.network?.targets?.[target])} onChange={(event) => patchNetworkTarget(target, event.target.checked)} /><span>{label}</span></label>)}</div></>}
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
        {environmentUnavailable && <div className="experiment-environment-warning"><CircleAlert size={15} /><span>当前实验环境不可用：{environment.python}。请先到“运行环境”选择可用环境并设为实验环境。</span></div>}
        <button className="primary" disabled={!canAssemble || busy} onClick={instantiate}><Check size={15} />实例化</button>
        </section>
        <section className="management-instance-registry">
        <div className="experiment-queue-toolbar"><div><h2>ExperimentInstance</h2><span>Scheduler · 容量感知并行 · {catalog.queue?.running ? 'running' : 'stopped'} · running={runningCount} · queued={queuedCount} · orphaned={orphanedLaunches.length}</span></div><div className="toolbar-actions"><label className="queue-parallel-limit"><span>实例上限</span><input type="number" min="1" max="64" value={maxParallelInstances} disabled={Boolean(catalog.queue?.running)} onChange={(event) => setMaxParallelInstances(Math.max(1, Math.min(64, Number(event.target.value) || 1)))} /></label><button className="secondary" title={queuedCount ? '按优先级和 DeploymentInstance 剩余容量动态并行运行' : '请先将 ready 实例加入队列'} disabled={catalog.queue?.running || queuedCount === 0 || busy} onClick={startQueue}><Play size={14} />启动调度器</button><button className="secondary" title="不再启动新实例；已经运行的实例继续完成" disabled={!catalog.queue?.running || busy} onClick={pauseQueue}><Square size={14} />停止接纳</button></div></div>
        <div className="queue-summary"><span>active={(catalog.queue?.active_instance_ids || []).join(', ') || 'none'}</span><span>queued={(catalog.queue?.queued_instance_ids || []).join(', ') || 'none'}</span>{Object.entries(catalog.queue?.resource_usage || {}).map(([deploymentId, usage]: [string, any]) => <span key={deploymentId}>{deploymentId}: {usage.used}/{usage.capacity} slots</span>)}</div>
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
        <div className="experiment-instance-list">{(catalog.instances || []).map((item: Json) => <article className={`experiment-instance-card ${plan && planInstanceId === item.id ? 'command-open' : ''}`} key={item.id}>
          <div className="experiment-instance-card-main">
            <header><span className={`status-pill ${item.progress?.run_status === 'complete_with_errors' ? 'failed' : item.status}`}>{item.progress?.run_status === 'complete_with_errors' ? 'complete_with_errors' : item.status}</span><strong title={item.id}>{item.id}</strong></header>
            <dl className="experiment-instance-meta"><dt>ExperimentSpec</dt><dd>{item.experiment_spec_id}</dd><dt>Launcher</dt><dd>{launcherLabels[item.launcher?.type] || item.launcher?.type}</dd><dt>BenchmarkInstance</dt><dd>{item.benchmark_instance_id}</dd><dt>TeamInstance</dt><dd>{item.team_instance_id}</dd></dl>
            {item.run_dir && <code className="experiment-instance-path" title={item.run_dir}>{item.run_dir}</code>}
            {item.progress && <ExperimentProgress progress={item.progress} />}
            {item.error && <div className="experiment-instance-error"><CircleAlert size={14} /><span>{item.error}</span></div>}
            <DetailsDisclosure value={item} label="展开 ExperimentInstance" className="card-details" />
          </div>
          <div className="row-actions experiment-instance-actions"><button className="icon-button" title={plan && planInstanceId === item.id ? '收起完整命令' : '生成完整命令'} disabled={busy} onClick={() => previewInstance(item)}><Clipboard size={14} /></button>{item.status === 'ready' && <><button className="icon-button" title="立即运行" disabled={busy} onClick={() => launchInstance(item)}><Play size={14} /></button><button className="icon-button" title="加入队列" onClick={() => enqueueInstance(item)}><ListPlus size={14} /></button></>}{item.status === 'queued' && <button className="icon-button" title="移出队列" onClick={() => dequeueInstance(item)}><ListMinus size={14} /></button>}{item.status === 'running' && <button className="icon-button danger" title="停止运行" onClick={() => stopInstance(item)}><Square size={14} /></button>}{item.status !== 'running' && item.status !== 'queued' && <button className="icon-button danger" title="删除 ExperimentInstance" onClick={() => removeInstance(item)}><Trash2 size={14} /></button>}</div>
          {plan && planInstanceId === item.id && <section className="experiment-instance-command">
            <header>
              <div><strong>完整运行命令</strong><span title={plan.run_dir}>运行数据：{plan.run_dir}</span></div>
              <div className="row-actions"><button className="icon-button" title="复制完整运行命令" disabled={!planCommand} onClick={copyPlanCommand}><Clipboard size={14} /></button><button className="icon-button" title="收起完整运行命令" onClick={closePlan}><X size={15} /></button></div>
            </header>
            <pre><code>{planCommand}</code></pre>
          </section>}
        </article>)}</div>
        </section>
      </aside>
    </div>

  </section>
}

function ExperimentProgress({ progress }: { progress: Json }) {
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)))
  const completed = progress.completed_predictions
  const expected = progress.expected_predictions
  const currentCases: Json[] = progress.current_cases || []
  const activity = progress.current_activity
  const lines: string[] = progress.lines || []
  const stage = activity
    ? `${activity.role} · ${activity.phase} · turn ${activity.turn}`
    : progress.run_status || progress.state?.status || 'pending'
  return <section className="experiment-progress" aria-label="运行进度">
    {progress.observation_status === 'unavailable' && <div className="experiment-instance-error"><CircleAlert size={14} /><span>{progress.message || '暂时无法读取运行进度；实验生命周期状态未改变。'}</span></div>}
    <div className="experiment-progress-heading">
      <strong>{completed !== null && completed !== undefined && expected ? `${completed} / ${expected}` : stage}</strong>
      <span>{percent}%</span>
    </div>
    <div className="experiment-progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div>
    <div className="experiment-progress-metrics">
      <span>成功 {progress.successful_predictions ?? '—'}</span>
      <span>错误 {progress.error_predictions ?? '—'}</span>
      <span>剩余 {progress.remaining_predictions ?? '—'}</span>
      <span>并发 {progress.current_case_concurrency ?? progress.case_concurrency ?? '—'}</span>
      <span>耗时 {duration(progress.elapsed_s)}</span>
    </div>
    <div className="experiment-current-work">
      <strong>{stage}</strong>
      {currentCases.length > 0
        ? currentCases.map((item) => <span key={`${item.worker_id ?? 'log'}-${item.case_order}-${item.k_index ?? 0}`}>#{item.case_order}/{item.num_cases} · {item.case_id}{item.samples_per_case > 1 ? ` · k=${Number(item.k_index || 0) + 1}/${item.samples_per_case}` : ''}</span>)
        : <span>{progress.state?.status === 'running' ? '正在等待下一条可观测事件' : progress.run_status || progress.state?.status}</span>}
    </div>
    {progress.message && <small className="experiment-progress-message" title={progress.message}>{progress.message}</small>}
    {lines.length > 0 && <details className="experiment-log-tail"><summary>最近日志 · {lines.length} 行</summary><pre>{lines.join('\n')}</pre></details>}
  </section>
}
