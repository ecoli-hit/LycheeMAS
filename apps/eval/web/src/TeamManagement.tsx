import { Boxes, Plus, RefreshCw, Save, Search, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import SpecSelector from './SpecSelector'
import TeamGraphCanvas from './TeamGraphCanvas'
import { deriveCoordinationSummary } from './teamGraph'
import type { Bootstrap, Json } from './types'

interface Props { bootstrap: Bootstrap; notify: (message: string) => void }

const FRAMEWORKS = ['autogen', 'langgraph', 'crewai'] as const
const NODE_KINDS = ['model_agent', 'tool_executor', 'function', 'human', 'team', 'remote']
const BEHAVIORS = ['assistant', 'selector', 'code_author', 'code_executor', 'file_navigator', 'web_navigator', 'custom']
const CONTROL_TRIGGERS = ['trial_started', 'completed', 'failed', 'output_emitted', 'result_submitted', 'always']
const CONTROL_ACTIONS = ['activate', 'select_next', 'delegate', 'handoff', 'return', 'finish', 'cancel']
const STATE_KINDS = ['message_channel', 'structured_state', 'artifact_store', 'memory']
const STATE_UPDATES = ['append', 'replace', 'merge', 'commit']
const DATA_VIEWS = ['all', 'latest', 'latest_n', 'from_source', 'summary']

const splitList = (value: string): string[] => value.split(',').map((item) => item.trim()).filter(Boolean)
const nextId = (items: Json[], prefix: string): string => {
  const ids = new Set(items.map((item) => String(item.id)))
  let index = items.length + 1
  while (ids.has(`${prefix}_${index}`)) index += 1
  return `${prefix}_${index}`
}

const defaultNode = (id: string): Json => ({
  id, name: id, kind: 'model_agent', behavior: { type: 'assistant', options: {} },
  purpose: id, instructions: '', capabilities: ['reason'], operations: [], tools: [],
  context: { type: 'unbounded' }, limits: { max_tool_iterations: 1 },
})

const defaultTransfer = (): Json => ({
  source: 'trial.task', target: 'task', required: true, view: 'all', latest_n: null,
  filter: null, transform: null, schema: {},
})

const defaultRelation = (id: string, source: string, target: string): Json => ({
  id, from: source, to: target,
  control: {
    trigger: 'completed', action: 'activate', condition: null, priority: 0,
    on_failure: 'fail_trial', max_uses: null,
  },
  data: { transfers: [defaultTransfer()] },
})

const defaultState = (id: string): Json => ({
  id, kind: 'message_channel', description: '', readers: [], writers: [], update: 'append',
  lifetime: 'trial', initial: [], schema: {},
  retention: { max_items: null, max_tokens: null, overflow: 'drop_oldest' },
})

const fallbackTeam = (): Json => ({
  schema_version: 14,
  id: 'team',
  metadata: {
    name: 'team', description: '', tags: [],
    provenance: { track: 'custom', evidence_level: 'hypothesis', sources: [], notes: '' },
  },
  nodes: [defaultNode('Solver')],
  relations: [],
  shared_state: [{ ...defaultState('SharedConversation'), readers: ['Solver'], writers: ['Solver'] }],
  lifecycle: {
    entry: [{ node: 'Solver', inputs: [defaultTransfer()] }],
    result: { submissions: [{ from: 'Solver', source: 'source.output', key: 'final_answer' }], mode: 'first_valid', schema: {} },
    termination: { condition: 'result_submitted' },
    failure: { unhandled: 'fail_trial', deadlock: 'fail_trial' },
    limits: { max_turns: null, max_node_calls: null, max_stalls: null, timeout_seconds: null },
  },
})

const modelRequirements = (spec: Json): Json[] => (spec.nodes || [])
  .filter((node: Json) => node.kind === 'model_agent')
  .map((node: Json) => ({
    node_id: String(node.id), requirement: 'model_inference',
    required_capabilities: [
      'text_generation', ...(node.behavior?.type === 'web_navigator' ? ['vision'] : []),
    ],
  }))

const isDeploymentReady = (value: Json) => ['running', 'ready_on_run'].includes(
  String(value.observed_status || value.status || ''),
)

function JsonTextarea({ value, onChange, rows = 4 }: {
  value: Json; onChange: (value: Json) => void; rows?: number
}) {
  const canonical = JSON.stringify(value ?? {}, null, 2)
  const [text, setText] = useState(canonical)
  useEffect(() => setText(canonical), [canonical])
  return <textarea className="json-editor" rows={rows} value={text} onChange={(event) => {
    const next = event.target.value
    setText(next)
    try { onChange(JSON.parse(next)) } catch { /* keep editing until JSON is valid */ }
  }} />
}

function ModelContextEditor({ value, onChange }: { value: Json; onChange: (value: Json) => void }) {
  const type = value?.type || 'unbounded'
  return <div className="model-context-controls">
    <label><span>Model context</span><select value={type} onChange={(event) => onChange(
      event.target.value === 'buffered'
        ? { type: 'buffered', buffer_size: 12 }
        : event.target.value === 'token_limited'
          ? { type: 'token_limited' }
          : { type: 'unbounded' }
    )}><option value="unbounded">Unbounded</option><option value="buffered">Buffered</option><option value="token_limited">Token limited</option></select></label>
    {type === 'buffered' && <label><span>Buffer size</span><input type="number" min="1" value={value.buffer_size || 12} onChange={(event) => onChange({ type, buffer_size: Number(event.target.value) })} /></label>}
    {type === 'token_limited' && <label><span>Token limit</span><input type="number" min="1" value={value.token_limit || ''} onChange={(event) => onChange({ type, ...(event.target.value ? { token_limit: Number(event.target.value) } : {}) })} /></label>}
  </div>
}

function FrameworkBindingView({ framework, report, spec }: {
  framework: string; report: Json | null; spec: Json
}) {
  if (!report) return <div className="empty-state">正在编译 {framework} Binding Plan</div>
  const plan = report.adapter_plan || {}
  const frameworkName = framework === 'autogen' ? 'AutoGen' : framework === 'langgraph' ? 'LangGraph' : 'CrewAI'
  const runtimeCarrier = framework === 'autogen'
    ? 'AgentChat Team + Agent Runtime'
    : framework === 'langgraph'
      ? 'Compiled StateGraph'
      : String(plan.strategy || '').includes('crew')
        ? 'Crew + Process'
        : 'Flow + Crew/Task callables'
  return <section className="framework-binding-view">
    <header>
      <div><strong>{frameworkName}</strong><small>Concrete Team Binding</small></div>
      <span className={`status-pill ${report.supported ? 'running' : 'stopped'}`}>{report.mapping_level || 'unknown'}</span>
    </header>
    <div className={`framework-runtime-diagram framework-${framework}`}>
      <article className="framework-runtime-shell">
        <small>Framework runtime carrier</small>
        <strong>{plan.implementation || report.implementation || runtimeCarrier}</strong>
        <span>{runtimeCarrier}</span>
      </article>
      <div className="framework-runtime-arrow" aria-hidden="true">↓</div>
      <div className="framework-native-node-grid">
        {(report.node_bindings || []).map((binding: Json) => <article key={binding.node_id}>
          <strong>{binding.node_id}</strong>
          <span>{binding.specialization || binding.runtime_role}</span>
          <small title={binding.runtime_implementation}>{binding.runtime_implementation}</small>
          <em>{binding.mapping_level}</em>
        </article>)}
      </div>
      <div className="framework-runtime-services">
        <span><strong>Control</strong>{(spec.relations || []).filter((item: Json) => item.control).length} Relations · {plan.strategy || report.adapter_strategy}</span>
        <span><strong>Data</strong>{(spec.relations || []).filter((item: Json) => item.data).length} Relations · {(spec.shared_state || []).length} SharedState</span>
        <span><strong>Memory</strong>{(report.memory_bindings || []).length ? `${report.memory_bindings.length} bindings` : '未声明'}</span>
        <span><strong>Model I/O</strong>LycheeMAS ModelGateway</span>
      </div>
    </div>
    <div className="binding-overview-grid">
      <div><span>运行策略</span><strong>{plan.strategy || report.implementation || '—'}</strong></div>
      <div><span>原生承载</span><strong>{plan.native_component || report.implementation || '—'}</strong></div>
      <div><span>可控比较</span><strong>{report.controlled_comparison_eligible ? '是' : '否'}</strong></div>
      <div><span>语义状态</span><strong>{report.supported ? '可实例化' : '不可实例化'}</strong></div>
    </div>
    <h3>Node Bindings</h3>
    <div className="framework-binding-table">
      {(report.node_bindings || []).map((binding: Json) => <article key={binding.node_id}>
        <strong>{binding.node_id}</strong>
        <span>{binding.specialization || binding.runtime_role}</span>
        <small>{binding.runtime_implementation}</small>
        <em>{binding.mapping_level}</em>
      </article>)}
    </div>
    <h3>Coordination</h3>
    <div className="binding-detail-list">
      {(plan.operation_bindings || report.operation_bindings || []).map((item: Json, index: number) => <div key={`${item.operation}-${index}`}><strong>{item.operation}</strong><span>{item.node_id} → {(item.candidates || []).join(', ') || 'runtime'}</span></div>)}
      {!(plan.operation_bindings || report.operation_bindings || []).length && <span>由 {plan.strategy || 'framework adapter'} 承载显式 Control Relations。</span>}
    </div>
    <h3>Memory Bindings</h3>
    <div className="framework-binding-table">
      {(report.memory_bindings || []).map((binding: Json) => <article key={binding.memory_id}>
        <strong>{binding.memory_id}</strong><span>{binding.native_component}</span>
        <small>{binding.read_hook} / {binding.write_hook}</small><em>{binding.mapping_level}</em>
      </article>)}
      {!(report.memory_bindings || []).length && <span className="empty-inline">该 TeamSpec 未声明 Memory SharedState。</span>}
    </div>
    {!!(report.semantic_deltas || []).length && <div className="semantic-deltas"><strong>Semantic deltas</strong>{report.semantic_deltas.map((item: string) => <span key={item}>{item}</span>)}</div>}
    <DetailsDisclosure label={`展开 ${framework} Binding Report`} value={report} />
  </section>
}

export default function TeamManagement({ bootstrap, notify }: Props) {
  const fallback = useMemo(() => fallbackTeam(), [])
  const initial = bootstrap.teams.specs?.[0] || fallback
  const [workspace, setWorkspace] = useState<'spec' | 'instance'>('spec')
  const [specs, setSpecs] = useState<Json[]>(bootstrap.teams.specs || [])
  const [instances, setInstances] = useState<Json[]>(bootstrap.team_instances?.instances || [])
  const [deployments, setDeployments] = useState<Json[]>(bootstrap.deployments?.instances || [])
  const [draft, setDraft] = useState<Json>(structuredClone(initial))
  const [selectedNodeId, setSelectedNodeId] = useState(String(initial.nodes?.[0]?.id || ''))
  const [selectedRelationId, setSelectedRelationId] = useState(String(initial.relations?.[0]?.id || ''))
  const [selectedStateId, setSelectedStateId] = useState(String(initial.shared_state?.[0]?.id || ''))
  const [editorTab, setEditorTab] = useState<'relations' | 'state' | 'lifecycle'>('relations')
  const [graphView, setGraphView] = useState<'authoring' | typeof FRAMEWORKS[number]>('authoring')
  const [bindingReports, setBindingReports] = useState<Record<string, Json | null>>({})
  const [busy, setBusy] = useState(false)
  const [instanceTeamSpecId, setInstanceTeamSpecId] = useState(String(initial.id))
  const [instanceId, setInstanceId] = useState(`${initial.id}-instance`)
  const [runtimeFramework, setRuntimeFramework] = useState('autogen')
  const [bindings, setBindings] = useState<Json[]>([])
  const [instanceQuery, setInstanceQuery] = useState('')

  const selectedNode = (draft.nodes || []).find((node: Json) => node.id === selectedNodeId)
  const selectedRelation = (draft.relations || []).find((item: Json) => item.id === selectedRelationId)
  const selectedState = (draft.shared_state || []).find((item: Json) => item.id === selectedStateId)
  const canonicalTeam = specs.find((item) => item.id === instanceTeamSpecId) || specs[0] || fallback
  const requirements = modelRequirements(canonicalTeam)
  const summary = deriveCoordinationSummary(draft)
  const filteredInstances = instances.filter((item) => !instanceQuery.trim() || [item.id, item.team_spec_id, item.runtime_framework].some((value) => String(value || '').toLowerCase().includes(instanceQuery.trim().toLowerCase())))

  const refresh = async (probe = false) => {
    const [teamCatalog, teamInstances, deploymentCatalog] = await Promise.all([
      api.teamSpecs(), api.teamInstances(probe), api.deployments(probe),
    ])
    setSpecs(teamCatalog.specs || [])
    setInstances(teamInstances.instances || [])
    setDeployments(deploymentCatalog.instances || [])
  }

  const resetBindings = (spec: Json) => {
    const first = deployments.find(isDeploymentReady)?.id || ''
    setInstanceTeamSpecId(spec.id)
    setInstanceId(`${spec.id}-instance`)
    setBindings(modelRequirements(spec).map((requirement) => ({
      ...requirement, resource_instance_type: 'DeploymentInstance', resource_instance_id: first,
    })))
  }

  useEffect(() => { resetBindings(canonicalTeam) }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    let cancelled = false
    const timer = window.setTimeout(() => {
      Promise.all(FRAMEWORKS.map(async (framework) => {
        try { return [framework, await api.teamBindingReport(framework, draft)] as const }
        catch { return [framework, null] as const }
      })).then((rows) => { if (!cancelled) setBindingReports(Object.fromEntries(rows)) })
    }, 350)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [draft])

  const loadSpec = (spec: Json) => {
    const copy = structuredClone(spec)
    delete copy.registry_path
    delete copy.updated_at_utc
    setDraft(copy)
    setSelectedNodeId(String(copy.nodes?.[0]?.id || ''))
    setSelectedRelationId(String(copy.relations?.[0]?.id || ''))
    setSelectedStateId(String(copy.shared_state?.[0]?.id || ''))
    resetBindings(copy)
  }

  const saveSpec = async () => {
    setBusy(true)
    try {
      const saved = await api.saveTeamSpec(draft.id, { ...draft, schema_version: 14 })
      await refresh()
      loadSpec(saved)
      notify(`TeamSpec ${saved.id} 已保存`)
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setBusy(false) }
  }

  const removeSpec = async () => {
    if (!window.confirm(`确认删除 TeamSpec ${draft.id}？`)) return
    try { await api.deleteTeamSpec(draft.id); await refresh(); notify(`TeamSpec ${draft.id} 已删除`) }
    catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const patchNode = (patch: Json) => setDraft((current: Json) => ({
    ...current, nodes: current.nodes.map((node: Json) => node.id === selectedNodeId ? { ...node, ...patch } : node),
  }))
  const renameNode = (id: string) => setDraft((current: Json) => {
    const previous = selectedNodeId
    setSelectedNodeId(id)
    return {
      ...current,
      nodes: current.nodes.map((node: Json) => node.id === previous ? { ...node, id } : node),
      relations: current.relations.map((item: Json) => ({ ...item, from: item.from === previous ? id : item.from, to: item.to === previous ? id : item.to })),
      shared_state: current.shared_state.map((item: Json) => ({ ...item, readers: item.readers.map((value: string) => value === previous ? id : value), writers: item.writers.map((value: string) => value === previous ? id : value) })),
      lifecycle: {
        ...current.lifecycle,
        entry: current.lifecycle.entry.map((item: Json) => ({ ...item, node: item.node === previous ? id : item.node })),
        result: { ...current.lifecycle.result, submissions: current.lifecycle.result.submissions.map((item: Json) => ({ ...item, from: item.from === previous ? id : item.from })) },
      },
    }
  })
  const addNode = () => {
    const id = nextId(draft.nodes || [], 'Node')
    setDraft((current: Json) => ({ ...current, nodes: [...current.nodes, defaultNode(id)] }))
    setSelectedNodeId(id)
  }
  const removeNode = () => {
    if (!selectedNode || (draft.nodes || []).length <= 1) { notify('TeamSpec 至少需要一个 Node'); return }
    const replacement = (draft.nodes || []).find((node: Json) => node.id !== selectedNodeId)?.id
    setDraft((current: Json) => ({
      ...current,
      nodes: current.nodes.filter((node: Json) => node.id !== selectedNodeId),
      relations: current.relations.filter((item: Json) => item.from !== selectedNodeId && item.to !== selectedNodeId),
      shared_state: current.shared_state.map((item: Json) => ({ ...item, readers: item.readers.filter((id: string) => id !== selectedNodeId), writers: item.writers.filter((id: string) => id !== selectedNodeId) })),
      lifecycle: {
        ...current.lifecycle,
        entry: current.lifecycle.entry.filter((item: Json) => item.node !== selectedNodeId),
        result: { ...current.lifecycle.result, submissions: current.lifecycle.result.submissions.filter((item: Json) => item.from !== selectedNodeId) },
      },
    }))
    setSelectedNodeId(String(replacement || ''))
  }

  const patchRelation = (patch: Json) => setDraft((current: Json) => ({
    ...current, relations: current.relations.map((item: Json) => item.id === selectedRelationId ? { ...item, ...patch } : item),
  }))
  const addRelation = (source?: string, target?: string) => {
    const nodes = draft.nodes || []
    if (nodes.length < 2) { notify('至少需要两个 Node 才能创建 Relation'); return }
    const id = nextId(draft.relations || [], 'relation')
    setDraft((current: Json) => ({ ...current, relations: [...current.relations, defaultRelation(id, source || nodes[0].id, target || nodes[1].id)] }))
    setSelectedRelationId(id)
  }
  const removeRelation = () => {
    setDraft((current: Json) => ({ ...current, relations: current.relations.filter((item: Json) => item.id !== selectedRelationId) }))
    setSelectedRelationId(String((draft.relations || []).find((item: Json) => item.id !== selectedRelationId)?.id || ''))
  }

  const patchState = (patch: Json) => setDraft((current: Json) => ({
    ...current, shared_state: current.shared_state.map((item: Json) => item.id === selectedStateId ? { ...item, ...patch } : item),
  }))
  const addState = () => {
    const id = nextId(draft.shared_state || [], 'SharedState')
    setDraft((current: Json) => ({ ...current, shared_state: [...current.shared_state, defaultState(id)] }))
    setSelectedStateId(id)
  }
  const removeState = () => {
    setDraft((current: Json) => ({ ...current, shared_state: current.shared_state.filter((item: Json) => item.id !== selectedStateId) }))
    setSelectedStateId(String((draft.shared_state || []).find((item: Json) => item.id !== selectedStateId)?.id || ''))
  }

  const instantiate = async () => {
    setBusy(true)
    try {
      await api.saveTeamInstance(instanceId, {
        schema_version: 5, id: instanceId, team_spec_id: canonicalTeam.id,
        runtime_framework: runtimeFramework, framework_options: {},
        resource_bindings: bindings.map((item) => ({
          node_id: item.node_id, requirement: item.requirement,
          resource_instance_type: 'DeploymentInstance', resource_instance_id: item.resource_instance_id,
          ...(item.generation_overrides && Object.keys(item.generation_overrides).length ? { generation_overrides: item.generation_overrides } : {}),
        })),
      })
      await refresh(true)
      notify(`TeamInstance ${instanceId} 已实例化`)
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setBusy(false) }
  }

  return <section className="studio-view team-management">
    <nav className="experiment-workspace-tabs team-workspace-tabs" aria-label="团队管理工作区">
      <button className={workspace === 'spec' ? 'active' : ''} onClick={() => setWorkspace('spec')}><span>1</span><strong>TeamSpec</strong><small>框架无关团队语义</small></button>
      <button className={workspace === 'instance' ? 'active' : ''} onClick={() => setWorkspace('instance')}><span>2</span><strong>TeamInstance</strong><small>框架与部署绑定</small></button>
    </nav>
    {workspace === 'spec' ? <main className="team-spec-v14 stage-spec">
      <section className="team-spec-header management-spec-header">
        <div className="section-toolbar"><div><h1>TeamSpec v14</h1><span>Node、Relation、SharedState 与 Lifecycle 的唯一作者合同</span></div><div className="toolbar-actions">
          <button className="icon-button" title="新建 TeamSpec" onClick={() => loadSpec(fallbackTeam())}><Plus size={16} /></button>
          <button className="secondary" disabled={busy} onClick={saveSpec}><Save size={15} />保存</button>
          <button className="icon-button danger" title="删除 TeamSpec" onClick={removeSpec}><Trash2 size={15} /></button>
        </div></div>
        <SpecSelector kind="TeamSpec" specs={specs} draftId={draft.id} optionLabel={(item) => `${item.id} · ${deriveCoordinationSummary(item).topology}`} onLoad={loadSpec} />
        <div className="team-spec-fields">
          <label><span>TeamSpec ID</span><input value={draft.id} onChange={(event) => setDraft({ ...draft, id: event.target.value })} /></label>
          <label><span>Name</span><input value={draft.metadata?.name || ''} onChange={(event) => setDraft({ ...draft, metadata: { ...draft.metadata, name: event.target.value } })} /></label>
          <label><span>Description</span><input value={draft.metadata?.description || ''} onChange={(event) => setDraft({ ...draft, metadata: { ...draft.metadata, description: event.target.value } })} /></label>
        </div>
        <div className="team-contract-note"><strong>{summary.topology}</strong><span>{summary.label}</span><small>拓扑仅由显式 Node 操作与 Control Relations 推导，不是运行时分派开关。</small></div>
      </section>
      <section className="team-spec-graph-workspace">
        <aside className="left-panel team-node-panel">
          <div className="panel-heading"><h2>Node</h2><div className="toolbar-actions"><button className="icon-button" title="添加 Node" onClick={addNode}><Plus size={13} /></button><button className="danger icon-button" title="删除 Node" onClick={removeNode}><Trash2 size={13} /></button></div></div>
          <div className="role-list node-record-list node-list-scroll">{(draft.nodes || []).map((node: Json) => <button key={node.id} className={node.id === selectedNodeId ? 'selected' : ''} onClick={() => setSelectedNodeId(node.id)}><i className={`role-dot node-kind-${node.kind}`} /><span><strong>{node.name || node.id}</strong><small>{node.id} · {node.kind} · {node.behavior?.type}</small></span></button>)}</div>
          {selectedNode && <div className="node-inspector-form inspector-form node-detail-scroll">
            <label><span>Node ID</span><input value={selectedNode.id} onChange={(event) => renameNode(event.target.value)} /></label>
            <label><span>Name</span><input value={selectedNode.name || ''} onChange={(event) => patchNode({ name: event.target.value })} /></label>
            <label><span>Kind</span><select value={selectedNode.kind} onChange={(event) => patchNode({ kind: event.target.value })}>{NODE_KINDS.map((value) => <option key={value}>{value}</option>)}</select></label>
            <label><span>Behavior</span><select value={selectedNode.behavior?.type || 'assistant'} onChange={(event) => patchNode({ behavior: { ...selectedNode.behavior, type: event.target.value } })}>{BEHAVIORS.map((value) => <option key={value}>{value}</option>)}</select></label>
            <label><span>Purpose</span><textarea value={selectedNode.purpose || ''} onChange={(event) => patchNode({ purpose: event.target.value })} /></label>
            <label><span>Instructions</span><textarea rows={7} value={selectedNode.instructions || ''} onChange={(event) => patchNode({ instructions: event.target.value })} /></label>
            <label><span>Capabilities</span><input value={(selectedNode.capabilities || []).join(', ')} onChange={(event) => patchNode({ capabilities: splitList(event.target.value) })} /></label>
            <label><span>Tools</span><input value={(selectedNode.tools || []).map((item: Json) => item.id).join(', ')} onChange={(event) => patchNode({ tools: splitList(event.target.value).map((id) => ({ id, required: true })) })} /></label>
            <ModelContextEditor value={selectedNode.context || { type: 'unbounded' }} onChange={(value) => patchNode({ context: value })} />
            <label><span>Max tool iterations</span><input type="number" min="1" value={selectedNode.limits?.max_tool_iterations || 1} onChange={(event) => patchNode({ limits: { max_tool_iterations: Number(event.target.value) } })} /></label>
            <label><span>Operations</span><JsonTextarea value={selectedNode.operations || []} onChange={(value) => patchNode({ operations: value })} rows={7} /></label>
            <label><span>Behavior options</span><JsonTextarea value={selectedNode.behavior?.options || {}} onChange={(value) => patchNode({ behavior: { ...selectedNode.behavior, options: value } })} /></label>
          </div>}
        </aside>
        <section className="team-graph-and-bindings">
          <nav className="team-graph-view-tabs">
            <button className={graphView === 'authoring' ? 'active' : ''} onClick={() => setGraphView('authoring')}>TeamSpec 图</button>
            {FRAMEWORKS.map((framework) => <button key={framework} className={graphView === framework ? 'active' : ''} onClick={() => setGraphView(framework)}>{framework === 'autogen' ? 'AutoGen' : framework === 'langgraph' ? 'LangGraph' : 'CrewAI'} 实现</button>)}
          </nav>
          {graphView === 'authoring'
            ? <TeamGraphCanvas
              spec={draft}
              selectedNodeId={selectedNodeId}
              selectedEdgeId={selectedRelationId}
              onNodeSelect={setSelectedNodeId}
              onEdgeSelect={(_, relationId) => { setSelectedRelationId(relationId); setEditorTab('relations') }}
              onStateSelect={(stateId) => { setSelectedStateId(stateId); setEditorTab('state') }}
              onConnect={(source, target) => addRelation(source, target)}
            />
            : <FrameworkBindingView framework={graphView} report={bindingReports[graphView] || null} spec={draft} />}
        </section>
        <aside className="right-panel team-edge-panel">
          <nav className="team-contract-tabs"><button className={editorTab === 'relations' ? 'active' : ''} onClick={() => setEditorTab('relations')}>Relations</button><button className={editorTab === 'state' ? 'active' : ''} onClick={() => setEditorTab('state')}>SharedState</button><button className={editorTab === 'lifecycle' ? 'active' : ''} onClick={() => setEditorTab('lifecycle')}>Lifecycle</button></nav>
          {editorTab === 'relations' && <>
            <div className="panel-heading"><h2>Relation</h2><div className="toolbar-actions"><button className="icon-button" title="添加 Relation" onClick={() => addRelation()}><Plus size={13} /></button><button className="danger icon-button" title="删除 Relation" onClick={removeRelation}><Trash2 size={13} /></button></div></div>
            <div className="relation-list-summary edge-record-list relation-list-scroll">{(draft.relations || []).map((item: Json) => <button key={item.id} className={item.id === selectedRelationId ? 'selected' : ''} onClick={() => setSelectedRelationId(item.id)}><i className={`relation-family-dot ${item.control && item.data ? 'mixed' : item.control ? 'control' : 'data'}`} /><span><strong>{item.id}</strong><small>{item.from} → {item.to} · {item.control ? 'C' : ''}{item.data ? 'D' : ''}</small></span></button>)}</div>
            {selectedRelation && <div className="relation-inspector inspector-form relation-detail-scroll">
              <label><span>Relation ID</span><input value={selectedRelation.id} onChange={(event) => { patchRelation({ id: event.target.value }); setSelectedRelationId(event.target.value) }} /></label>
              <div className="two-fields"><label><span>From</span><select value={selectedRelation.from} onChange={(event) => patchRelation({ from: event.target.value })}>{draft.nodes.map((node: Json) => <option key={node.id}>{node.id}</option>)}</select></label><label><span>To</span><select value={selectedRelation.to} onChange={(event) => patchRelation({ to: event.target.value })}>{draft.nodes.map((node: Json) => <option key={node.id}>{node.id}</option>)}</select></label></div>
              <div className="relation-kind-switches"><label className="check-field"><input type="checkbox" checked={Boolean(selectedRelation.control)} onChange={(event) => patchRelation({ control: event.target.checked ? defaultRelation('x', 'a', 'b').control : null })} /><span>Control</span></label><label className="check-field"><input type="checkbox" checked={Boolean(selectedRelation.data)} onChange={(event) => patchRelation({ data: event.target.checked ? { transfers: [defaultTransfer()] } : null })} /><span>Data</span></label></div>
              {selectedRelation.control && <><div className="two-fields"><label><span>Trigger</span><select value={selectedRelation.control.trigger} onChange={(event) => patchRelation({ control: { ...selectedRelation.control, trigger: event.target.value } })}>{CONTROL_TRIGGERS.map((value) => <option key={value}>{value}</option>)}</select></label><label><span>Action</span><select value={selectedRelation.control.action} onChange={(event) => patchRelation({ control: { ...selectedRelation.control, action: event.target.value } })}>{CONTROL_ACTIONS.map((value) => <option key={value}>{value}</option>)}</select></label></div><div className="two-fields"><label><span>Priority</span><input type="number" value={selectedRelation.control.priority || 0} onChange={(event) => patchRelation({ control: { ...selectedRelation.control, priority: Number(event.target.value) } })} /></label><label><span>On failure</span><select value={selectedRelation.control.on_failure || 'fail_trial'} onChange={(event) => patchRelation({ control: { ...selectedRelation.control, on_failure: event.target.value } })}><option>fail_trial</option><option>continue</option><option>try_next</option><option>return_to_source</option></select></label></div></>}
              {selectedRelation.data && <div className="transfer-editor"><header><strong>Data transfers</strong><button className="icon-button" title="添加 transfer" onClick={() => patchRelation({ data: { transfers: [...selectedRelation.data.transfers, defaultTransfer()] } })}><Plus size={13} /></button></header>{selectedRelation.data.transfers.map((transfer: Json, index: number) => <article key={index}><div className="two-fields"><label><span>Source</span><input value={transfer.source} onChange={(event) => patchRelation({ data: { transfers: selectedRelation.data.transfers.map((item: Json, cursor: number) => cursor === index ? { ...item, source: event.target.value } : item) } })} /></label><label><span>Target</span><input value={transfer.target} onChange={(event) => patchRelation({ data: { transfers: selectedRelation.data.transfers.map((item: Json, cursor: number) => cursor === index ? { ...item, target: event.target.value } : item) } })} /></label></div><div className="two-fields"><label><span>View</span><select value={transfer.view || 'all'} onChange={(event) => patchRelation({ data: { transfers: selectedRelation.data.transfers.map((item: Json, cursor: number) => cursor === index ? { ...item, view: event.target.value, latest_n: event.target.value === 'latest_n' ? 1 : null } : item) } })}>{DATA_VIEWS.map((value) => <option key={value}>{value}</option>)}</select></label><label className="check-field"><input type="checkbox" checked={Boolean(transfer.required)} onChange={(event) => patchRelation({ data: { transfers: selectedRelation.data.transfers.map((item: Json, cursor: number) => cursor === index ? { ...item, required: event.target.checked } : item) } })} /><span>Required</span></label></div><button className="danger icon-button" title="删除 transfer" onClick={() => patchRelation({ data: { transfers: selectedRelation.data.transfers.filter((_: Json, cursor: number) => cursor !== index) } })}><Trash2 size={12} /></button></article>)}</div>}
            </div>}
          </>}
          {editorTab === 'state' && <>
            <div className="panel-heading"><h2>SharedState</h2><div className="toolbar-actions"><button className="icon-button" title="添加 SharedState" onClick={addState}><Plus size={13} /></button><button className="danger icon-button" title="删除 SharedState" onClick={removeState}><Trash2 size={13} /></button></div></div>
            <div className="relation-list-summary edge-record-list relation-list-scroll">{(draft.shared_state || []).map((item: Json) => <button key={item.id} className={item.id === selectedStateId ? 'selected' : ''} onClick={() => setSelectedStateId(item.id)}><i className="role-dot node-scope-store" /><span><strong>{item.id}</strong><small>{item.kind} · {item.lifetime}</small></span></button>)}</div>
            {selectedState && <div className="relation-inspector inspector-form relation-detail-scroll">
              <label><span>State ID</span><input value={selectedState.id} onChange={(event) => { patchState({ id: event.target.value }); setSelectedStateId(event.target.value) }} /></label>
              <label><span>Kind</span><select value={selectedState.kind} onChange={(event) => patchState({ kind: event.target.value, update: event.target.value === 'structured_state' ? 'merge' : event.target.value === 'artifact_store' ? 'commit' : 'append', ...(event.target.value === 'memory' ? { memory: { retrieval: { mode: 'chronological', top_k: 8, score_threshold: null, query_source: 'task_and_node_input' }, write: { policy: 'node_output', source: 'source.output' }, injection: { target: 'model_context', template: 'Relevant memory from {memory_id}:\n{items}' } } } : { memory: undefined }) })}>{STATE_KINDS.map((value) => <option key={value}>{value}</option>)}</select></label>
              <label><span>Description</span><textarea value={selectedState.description || ''} onChange={(event) => patchState({ description: event.target.value })} /></label>
              <label><span>Readers</span><input value={(selectedState.readers || []).join(', ')} onChange={(event) => patchState({ readers: splitList(event.target.value) })} /></label>
              <label><span>Writers</span><input value={(selectedState.writers || []).join(', ')} onChange={(event) => patchState({ writers: splitList(event.target.value) })} /></label>
              <div className="two-fields"><label><span>Update</span><select value={selectedState.update} onChange={(event) => patchState({ update: event.target.value })}>{STATE_UPDATES.map((value) => <option key={value}>{value}</option>)}</select></label><label><span>Lifetime</span><select value={selectedState.lifetime} onChange={(event) => patchState({ lifetime: event.target.value })}><option>invocation</option><option>trial</option><option>run</option></select></label></div>
              <div className="two-fields"><label><span>Max items</span><input type="number" min="1" value={selectedState.retention?.max_items ?? ''} onChange={(event) => patchState({ retention: { ...selectedState.retention, max_items: event.target.value ? Number(event.target.value) : null } })} /></label><label><span>Max tokens</span><input type="number" min="1" value={selectedState.retention?.max_tokens ?? ''} onChange={(event) => patchState({ retention: { ...selectedState.retention, max_tokens: event.target.value ? Number(event.target.value) : null } })} /></label></div>
              {selectedState.kind === 'memory' && <><label><span>Retrieval</span><select value={selectedState.memory?.retrieval?.mode || 'chronological'} onChange={(event) => patchState({ memory: { ...selectedState.memory, retrieval: { ...selectedState.memory.retrieval, mode: event.target.value } } })}><option>chronological</option><option>semantic</option><option>hybrid</option></select></label><label><span>Top K</span><input type="number" min="1" value={selectedState.memory?.retrieval?.top_k || 8} onChange={(event) => patchState({ memory: { ...selectedState.memory, retrieval: { ...selectedState.memory.retrieval, top_k: Number(event.target.value) } } })} /></label><label><span>Injection target</span><select value={selectedState.memory?.injection?.target || 'model_context'} onChange={(event) => patchState({ memory: { ...selectedState.memory, injection: { ...selectedState.memory.injection, target: event.target.value } } })}><option>model_context</option><option>node_input</option></select></label><label><span>Template</span><textarea value={selectedState.memory?.injection?.template || ''} onChange={(event) => patchState({ memory: { ...selectedState.memory, injection: { ...selectedState.memory.injection, template: event.target.value } } })} /></label></>}
            </div>}
          </>}
          {editorTab === 'lifecycle' && <div className="relation-inspector inspector-form relation-detail-scroll lifecycle-editor">
            <h2>Lifecycle</h2>
            <label><span>Entry Node</span><select value={draft.lifecycle?.entry?.[0]?.node || ''} onChange={(event) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, entry: [{ ...(draft.lifecycle.entry?.[0] || {}), node: event.target.value, inputs: draft.lifecycle.entry?.[0]?.inputs || [defaultTransfer()] }] } })}>{draft.nodes.map((node: Json) => <option key={node.id}>{node.id}</option>)}</select></label>
            <label><span>Result submitter</span><select value={draft.lifecycle?.result?.submissions?.[0]?.from || ''} onChange={(event) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, result: { ...draft.lifecycle.result, submissions: [{ ...(draft.lifecycle.result.submissions?.[0] || {}), from: event.target.value, source: 'source.output', key: 'final_answer' }] } } })}>{draft.nodes.map((node: Json) => <option key={node.id}>{node.id}</option>)}</select></label>
            <div className="two-fields"><label><span>Result mode</span><select value={draft.lifecycle.result.mode} onChange={(event) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, result: { ...draft.lifecycle.result, mode: event.target.value } } })}><option>first_valid</option><option>all</option><option>aggregate</option></select></label><label><span>Deadlock</span><select value={draft.lifecycle.failure.deadlock} onChange={(event) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, failure: { ...draft.lifecycle.failure, deadlock: event.target.value } } })}><option>fail_trial</option><option>submit_best_effort</option></select></label></div>
            {['max_turns', 'max_node_calls', 'max_stalls', 'timeout_seconds'].map((field) => <label key={field}><span>{field}</span><input type="number" min="1" value={draft.lifecycle.limits?.[field] ?? ''} onChange={(event) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, limits: { ...draft.lifecycle.limits, [field]: event.target.value ? Number(event.target.value) : null } } })} /></label>)}
            <label><span>Entry contract</span><JsonTextarea value={draft.lifecycle.entry || []} onChange={(value) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, entry: value } })} rows={8} /></label>
            <label><span>Result contract</span><JsonTextarea value={draft.lifecycle.result || {}} onChange={(value) => setDraft({ ...draft, lifecycle: { ...draft.lifecycle, result: value } })} rows={8} /></label>
          </div>}
        </aside>
      </section>
    </main> : <main className="management-workspace team-workspace stage-instance">
      <section className="management-instance-pane">
        <section className="management-instance-builder role-binding-builder">
          <div className="section-toolbar"><div><h1>TeamInstance Builder</h1><span>选择框架并把每个模型 Node 绑定到可用 DeploymentInstance</span></div><button className="icon-button" title="刷新并检测 TeamInstance" onClick={() => refresh(true)}><RefreshCw size={15} /></button></div>
          <label><span>TeamSpec</span><select value={instanceTeamSpecId} onChange={(event) => { const spec = specs.find((item) => item.id === event.target.value); if (spec) resetBindings(spec) }}>{specs.map((item) => <option key={item.id} value={item.id}>{item.id} · {deriveCoordinationSummary(item).topology}</option>)}</select></label>
          <div className="two-fields"><label><span>TeamInstance ID</span><input value={instanceId} onChange={(event) => setInstanceId(event.target.value)} /></label><label><span>Runtime framework</span><select value={runtimeFramework} onChange={(event) => setRuntimeFramework(event.target.value)}>{FRAMEWORKS.map((value) => <option key={value} value={value}>{value === 'autogen' ? 'AutoGen' : value === 'langgraph' ? 'LangGraph' : 'CrewAI'}</option>)}</select></label></div>
          <FrameworkBindingView framework={runtimeFramework} report={bindingReports[runtimeFramework] || null} spec={canonicalTeam} />
          <div className="role-binding-table">{requirements.map((requirement) => {
            const binding = bindings.find((item) => item.node_id === requirement.node_id) || requirement
            return <label key={requirement.node_id}><span><strong>{requirement.node_id}</strong><small>{requirement.requirement}</small></span><select value={binding.resource_instance_id || ''} onChange={(event) => setBindings((current) => current.map((item) => item.node_id === requirement.node_id ? { ...item, resource_instance_id: event.target.value } : item))}><option value="" disabled>选择可用 DeploymentInstance</option>{deployments.filter(isDeploymentReady).filter((item) => requirement.required_capabilities.every((capability: string) => item.capabilities?.[capability] === true)).map((item) => <option key={item.id} value={item.id}>{item.model_id} · {item.kind} · {item.observed_status || item.status}</option>)}</select></label>
          })}</div>
          <button className="primary" disabled={busy || bindings.some((item) => !item.resource_instance_id)} onClick={instantiate}><Boxes size={14} />实例化</button>
        </section>
        <section className="management-instance-registry team-instance-registry">
          <div className="section-toolbar"><div><h1>TeamInstance</h1><span>持久化的框架实现与部署绑定</span></div></div>
          <label className="field-icon"><Search size={14} /><input placeholder="搜索 TeamInstance" value={instanceQuery} onChange={(event) => setInstanceQuery(event.target.value)} /></label>
          <div className="team-instance-list">{filteredInstances.map((item) => <article key={item.id}><div><span className={`status-pill ${item.available ? 'running' : 'stopped'}`}>{item.observed_status || 'unknown'}</span><strong>{item.id}</strong><small>{item.team_spec_id} · {item.runtime_framework} · {(item.resource_bindings || []).length} bindings</small></div><div className="toolbar-actions"><button className="secondary compact-button" onClick={() => { setInstanceTeamSpecId(item.team_spec_id); setInstanceId(item.id); setRuntimeFramework(item.runtime_framework); setBindings(structuredClone(item.resource_bindings || [])) }}>载入</button><button className="danger icon-button" title="删除 TeamInstance" onClick={async () => { if (!window.confirm(`确认删除 ${item.id}？`)) return; try { await api.deleteTeamInstance(item.id); await refresh(); notify(`TeamInstance ${item.id} 已删除`) } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) } }}><Trash2 size={13} /></button></div><DetailsDisclosure label="展开 TeamInstance" value={item} /></article>)}{filteredInstances.length === 0 && <div className="empty-state">没有匹配的 TeamInstance</div>}</div>
        </section>
      </section>
    </main>}
  </section>
}
