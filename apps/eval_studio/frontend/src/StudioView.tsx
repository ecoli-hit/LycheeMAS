import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type NodeProps,
} from '@xyflow/react'
import {
  Check,
  Plus,
  RefreshCw,
  Save,
  ServerCog,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import SpecSelector from './SpecSelector'
import type { AgentData, AgentNode, Bootstrap, Json, TeamEdge } from './types'

interface Props {
  bootstrap: Bootstrap
  environment: Json
  notify: (message: string) => void
  workspace: 'team' | 'deployment'
}

const modelParticipant = (participant: Json) => participant.agent_type !== 'computer_terminal'

const groupChatLabel = (type: unknown) => ({
  round_robin: 'RoundRobinGroupChat',
  selector: 'SelectorGroupChat',
  magentic_one: 'MagenticOneGroupChat',
}[String(type || 'round_robin')] || String(type || 'GroupChat'))

function inferenceSlots(spec: Json): Json[] {
  const participants: Json[] = (spec.participants || [])
    .filter(modelParticipant)
    .map((participant: Json) => ({
      id: String(participant.id),
      kind: 'participant',
      participant_id: String(participant.id),
      required_capabilities: ['text_generation'],
    }))
  const type = spec.group_chat?.type || 'round_robin'
  if (type === 'magentic_one') {
    participants.push({
      id: 'Orchestrator',
      kind: 'controller',
      controller_type: 'orchestrator',
      required_capabilities: ['text_generation'],
    })
  } else if (type === 'selector' && !spec.group_chat?.selector_func_factory) {
    participants.push({
      id: 'Selector',
      kind: 'controller',
      controller_type: 'selector',
      required_capabilities: ['text_generation'],
    })
  }
  return participants
}

function graphNodes(spec: Json): AgentNode[] {
  const participants = spec.participants || []
  const controlled = ['selector', 'magentic_one'].includes(spec.group_chat?.type)
  const startX = controlled ? 100 : 70
  const nodes: AgentNode[] = participants.map((participant: Json, index: number) => ({
    id: String(participant.id),
    type: 'agent',
    position: controlled
      ? { x: startX + (index % 2) * 260, y: 185 + Math.floor(index / 2) * 135 }
      : { x: 55 + index * 220, y: 210 },
    data: {
      label: String(participant.name || participant.id),
      role: String(participant.name || participant.id),
      agent_type: String(participant.agent_type || 'assistant'),
      system_prompt: String(participant.system_prompt || ''),
      description: String(participant.description || ''),
      tools: participant.tools || [],
      model_context: participant.model_context || spec.model_context || { type: 'unbounded' },
    },
  }))
  if (spec.group_chat?.type === 'selector' || spec.group_chat?.type === 'magentic_one') {
    const orchestrator = spec.group_chat.type === 'magentic_one'
    nodes.push({
      id: orchestrator ? '__orchestrator__' : '__selector__',
      type: 'agent',
      position: { x: 230, y: 35 },
      draggable: false,
      data: {
        label: orchestrator ? 'Orchestrator' : 'Selector',
        role: orchestrator ? 'Orchestrator' : 'Selector',
        agent_type: orchestrator ? 'orchestrator' : 'selector',
        system_prompt: '',
        description: orchestrator
          ? 'MagenticOneGroupChat runtime controller'
          : 'SelectorGroupChat runtime controller',
        tools: [],
      },
    })
  }
  return nodes
}

function graphEdges(spec: Json, nodes: AgentNode[]): TeamEdge[] {
  const control = nodes.find((node) => node.id.startsWith('__'))
  const participants = nodes.filter((node) => !node.id.startsWith('__'))
  const edge = (source: string, target: string): TeamEdge => ({
    id: `${source}-${target}`,
    source,
    target,
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
  })
  const explicitTopology = spec.extensions?.topology?.edges || {}
  const topologyEdges: TeamEdge[] = Object.entries(explicitTopology).flatMap(
    ([source, targets]) => (Array.isArray(targets) ? targets : []).map((target: string) => ({
      ...edge(source, target),
      id: `topology-${source}-${target}`,
      animated: true,
      style: { stroke: '#477f73', strokeDasharray: '5 4' },
    })),
  )
  if (control) {
    return [
      ...participants.flatMap((node) => [edge(control.id, node.id), edge(node.id, control.id)]),
      ...topologyEdges,
    ]
  }
  if (topologyEdges.length) return topologyEdges
  const edges = participants.slice(0, -1).map((node, index) => edge(node.id, participants[index + 1].id))
  if (participants.length > 1) edges.push(edge(participants.at(-1)!.id, participants[0].id))
  return edges
}

function AgentNodeView({ data }: NodeProps<AgentNode>) {
  return <div className={`agent-node-inner agent-${data.agent_type}`}>
    <Handle type="target" position={Position.Top} />
    <Handle type="source" position={Position.Bottom} />
    <strong>{data.label}</strong>
    <small>{data.agent_type}</small>
  </div>
}

const nodeTypes = { agent: AgentNodeView }

function status(value: Json) {
  return String(value.observed_status || value.status || 'unknown')
}

function isDeploymentReady(value: Json) {
  return ['running', 'ready_on_run'].includes(status(value))
}

function compatible(slot: Json, deployment: Json) {
  return (slot.required_capabilities || []).every(
    (capability: string) => deployment.capabilities?.[capability] === true,
  )
}

function compactOverrides(value: Json): Json {
  const result: Json = {}
  for (const key of [
    'max_new_tokens',
    'max_input_tokens',
    'min_output_reserve_tokens',
    'min_thinking_reserve_tokens',
    'max_thinking_budget_tokens',
    'min_final_reserve_tokens',
    'safety_margin_tokens',
    'top_k',
    'temperature',
    'top_p',
    'min_p',
    'presence_penalty',
    'repetition_penalty',
  ]) {
    if (value[key] !== '' && value[key] != null) result[key] = Number(value[key])
  }
  if (value.thinking_mode && value.thinking_mode !== 'inherit') {
    result.thinking_mode = value.thinking_mode
  }
  if (value.preserve_thinking === 'preserve') result.preserve_thinking = true
  if (value.preserve_thinking === 'clear') result.preserve_thinking = false
  if (value.do_sample === 'enabled' || value.do_sample === true) result.do_sample = true
  if (value.do_sample === 'disabled' || value.do_sample === false) result.do_sample = false
  return result
}

function editableOverrides(value: Json): Json {
  return {
    ...value,
    preserve_thinking: value.preserve_thinking === true
      ? 'preserve'
      : value.preserve_thinking === false
        ? 'clear'
        : 'inherit',
    do_sample: value.do_sample === true
      ? 'enabled'
      : value.do_sample === false
        ? 'disabled'
        : 'inherit',
  }
}

function InvocationOverrides({ value, onChange }: { value: Json; onChange: (value: Json) => void }) {
  const patch = (field: string, next: unknown) => onChange({ ...value, [field]: next })
  return <div className="invocation-policy-controls">
    <label><span>Thinking</span><select value={value.thinking_mode || 'inherit'} onChange={(event) => patch('thinking_mode', event.target.value)}><option value="inherit">使用 DeploymentInstance 默认</option><option value="enabled">开启</option><option value="disabled">关闭</option></select></label>
    <label><span>历史思考</span><select value={value.preserve_thinking || 'inherit'} onChange={(event) => patch('preserve_thinking', event.target.value)}><option value="inherit">使用 DeploymentInstance 默认</option><option value="preserve">保留</option><option value="clear">不保留</option></select></label>
    <label><span>Max output</span><input type="number" min="1" placeholder="Experiment 默认" value={value.max_new_tokens || ''} onChange={(event) => patch('max_new_tokens', event.target.value)} /></label>
    <label><span>Max input</span><input type="number" min="1" placeholder="自动" value={value.max_input_tokens || ''} onChange={(event) => patch('max_input_tokens', event.target.value)} /></label>
    <label><span>Min output reserve</span><input type="number" min="0" value={value.min_output_reserve_tokens ?? ''} onChange={(event) => patch('min_output_reserve_tokens', event.target.value)} /></label>
    <label><span>Min thinking reserve</span><input type="number" min="0" value={value.min_thinking_reserve_tokens ?? ''} onChange={(event) => patch('min_thinking_reserve_tokens', event.target.value)} /></label>
    <label><span>Max thinking budget</span><input type="number" min="1" value={value.max_thinking_budget_tokens || ''} onChange={(event) => patch('max_thinking_budget_tokens', event.target.value)} /></label>
    <label><span>Min final reserve</span><input type="number" min="0" value={value.min_final_reserve_tokens ?? ''} onChange={(event) => patch('min_final_reserve_tokens', event.target.value)} /></label>
    <label><span>Safety margin</span><input type="number" min="0" value={value.safety_margin_tokens ?? ''} onChange={(event) => patch('safety_margin_tokens', event.target.value)} /></label>
    <label><span>Sampling</span><select value={value.do_sample ?? 'inherit'} onChange={(event) => patch('do_sample', event.target.value)}><option value="inherit">使用 Experiment 默认</option><option value="enabled">开启</option><option value="disabled">关闭 / 确定性解码</option></select></label>
    <label><span>Temperature</span><input type="number" min="0" step="0.05" value={value.temperature ?? ''} placeholder="Experiment 默认" onChange={(event) => patch('temperature', event.target.value)} /></label>
    <label><span>Top P</span><input type="number" min="0.01" max="1" step="0.05" value={value.top_p ?? ''} placeholder="Experiment 默认" onChange={(event) => patch('top_p', event.target.value)} /></label>
    <label><span>Top K</span><input type="number" min="1" step="1" value={value.top_k ?? ''} placeholder="Provider 默认" onChange={(event) => patch('top_k', event.target.value)} /></label>
    <label><span>Min P</span><input type="number" min="0" max="1" step="0.05" value={value.min_p ?? ''} placeholder="Provider 默认" onChange={(event) => patch('min_p', event.target.value)} /></label>
    <label><span>Presence penalty</span><input type="number" min="-2" max="2" step="0.1" value={value.presence_penalty ?? ''} placeholder="Provider 默认" onChange={(event) => patch('presence_penalty', event.target.value)} /></label>
    <label><span>Repetition penalty</span><input type="number" min="0" step="0.05" value={value.repetition_penalty ?? ''} placeholder="Provider 默认" onChange={(event) => patch('repetition_penalty', event.target.value)} /></label>
  </div>
}

function ModelContextEditor({ value, onChange }: { value: Json; onChange: (value: Json) => void }) {
  const type = value?.type || 'unbounded'
  return <div className="model-context-controls">
    <label><span>Model context</span><select value={type} onChange={(event) => onChange(event.target.value === 'buffered' ? { type: 'buffered', buffer_size: 12 } : event.target.value === 'token_limited' ? { type: 'token_limited' } : { type: 'unbounded' })}><option value="unbounded">Unbounded</option><option value="buffered">Buffered</option><option value="token_limited">Token limited</option></select></label>
    {type === 'buffered' && <label><span>Buffer size</span><input type="number" min="1" value={value.buffer_size || 12} onChange={(event) => onChange({ type, buffer_size: Number(event.target.value) })} /></label>}
    {type === 'token_limited' && <label><span>Token limit</span><input type="number" min="1" value={value.token_limit || ''} placeholder="模型窗口" onChange={(event) => onChange({ type, ...(event.target.value ? { token_limit: Number(event.target.value) } : {}) })} /></label>}
  </div>
}

function TeamManagement({ bootstrap, notify }: Omit<Props, 'workspace' | 'environment'>) {
  const initial = bootstrap.teams.specs[0] || {
    schema_version: 4,
    id: 'team',
    participants: [],
    group_chat: { type: 'round_robin' },
    model_context: { type: 'unbounded' },
    controller_model_context: { type: 'unbounded' },
    termination: { conditions: [] },
    extensions: {},
  }
  const [specs, setSpecs] = useState<Json[]>(bootstrap.teams.specs || [])
  const [instances, setInstances] = useState<Json[]>(bootstrap.team_instances?.instances || [])
  const [deployments, setDeployments] = useState<Json[]>(bootstrap.deployments?.instances || [])
  const [draft, setDraft] = useState<Json>(structuredClone(initial))
  const [selectedParticipant, setSelectedParticipant] = useState(0)
  const [selectedController, setSelectedController] = useState(false)
  const [instanceTeamSpecId, setInstanceTeamSpecId] = useState(initial.id)
  const [instanceId, setInstanceId] = useState(`${initial.id}-instance`)
  const [bindings, setBindings] = useState<Json[]>(() => inferenceSlots(initial).map((slot) => ({ slot_id: slot.id, deployment_instance_id: '', generation_overrides: {} })))
  const [busy, setBusy] = useState(false)

  const canonicalTeam = specs.find((item) => item.id === instanceTeamSpecId) || specs[0]
  const slots = inferenceSlots(canonicalTeam || { participants: [] })
  const nodes = useMemo(() => graphNodes(draft).map((node) => ({
    ...node,
    selected: selectedController ? node.id.startsWith('__') : node.id === draft.participants?.[selectedParticipant]?.id,
  })), [draft, selectedController, selectedParticipant])
  const edges = useMemo(() => graphEdges(draft, nodes), [draft, nodes])
  const participant = selectedController ? undefined : draft.participants?.[selectedParticipant]
  const controllerType = draft.group_chat?.type === 'magentic_one'
    ? 'orchestrator'
    : draft.group_chat?.type === 'selector'
      ? 'selector'
      : ''
  const participantUsesCustomPrompt = participant?.agent_type === 'assistant'
  const participantUsesModel = participant?.agent_type !== 'computer_terminal'

  const refresh = async (probe = false) => {
    const [teamCatalog, teamInstances, deploymentCatalog] = await Promise.all([
      api.teamSpecs(),
      api.teamInstances(probe),
      api.deployments(probe),
    ])
    setSpecs(teamCatalog.specs || [])
    setInstances(teamInstances.instances || [])
    setDeployments(deploymentCatalog.instances || [])
  }

  const resetBindings = (spec: Json) => {
    const firstReady = deployments.find(isDeploymentReady)?.id || ''
    setInstanceTeamSpecId(spec.id)
    setInstanceId(`${spec.id}-instance`)
    setBindings(inferenceSlots(spec).map((slot) => ({
      slot_id: slot.id,
      deployment_instance_id: firstReady,
      generation_overrides: {},
    })))
  }

  const loadSpec = (spec: Json) => {
    setDraft(structuredClone(spec))
    setSelectedParticipant(0)
    setSelectedController(false)
  }

  const newSpec = () => {
    const id = `team-${Date.now().toString(36)}`
    const value = {
      schema_version: 4,
      id,
      participants: [{ id: 'solver', name: 'solver', agent_type: 'assistant', system_prompt: '', description: '', tools: [] }],
      group_chat: { type: 'round_robin' },
      model_context: { type: 'unbounded' },
      controller_model_context: { type: 'unbounded' },
      termination: { conditions: [] },
      extensions: {},
    }
    loadSpec(value)
  }

  const saveSpec = async () => {
    setBusy(true)
    try {
      const payload: Json = {
        ...draft,
        schema_version: 4,
        inference_slots: inferenceSlots(draft),
      }
      const saved = await api.saveTeamSpec(payload.id, payload)
      await refresh()
      setDraft(saved)
      resetBindings(saved)
      notify(`TeamSpec ${saved.id} 已保存`)
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause))
    } finally { setBusy(false) }
  }

  const removeSpec = async () => {
    if (!window.confirm(`确认删除 TeamSpec ${draft.id}？`)) return
    try { await api.deleteTeamSpec(draft.id); await refresh(); notify(`TeamSpec ${draft.id} 已删除`) }
    catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const chooseInstanceTeam = (id: string) => {
    const spec = specs.find((item) => item.id === id)
    if (spec) resetBindings(spec)
  }

  const updateBinding = (slotId: string, patch: Json) => setBindings((current) => current.map((item) => item.slot_id === slotId ? { ...item, ...patch } : item))

  const instantiate = async () => {
    setBusy(true)
    try {
      const saved = await api.saveTeamInstance(instanceId, {
        schema_version: 3,
        id: instanceId,
        team_spec_id: canonicalTeam.id,
        inference_bindings: bindings.map((item) => ({
          slot_id: item.slot_id,
          deployment_instance_id: item.deployment_instance_id,
          ...(Object.keys(compactOverrides(item.generation_overrides || {})).length
            ? { generation_overrides: compactOverrides(item.generation_overrides || {}) }
            : {}),
        })),
      })
      await refresh(true)
      notify(`TeamInstance ${saved.id} 已实例化`)
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause))
    } finally { setBusy(false) }
  }

  const loadInstance = (item: Json) => {
    const spec = specs.find((candidate) => candidate.id === item.team_spec_id)
    if (!spec) return
    setInstanceTeamSpecId(spec.id)
    setInstanceId(item.id)
    setBindings((item.inference_bindings || []).map((binding: Json) => ({
      ...structuredClone(binding),
      generation_overrides: editableOverrides(binding.generation_overrides || {}),
    })))
  }

  const removeInstance = async (item: Json) => {
    if (!window.confirm(`确认删除 TeamInstance ${item.id}？`)) return
    try { await api.deleteTeamInstance(item.id); await refresh(); notify(`TeamInstance ${item.id} 已删除`) }
    catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const patchParticipant = (patch: Json) => setDraft((current: Json) => ({
    ...current,
    participants: current.participants.map((item: Json, index: number) => index === selectedParticipant ? { ...item, ...patch } : item),
  }))

  const patchGroupChat = (patch: Json) => setDraft((current: Json) => ({
    ...current,
    group_chat: { ...(current.group_chat || {}), ...patch },
  }))

  const patchTerminationText = (text: string) => setDraft((current: Json) => ({
    ...current,
    termination: text
      ? { conditions: [{ type: 'text_mention', text }] }
      : { conditions: [] },
  }))

  const setContextVisibility = (type: string) => setDraft((current: Json) => ({
    ...current,
    extensions: {
      ...(current.extensions || {}),
      context_visibility: { type },
    },
  }))

  const setTopologyEnabled = (enabled: boolean) => setDraft((current: Json) => {
    const extensions = { ...(current.extensions || {}) }
    if (!enabled) {
      delete extensions.topology
      return { ...current, extensions }
    }
    const ids = (current.participants || []).map((item: Json) => item.id)
    const edges = Object.fromEntries(ids.map((id: string, index: number) => [
      id,
      index + 1 < ids.length ? [ids[index + 1]] : [],
    ]))
    return { ...current, extensions: { ...extensions, topology: { edges, speaking_order: ids } } }
  })

  const setTopologyEdge = (source: string, target: string, enabled: boolean) => setDraft((current: Json) => {
    const topology = current.extensions?.topology || { edges: {}, speaking_order: [] }
    const targets = new Set<string>(topology.edges?.[source] || [])
    if (enabled) targets.add(target)
    else targets.delete(target)
    return {
      ...current,
      extensions: {
        ...(current.extensions || {}),
        topology: {
          ...topology,
          edges: { ...(topology.edges || {}), [source]: [...targets] },
        },
      },
    }
  })

  const setSpeakingOrder = (raw: string) => setDraft((current: Json) => ({
    ...current,
    extensions: {
      ...(current.extensions || {}),
      topology: {
        ...(current.extensions?.topology || { edges: {} }),
        speaking_order: raw.split(',').map((item) => item.trim()).filter(Boolean),
      },
    },
  }))

  const addParticipant = () => {
    const id = `agent_${(draft.participants || []).length + 1}`
    setDraft({ ...draft, participants: [...(draft.participants || []), { id, name: id, agent_type: 'assistant', system_prompt: '', description: '', tools: [] }] })
    setSelectedParticipant((draft.participants || []).length)
    setSelectedController(false)
  }

  const removeParticipant = () => {
    if (!participant) return
    setDraft({ ...draft, participants: draft.participants.filter((_: Json, index: number) => index !== selectedParticipant) })
    setSelectedParticipant(Math.max(0, selectedParticipant - 1))
  }

  return <section className="studio-view">
    <div className="team-workspace management-workspace">
      <section className="team-spec-pane management-spec-pane">
        <header className="team-spec-header management-spec-header">
          <div className="section-toolbar"><div><h1>TeamSpec</h1><span>定义参与者与 AutoGen GroupChat，不绑定模型实例</span></div><div className="toolbar-actions"><button className="icon-button" title="新建 TeamSpec" onClick={newSpec}><Plus size={16} /></button><button className="secondary" disabled={busy} onClick={saveSpec}><Save size={15} />保存</button><button className="icon-button danger" title="删除 TeamSpec" onClick={removeSpec}><Trash2 size={15} /></button></div></div>
          <SpecSelector kind="TeamSpec" specs={specs} draftId={draft.id || ''} optionLabel={(item) => `${item.id} · ${groupChatLabel(item.group_chat?.type)}`} onLoad={loadSpec} />
        </header>
        <div className="team-spec-fields">
          <label><span>TeamSpec ID</span><input value={draft.id || ''} onChange={(event) => setDraft({ ...draft, id: event.target.value })} /></label>
          <label><span>GroupChat</span><select value={draft.group_chat?.type || 'round_robin'} onChange={(event) => { setDraft({ ...draft, group_chat: event.target.value === 'magentic_one' ? { type: 'magentic_one', max_stalls: 3 } : event.target.value === 'selector' ? { type: 'selector', max_selector_attempts: 3, allow_repeated_speaker: false } : { type: 'round_robin' } }); setSelectedController(event.target.value !== 'round_robin') }}><option value="round_robin">RoundRobinGroupChat</option><option value="selector">SelectorGroupChat</option><option value="magentic_one">MagenticOneGroupChat</option></select></label>
          <label><span>Termination text</span><input value={draft.termination?.conditions?.[0]?.text || ''} placeholder="留空表示不增加文本终止条件" onChange={(event) => patchTerminationText(event.target.value)} /><small>当前编译为 AutoGen TextMentionTermination。</small></label>
        </div>
        <div className="team-communication-settings">
          <label><span>Message visibility</span><select value={draft.extensions?.context_visibility?.type || 'shared'} onChange={(event) => setContextVisibility(event.target.value)}><option value="shared">Shared · AutoGen 默认完整 GroupChat 历史</option><option value="topology_filtered">Topology filtered · 按入边过滤可见消息</option></select></label>
          <label className="check-field"><input type="checkbox" checked={Boolean(draft.extensions?.topology)} onChange={(event) => setTopologyEnabled(event.target.checked)} /><span>启用显式拓扑扩展</span></label>
          {draft.extensions?.topology && <>
            <label className="wide-field"><span>Speaking order</span><input value={(draft.extensions.topology.speaking_order || []).join(', ')} onChange={(event) => setSpeakingOrder(event.target.value)} /><small>RoundRobin 的顺序，或 topology selector 的稳定候选顺序；填写 participant ID，以逗号分隔。</small></label>
            <div className="topology-edge-matrix wide-field"><strong>允许的 participant 路径</strong>{(draft.participants || []).map((source: Json) => <div key={source.id}><span>{source.id}</span>{(draft.participants || []).filter((target: Json) => target.id !== source.id).map((target: Json) => <label key={`${source.id}-${target.id}`} className="check-field"><input type="checkbox" checked={Boolean(draft.extensions?.topology?.edges?.[source.id]?.includes(target.id))} onChange={(event) => setTopologyEdge(source.id, target.id, event.target.checked)} /><span>{target.id}</span></label>)}</div>)}</div>
          </>}
        </div>
        <div className="studio-grid">
          <aside className="left-panel">
            <div className="panel-heading"><h2>Participants</h2><button className="icon-button" title="添加 participant" onClick={addParticipant}><Plus size={14} /></button></div>
            <div className="role-list">{(draft.participants || []).map((item: Json, index: number) => <button key={`${item.id}-${index}`} className={!selectedController && index === selectedParticipant ? 'selected' : ''} onClick={() => { setSelectedParticipant(index); setSelectedController(false) }}><span className={`role-dot type-${item.agent_type}`} /><span><strong>{item.name || item.id}</strong><small>{item.agent_type}</small></span></button>)}{controllerType && <button className={selectedController ? 'selected controller-list-item' : 'controller-list-item'} onClick={() => setSelectedController(true)}><span className={`role-dot type-${controllerType}`} /><span><strong>{controllerType === 'orchestrator' ? 'Orchestrator' : 'Selector'}</strong><small>GroupChat controller</small></span></button>}</div>
            <div className="team-contract-note"><strong>Inference slots</strong>{inferenceSlots(draft).map((slot) => <span key={slot.id}>{slot.id} · {slot.kind}</span>)}</div>
          </aside>
          <section className="graph-panel">
            <div className="team-graph-canvas"><ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.25 }} nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, node) => { if (node.id.startsWith('__')) { setSelectedController(true); return } const index = (draft.participants || []).findIndex((item: Json) => item.id === node.id); if (index >= 0) { setSelectedParticipant(index); setSelectedController(false) } }}><Background gap={22} size={1} /><Controls showInteractive={false} /></ReactFlow></div>
            <div className="team-behavior-preview"><div className="behavior-heading"><div><strong>实际运行结构</strong><span>黄色节点是 GroupChat 在运行时创建的控制器</span></div><code>{draft.group_chat?.type}</code></div><p className="behavior-note">Participant 是团队成员；Selector / Orchestrator 是 GroupChat 控制器。所有需要模型推理的对象都会生成平等的 inference slot，并在 TeamInstance 中绑定 DeploymentInstance。</p><DetailsDisclosure value={draft} label="展开 TeamSpec" /></div>
          </section>
          <aside className="right-panel">
            <div className="panel-heading"><h2>{selectedController ? 'Controller 配置' : 'Participant 配置'}</h2>{participant && <button className="icon-button danger" title="删除 participant" onClick={removeParticipant}><Trash2 size={14} /></button>}</div>
            {selectedController && controllerType ? <div className="inspector-form">
              <div className="controller-explanation"><strong>{controllerType === 'orchestrator' ? 'Orchestrator' : 'Selector'}</strong><span>由 {groupChatLabel(draft.group_chat?.type)} 在运行时创建，不属于 participants；它会调用模型，因此拥有独立 inference slot 和 DeploymentInstance 绑定。</span></div>
              {controllerType === 'selector' && <>
                <label><span>Selection strategy</span><select value={draft.group_chat?.selector_func_factory ? 'topology_selector' : draft.group_chat?.candidate_func_factory ? 'topology_candidates' : 'model'} onChange={(event) => patchGroupChat(event.target.value === 'topology_selector' ? { selector_func_factory: 'topology_selector', candidate_func_factory: undefined } : event.target.value === 'topology_candidates' ? { selector_func_factory: undefined, candidate_func_factory: 'topology_candidates' } : { selector_func_factory: undefined, candidate_func_factory: undefined })}><option value="model">Model selector · 模型直接选下一角色</option><option value="topology_selector">Topology selector_func · Python 函数直接选人</option><option value="topology_candidates">Topology candidate_func · 先过滤候选，再由模型选择</option></select></label>
                {!draft.group_chat?.selector_func_factory && <label><span>Selector prompt</span><textarea rows={7} value={draft.group_chat?.selector_prompt || ''} placeholder="留空使用 AutoGen 默认 selector prompt" onChange={(event) => patchGroupChat({ selector_prompt: event.target.value || undefined })} /></label>}
                <label><span>Max selector attempts</span><input type="number" min="1" value={draft.group_chat?.max_selector_attempts || 3} onChange={(event) => patchGroupChat({ max_selector_attempts: Number(event.target.value) })} /></label>
                <label className="check-field"><input type="checkbox" checked={Boolean(draft.group_chat?.allow_repeated_speaker)} onChange={(event) => patchGroupChat({ allow_repeated_speaker: event.target.checked })} /><span>允许连续选择同一 participant</span></label>
              </>}
              {controllerType === 'orchestrator' && <>
                <label><span>Max stalls</span><input type="number" min="1" value={draft.group_chat?.max_stalls || 3} onChange={(event) => patchGroupChat({ max_stalls: Number(event.target.value) })} /><small>连续停滞达到该值后，Magentic-One 会重新规划或结束。</small></label>
                <label><span>Final answer prompt</span><textarea rows={7} value={draft.group_chat?.final_answer_prompt || ''} placeholder="留空使用 AutoGen 默认 final answer prompt" onChange={(event) => patchGroupChat({ final_answer_prompt: event.target.value || undefined })} /></label>
              </>}
              <ModelContextEditor value={draft.controller_model_context || draft.model_context || { type: 'unbounded' }} onChange={(value) => setDraft({ ...draft, controller_model_context: value })} />
            </div> : participant ? <div className="inspector-form">
              <label><span>ID</span><input value={participant.id || ''} onChange={(event) => patchParticipant({ id: event.target.value, name: participant.name === participant.id ? event.target.value : participant.name })} /></label>
              <label><span>Name</span><input value={participant.name || ''} onChange={(event) => patchParticipant({ name: event.target.value })} /></label>
              <label><span>Agent type</span><select value={participant.agent_type || 'assistant'} onChange={(event) => patchParticipant({ agent_type: event.target.value })}><option value="assistant">AssistantAgent</option><option value="coder">Coder</option><option value="computer_terminal">ComputerTerminal</option><option value="file_surfer">FileSurfer</option><option value="web_surfer">WebSurfer</option></select></label>
              <label><span>Description</span><textarea disabled={!participantUsesCustomPrompt} rows={3} value={participant.description || ''} onChange={(event) => patchParticipant({ description: event.target.value })} />{!participantUsesCustomPrompt && <small>该 specialized agent 使用 AutoGen 官方 description。</small>}</label>
              <label><span>System prompt</span><textarea disabled={!participantUsesCustomPrompt} rows={9} value={participant.system_prompt || ''} onChange={(event) => patchParticipant({ system_prompt: event.target.value })} />{!participantUsesCustomPrompt && <small>该 specialized agent 使用 AutoGen 官方构造与 prompt。</small>}</label>
              {participantUsesCustomPrompt && <label><span>Function tools</span><input value={(participant.tools || []).join(', ')} placeholder="list_workspace, read_text_file 或 module:function" onChange={(event) => patchParticipant({ tools: event.target.value.split(',').map((item) => item.trim()).filter(Boolean) })} /><small>工具名称会在运行时解析；未知名称不会自动安装工具。</small></label>}
              {participantUsesModel
                ? <ModelContextEditor value={participant.model_context || draft.model_context || { type: 'unbounded' }} onChange={(value) => patchParticipant({ model_context: value })} />
                : <small>ComputerTerminal 不调用模型，因此没有 model context。</small>}
            </div> : <div className="empty-state">添加一个 participant</div>}
          </aside>
        </div>
      </section>

      <aside className="team-instance-pane management-instance-pane">
        <section className="role-binding-builder management-instance-builder">
          <div className="section-toolbar"><div><h1>TeamInstance</h1><span>选择 TeamSpec，并为每个 inference slot 绑定 DeploymentInstance</span></div><button className="icon-button" title="刷新并探测 DeploymentInstance" onClick={() => refresh(true)}><RefreshCw size={15} /></button></div>
          <label><span>TeamSpec</span><select value={instanceTeamSpecId} onChange={(event) => chooseInstanceTeam(event.target.value)}>{specs.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.group_chat?.type}</option>)}</select></label>
          <label><span>TeamInstance ID</span><input value={instanceId} onChange={(event) => setInstanceId(event.target.value)} /></label>
          <div className="role-binding-table">{slots.map((slot) => {
            const binding = bindings.find((item) => item.slot_id === slot.id) || { slot_id: slot.id, deployment_instance_id: '', generation_overrides: {} }
            return <div key={slot.id} className={`role-binding-row ${slot.kind === 'controller' ? 'control-binding' : ''}`}>
              <span><strong>{slot.id}</strong><small>{slot.kind === 'controller' ? `${slot.controller_type} controller` : `participant · ${slot.participant_id}`}</small></span>
              <label><span>DeploymentInstance</span><select value={binding.deployment_instance_id || ''} onChange={(event) => updateBinding(slot.id, { deployment_instance_id: event.target.value })}><option value="">请选择已实例化部署</option>{deployments.map((item) => <option key={item.id} value={item.id} disabled={!isDeploymentReady(item) || !compatible(slot, item)}>{item.model_id} · {item.kind} · {status(item)} · {item.id}</option>)}</select></label>
              <InvocationOverrides value={binding.generation_overrides || {}} onChange={(value) => updateBinding(slot.id, { generation_overrides: value })} />
            </div>
          })}</div>
          <button className="primary" disabled={busy || !instanceId || bindings.some((item) => !item.deployment_instance_id)} onClick={instantiate}><Check size={15} />实例化 Team</button>
        </section>
        <section className="team-instance-registry management-instance-registry">
          <div className="panel-heading"><h2>已注册 TeamInstance</h2><span>{instances.length}</span></div>
          <div className="team-instance-list">{instances.map((item) => <article key={item.id} className={item.id === instanceId ? 'selected' : ''}><div><span className={`status-pill ${item.available ? 'ready' : 'unavailable'}`}>{item.observed_status || 'unknown'}</span><strong>{item.id}</strong><small>{item.team_spec_id} · {(item.inference_bindings || []).length} inference bindings</small>{!item.available && <small>{[...(item.missing_deployment_instance_ids || []), ...(item.unavailable_deployment_instance_ids || []), ...(item.capability_errors || [])].join(' · ')}</small>}<DetailsDisclosure value={item} label="展开 TeamInstance" className="card-details" /></div><div className="row-actions"><button className="secondary compact-button" onClick={() => loadInstance(item)}>载入</button><button className="icon-button danger" title="删除 TeamInstance" onClick={() => removeInstance(item)}><Trash2 size={14} /></button></div></article>)}</div>
        </section>
      </aside>
    </div>
  </section>
}

const preferredActualPricingSpec = (
  pricingSpecs: Json[],
  basis: 'allocated_gpu_time' | 'token_usage',
  modelId = '',
  sourceSpecId = '',
): Json | undefined => {
  const candidates = pricingSpecs.filter((item) => item.basis === basis)
  if (basis === 'allocated_gpu_time') {
    return candidates.find((item) => item.id === 'reference-a800-on-demand-cny')
      || candidates.find((item) => item.billing_mode === 'on_demand_gpu_hour')
      || candidates[0]
  }
  return candidates.find((item) => pricingAppliesToModel(item, modelId)
    && (!sourceSpecId || pricingAppliesToSource(item, sourceSpecId)))
    || candidates.find((item) => pricingAppliesToModel(item, modelId))
    || candidates[0]
}

function defaultDeploymentSpec(bootstrap: Bootstrap): Json {
  const modelSpec = bootstrap.model_registry?.specs?.[0]
  const pricingSpecs: Json[] = bootstrap.pricing_registry?.specs || []
  const actualPricingSpec = preferredActualPricingSpec(pricingSpecs, 'allocated_gpu_time')
    || pricingSpecs[0]
  const equivalentPricingSpec = preferredActualPricingSpec(
    pricingSpecs,
    'token_usage',
    String(modelSpec?.name || ''),
  )
  return {
    schema_version: 2,
    id: `deployment-${Date.now().toString(36)}`,
    kind: 'hf',
    source_spec: { type: 'model', id: modelSpec?.id || '' },
    actual_pricing_spec_id: actualPricingSpec?.id || '',
    api_equivalent_pricing_spec_id: equivalentPricingSpec?.id || null,
    python: bootstrap.environments?.default?.python || '',
    device: 'cuda:0',
    cuda_visible_devices: '0',
    capabilities: { text_generation: true },
  }
}

const actualPricingBasis = (deployment: Json): 'allocated_gpu_time' | 'token_usage' => (
  deployment.kind === 'hf' || (deployment.kind === 'vllm' && deployment.managed !== false)
    ? 'allocated_gpu_time'
    : 'token_usage'
)

const localInfrastructurePricing = (deployment: Json): boolean => (
  actualPricingBasis(deployment) === 'allocated_gpu_time'
)

const pricingAppliesToModel = (pricing: Json, modelId: string): boolean => {
  const spec = pricing.pricing_spec || pricing
  if (spec.basis !== 'token_usage') return true
  const expected = modelId.trim().toLocaleLowerCase()
  return Boolean(expected) && (pricing.metadata?.model_ids || spec.metadata?.model_ids || []).some(
    (item: unknown) => String(item).trim() === '*' || String(item).trim().toLocaleLowerCase() === expected,
  )
}

const pricingAppliesToSource = (pricing: Json, sourceSpecId: string): boolean => {
  const spec = pricing.pricing_spec || pricing
  if (spec.basis !== 'token_usage') return true
  const expected = sourceSpecId.trim().toLocaleLowerCase()
  return Boolean(expected) && (pricing.metadata?.source_spec_ids || spec.metadata?.source_spec_ids || []).some(
    (item: unknown) => String(item).trim() === '*' || String(item).trim().toLocaleLowerCase() === expected,
  )
}

const defaultRequestTimeout = (): Json => ({
  mode: 'adaptive',
  minimum_s: 120,
  maximum_s: 1800,
  base_s: 30,
  initial_generation_tokens_per_second: 20,
  observed_tokens_per_second_ceiling: 40,
  safety_factor: 1.5,
  ewma_alpha: 0.25,
})

function DeploymentManagement({ bootstrap, environment, notify }: Omit<Props, 'workspace'>) {
  const initial = bootstrap.deployments?.specs?.[0] || defaultDeploymentSpec(bootstrap)
  const initialDeploymentInstance = (bootstrap.deployments?.instances || []).find(
    (item: Json) => item.deployment_spec_id === initial.id,
  )
  const [catalog, setCatalog] = useState<Json>(bootstrap.deployments || { specs: [], instances: [] })
  const [pricingCatalog, setPricingCatalog] = useState<Json>(
    bootstrap.pricing_registry || { specs: [], instances: [] },
  )
  const [draft, setDraft] = useState<Json>(structuredClone(initial))
  const [capabilitiesText, setCapabilitiesText] = useState<string>(
    JSON.stringify(initial.capabilities || { text_generation: true }, null, 2),
  )
  const [extraBodyText, setExtraBodyText] = useState<string>(
    initial.extra_body ? JSON.stringify(initial.extra_body, null, 2) : '',
  )
  const [libraryPathsText, setLibraryPathsText] = useState<string>(
    (initial.library_paths || []).join('\n'),
  )
  const [sourceInstanceId, setSourceInstanceId] = useState('')
  const [instanceId, setInstanceId] = useState(`instance-${initial.id}`)
  const [apiKey, setApiKey] = useState('')
  const modelSpecs = bootstrap.model_registry?.specs || []
  const modelInstances = bootstrap.model_registry?.instances || []
  const apiSpecs = bootstrap.api_registry?.specs || []
  const apiInstances = bootstrap.api_registry?.instances || []
  const sourceSpecs = draft.source_spec?.type === 'api' ? apiSpecs : modelSpecs
  const sourceInstances = (draft.source_spec?.type === 'api' ? apiInstances : modelInstances)
    .filter((item: Json) => (draft.source_spec?.type === 'api' ? item.api_spec_id : item.model_spec_id) === draft.source_spec?.id)
  const selectedSourceSpec = sourceSpecs.find((item: Json) => item.id === draft.source_spec?.id)
  const pricingModelId = String(
    (draft.source_spec?.type === 'api' ? selectedSourceSpec?.model_id : selectedSourceSpec?.name) || '',
  ).trim()
  const pricingInstances: Json[] = pricingCatalog.instances || []
  const actualPricingSpecs: Json[] = (pricingCatalog.specs || []).filter(
    (item: Json) => item.basis === actualPricingBasis(draft),
  )
  const allowsApiEquivalent = localInfrastructurePricing(draft)
  const actualPricingCandidates = pricingInstances.filter(
    (item) => item.pricing_spec_id === draft.actual_pricing_spec_id
      && pricingAppliesToModel(item, pricingModelId)
      && (actualPricingBasis(draft) !== 'token_usage'
        || pricingAppliesToSource(item, String(draft.source_spec?.id || ''))),
  )
  const equivalentPricingCandidates = pricingInstances.filter(
    (item) => item.pricing_spec_id === draft.api_equivalent_pricing_spec_id
      && pricingAppliesToModel(item, pricingModelId),
  )
  const [actualPricingInstanceId, setActualPricingInstanceId] = useState(
    initialDeploymentInstance?.actual_pricing_instance_id || actualPricingCandidates[0]?.id || '',
  )
  const [equivalentPricingInstanceId, setEquivalentPricingInstanceId] = useState(
    initialDeploymentInstance?.api_equivalent_pricing_instance_id
      || equivalentPricingCandidates[0]?.id
      || '',
  )
  const [busy, setBusy] = useState(false)

  const refresh = async (probe = false) => {
    const [deployments, pricing] = await Promise.all([
      api.deployments(probe),
      api.pricing(),
    ])
    setCatalog(deployments)
    setPricingCatalog(pricing)
  }
  const loadSpec = (spec: Json) => {
    const existing = (catalog.instances || []).find(
      (item: Json) => item.deployment_spec_id === spec.id,
    )
    setDraft(structuredClone(spec))
    setCapabilitiesText(JSON.stringify(spec.capabilities || { text_generation: true }, null, 2))
    setExtraBodyText(spec.extra_body ? JSON.stringify(spec.extra_body, null, 2) : '')
    setLibraryPathsText((spec.library_paths || []).join('\n'))
    setSourceInstanceId('')
    setInstanceId(`instance-${spec.id}`)
    setApiKey('')
    setActualPricingInstanceId(
      existing?.actual_pricing_instance_id
        || pricingInstances.find((item) => item.pricing_spec_id === spec.actual_pricing_spec_id
          && pricingAppliesToModel(item, String((spec.source_spec?.type === 'api'
              ? apiSpecs.find((source: Json) => source.id === spec.source_spec?.id)?.model_id
              : modelSpecs.find((source: Json) => source.id === spec.source_spec?.id)?.name) || ''))
          && (actualPricingBasis(spec) !== 'token_usage'
            || pricingAppliesToSource(item, String(spec.source_spec?.id || ''))))?.id
        || '',
    )
    setEquivalentPricingInstanceId(
      existing?.api_equivalent_pricing_instance_id
        || pricingInstances.find((item) => item.pricing_spec_id === spec.api_equivalent_pricing_spec_id
          && pricingAppliesToModel(item, String((spec.source_spec?.type === 'api'
              ? apiSpecs.find((source: Json) => source.id === spec.source_spec?.id)?.model_id
              : modelSpecs.find((source: Json) => source.id === spec.source_spec?.id)?.name) || '')))?.id
        || '',
    )
  }

  const deploymentPayload = () => {
    let capabilities: Json
    let extraBody: Json | undefined
    try {
      capabilities = JSON.parse(capabilitiesText)
    } catch {
      throw new Error('Capabilities 必须是合法 JSON')
    }
    if (!capabilities || Array.isArray(capabilities) || typeof capabilities !== 'object') {
      throw new Error('Capabilities 必须是 JSON object')
    }
    if (extraBodyText.trim()) {
      try {
        extraBody = JSON.parse(extraBodyText)
      } catch {
        throw new Error('Extra request body 必须是合法 JSON')
      }
      if (!extraBody || Array.isArray(extraBody) || typeof extraBody !== 'object') {
        throw new Error('Extra request body 必须是 JSON object')
      }
    }
    const libraryPaths = libraryPathsText.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)
    const reasoningConfig = Object.fromEntries(
      Object.entries(draft.reasoning_config || {}).filter(([, value]) => value != null && value !== ''),
    )
    return {
      ...draft,
      capabilities,
      ...(extraBody ? { extra_body: extraBody } : { extra_body: undefined }),
      ...(libraryPaths.length ? { library_paths: libraryPaths } : { library_paths: undefined }),
      ...(Object.keys(reasoningConfig).length ? { reasoning_config: reasoningConfig } : { reasoning_config: undefined }),
      ...(!allowsApiEquivalent ? { api_equivalent_pricing_spec_id: null } : {}),
      schema_version: 2,
    }
  }

  const chooseKind = (kind: string) => {
    const managed = kind !== 'api'
    const sourceType = kind === 'api' ? 'api' : 'model'
    const candidates = sourceType === 'api' ? apiSpecs : modelSpecs
    const selectedSource = candidates[0]
    const modelId = String(
      (sourceType === 'api' ? selectedSource?.model_id : selectedSource?.name) || '',
    )
    const actualSpec = preferredActualPricingSpec(
      pricingCatalog.specs || [],
      kind === 'api' ? 'token_usage' : 'allocated_gpu_time',
      modelId,
      String(selectedSource?.id || ''),
    )
    const equivalentSpec = kind === 'api' ? null : preferredActualPricingSpec(
      pricingCatalog.specs || [],
      'token_usage',
      modelId,
    )
    const next = { ...draft }
    for (const field of ['tensor_parallel_size', 'data_parallel_size', 'max_num_seqs', 'per_request_metrics_mode', 'prompt_tokens_details_mode', 'reasoning_parser', 'reasoning_config', 'tool_call_parser', 'enable_auto_tool_choice', 'enable_prefix_caching', 'use_flashinfer_sampler', 'enforce_eager']) {
      delete next[field]
    }
    setDraft({
      ...next,
      kind,
      managed: kind !== 'api',
      ...(kind === 'vllm' ? { shared: false, host: '127.0.0.1', port: 8000, tensor_parallel_size: 1, data_parallel_size: 1, per_request_metrics_mode: 'auto', prompt_tokens_details_mode: 'auto' } : {}),
      source_spec: { type: sourceType, id: candidates[0]?.id || '' },
      actual_pricing_spec_id: actualSpec?.id || '',
      api_equivalent_pricing_spec_id: equivalentSpec?.id || null,
      ...(managed ? { python: draft.python || environment.python } : {}),
    })
    setActualPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === actualSpec?.id
        && pricingAppliesToModel(item, modelId)
        && (actualSpec?.basis !== 'token_usage'
          || pricingAppliesToSource(item, String(selectedSource?.id || ''))))?.id || '',
    )
    setEquivalentPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === equivalentSpec?.id
        && pricingAppliesToModel(item, modelId))?.id || '',
    )
    setSourceInstanceId('')
  }

  const chooseSourceSpec = (sourceSpecId: string) => {
    const sourceSpec = sourceSpecs.find((item: Json) => item.id === sourceSpecId)
    const modelId = String(
      (draft.source_spec?.type === 'api' ? sourceSpec?.model_id : sourceSpec?.name) || '',
    )
    const actualSpec = preferredActualPricingSpec(
      pricingCatalog.specs || [],
      actualPricingBasis(draft),
      modelId,
      sourceSpecId,
    )
    const equivalentSpec = allowsApiEquivalent ? preferredActualPricingSpec(
      pricingCatalog.specs || [],
      'token_usage',
      modelId,
    ) : null
    setDraft({
      ...draft,
      source_spec: { ...draft.source_spec, id: sourceSpecId },
      actual_pricing_spec_id: actualSpec?.id || '',
      api_equivalent_pricing_spec_id: equivalentSpec?.id || null,
    })
    setActualPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === actualSpec?.id
        && pricingAppliesToModel(item, modelId))?.id || '',
    )
    setEquivalentPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === equivalentSpec?.id
        && pricingAppliesToModel(item, modelId))?.id || '',
    )
    setSourceInstanceId('')
  }

  const chooseVllmMode = (managed: boolean) => {
    const sourceType = managed ? 'model' : 'api'
    const candidates = sourceType === 'model' ? modelSpecs : apiSpecs
    const selectedSource = candidates[0]
    const modelId = String(
      (sourceType === 'api' ? selectedSource?.model_id : selectedSource?.name) || '',
    )
    const actualSpec = preferredActualPricingSpec(
      pricingCatalog.specs || [],
      managed ? 'allocated_gpu_time' : 'token_usage',
      modelId,
      String(selectedSource?.id || ''),
    )
    const equivalentSpec = managed ? preferredActualPricingSpec(
      pricingCatalog.specs || [],
      'token_usage',
      modelId,
    ) : null
    setDraft({
      ...draft,
      managed,
      source_spec: { type: sourceType, id: candidates[0]?.id || '' },
      actual_pricing_spec_id: actualSpec?.id || '',
      api_equivalent_pricing_spec_id: equivalentSpec?.id || null,
    })
    setActualPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === actualSpec?.id
        && pricingAppliesToModel(item, modelId)
        && (actualSpec?.basis !== 'token_usage'
          || pricingAppliesToSource(item, String(selectedSource?.id || ''))))?.id || '',
    )
    setEquivalentPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === equivalentSpec?.id
        && pricingAppliesToModel(item, modelId))?.id || '',
    )
    setSourceInstanceId('')
  }

  const saveSpec = async () => {
    setBusy(true)
    try {
      const saved = await api.saveDeployment(draft.id, deploymentPayload())
      await refresh()
      setDraft(saved)
      setCapabilitiesText(JSON.stringify(saved.capabilities || {}, null, 2))
      setExtraBodyText(saved.extra_body ? JSON.stringify(saved.extra_body, null, 2) : '')
      setLibraryPathsText((saved.library_paths || []).join('\n'))
      setInstanceId(`instance-${saved.id}`)
      notify(`DeploymentSpec ${saved.id} 已保存`)
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setBusy(false) }
  }

  const removeSpec = async () => {
    if (!window.confirm(`确认删除 DeploymentSpec ${draft.id}？`)) return
    try { await api.deleteDeployment(draft.id); await refresh(); notify(`DeploymentSpec ${draft.id} 已删除`) }
    catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const instantiate = async () => {
    setBusy(true)
    try {
      await api.saveDeployment(draft.id, deploymentPayload())
      const saved = await api.deployDeployment(draft.id, {
        instance_id: instanceId,
        source_instance_id: sourceInstanceId,
        actual_pricing_instance_id: actualPricingInstanceId,
        ...(allowsApiEquivalent && equivalentPricingInstanceId
          ? { api_equivalent_pricing_instance_id: equivalentPricingInstanceId }
          : {}),
        ...(apiKey ? { api_key: apiKey } : {}),
      })
      await refresh(true)
      setApiKey('')
      notify(`DeploymentInstance ${saved.id} 已实例化并完成最小调用检查`)
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setBusy(false) }
  }

  const removeInstance = async (item: Json) => {
    if (!window.confirm(`确认删除 DeploymentInstance ${item.id}？`)) return
    try { await api.deleteDeploymentInstance(item.id); await refresh(); notify(`DeploymentInstance ${item.id} 已删除`) }
    catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const patchNumber = (field: string, raw: string) => setDraft({ ...draft, [field]: raw === '' ? undefined : Number(raw) })
  const patchRequestTimeout = (field: string, raw: string | number) => setDraft({
    ...draft,
    request_timeout: {
      ...defaultRequestTimeout(),
      ...(draft.request_timeout || {}),
      [field]: typeof raw === 'string' && field !== 'mode' ? Number(raw) : raw,
    },
  })
  const patchRequestLimits = (field: string, raw: string | number) => setDraft({
    ...draft,
    request_limits: {
      max_concurrency: 0,
      min_interval_s: 0,
      scope: 'process',
      ...(draft.request_limits || {}),
      [field]: typeof raw === 'string' && field !== 'scope' ? Number(raw) : raw,
    },
  })
  const patchReasoningConfig = (field: string, raw: string) => setDraft({
    ...draft,
    reasoning_config: {
      ...(draft.reasoning_config || {}),
      [field]: raw || undefined,
    },
  })

  return <section className="studio-view">
    <div className="deployment-workspace management-workspace">
      <section className="deployment-spec-pane management-spec-pane">
        <header className="management-spec-header">
          <div className="section-toolbar"><div><h1>DeploymentSpec</h1><span>声明部署配方并引用 ModelSpec 或 APISpec，不绑定资源实例</span></div><div className="toolbar-actions"><button className="icon-button" title="新建 DeploymentSpec" onClick={() => loadSpec(defaultDeploymentSpec(bootstrap))}><Plus size={16} /></button><button className="secondary" disabled={busy} onClick={saveSpec}><Save size={15} />保存</button><button className="icon-button danger" title="删除 DeploymentSpec" onClick={removeSpec}><Trash2 size={15} /></button></div></div>
          <SpecSelector kind="DeploymentSpec" specs={catalog.specs || []} draftId={draft.id || ''} optionLabel={(item) => `${item.id} · ${item.kind}`} onLoad={loadSpec} />
        </header>
        <div className="deployment-detail-grid">
          <label><span>DeploymentSpec ID</span><input value={draft.id || ''} onChange={(event) => setDraft({ ...draft, id: event.target.value })} /></label>
          <label><span>Backend kind</span><select value={draft.kind || 'hf'} onChange={(event) => chooseKind(event.target.value)}><option value="hf">Local HF</option><option value="vllm">vLLM</option><option value="api">OpenAI-compatible API</option></select></label>
          {draft.kind === 'vllm' && <label><span>vLLM 服务类型</span><select value={draft.managed === false ? 'external' : 'managed'} onChange={(event) => chooseVllmMode(event.target.value === 'managed')}><option value="managed">由 Studio 启动本地服务</option><option value="external">连接已注册 API 服务</option></select></label>}
          <label><span>Source Spec type</span><input disabled value={draft.source_spec?.type || ''} /></label>
          <label><span>{draft.source_spec?.type === 'api' ? 'APISpec' : 'ModelSpec'}</span><select value={draft.source_spec?.id || ''} onChange={(event) => chooseSourceSpec(event.target.value)}>{sourceSpecs.map((item: Json) => <option key={item.id} value={item.id}>{item.name || item.id} · {item.id}</option>)}</select></label>
          <label><span>实际成本 PricingSpec</span><select value={draft.actual_pricing_spec_id || ''} onChange={(event) => { const id = event.target.value; setDraft({ ...draft, actual_pricing_spec_id: id }); setActualPricingInstanceId(pricingInstances.find((item) => item.pricing_spec_id === id)?.id || '') }}>{actualPricingSpecs.map((item: Json) => <option key={item.id} value={item.id}>{item.id} · {item.billing_mode}</option>)}</select><small>{allowsApiEquivalent ? '本地算力按运行墙钟时间 × 实际分配 GPU 数计费。' : '外部服务按 provider 返回的 token usage 计费。'}</small></label>
          {allowsApiEquivalent && <label><span>API 等价成本 PricingSpec</span><select value={draft.api_equivalent_pricing_spec_id || ''} onChange={(event) => { const id = event.target.value; setDraft({ ...draft, api_equivalent_pricing_spec_id: id || null }); setEquivalentPricingInstanceId(pricingInstances.find((item) => item.pricing_spec_id === id)?.id || '') }}><option value="">不计算 API 等价成本</option>{(pricingCatalog.specs || []).filter((item: Json) => item.basis === 'token_usage').map((item: Json) => <option key={item.id} value={item.id}>{item.id}</option>)}</select><small>可选：用同一次运行的 token usage 估算对标云 API 费用。</small></label>}
          {(draft.kind === 'hf' || (draft.kind === 'vllm' && draft.managed !== false)) && <><label className="wide-field"><span>Python executable</span><input value={draft.python || environment.python || ''} onChange={(event) => setDraft({ ...draft, python: event.target.value })} /></label><label><span>{draft.kind === 'hf' ? 'Physical CUDA device' : 'CUDA visible devices'}</span><input value={draft.cuda_visible_devices || '0'} onChange={(event) => setDraft({ ...draft, cuda_visible_devices: event.target.value })} /><small>{draft.kind === 'hf' ? '每个 Local HF DeploymentInstance 分配一张物理 GPU；实例化团队后自动映射为进程内 cuda:N。' : '按 TP × DP 提供本地 vLLM 服务所需物理 GPU。'}</small></label></>}
          {draft.kind === 'vllm' && draft.managed !== false && <><label><span>Host</span><input value={draft.host || '127.0.0.1'} onChange={(event) => setDraft({ ...draft, host: event.target.value })} /></label><label><span>Port</span><input type="number" value={draft.port || 8000} onChange={(event) => patchNumber('port', event.target.value)} /></label><label><span>Tensor parallel</span><input type="number" min="1" value={draft.tensor_parallel_size || 1} onChange={(event) => patchNumber('tensor_parallel_size', event.target.value)} /><small>每个模型副本共同使用的 GPU 数。</small></label><label><span>Data parallel</span><input type="number" min="1" value={draft.data_parallel_size || 1} onChange={(event) => patchNumber('data_parallel_size', event.target.value)} /><small>完整模型副本数；至少需要 TP × DP 张可见 GPU。</small></label><label><span>Max model length</span><input type="number" min="1" value={draft.max_model_len || 32768} onChange={(event) => patchNumber('max_model_len', event.target.value)} /></label><label><span>GPU memory utilization</span><input type="number" min="0" max="1" step="0.01" value={draft.gpu_memory_utilization || 0.9} onChange={(event) => patchNumber('gpu_memory_utilization', event.target.value)} /></label><label><span>Max concurrent sequences / replica</span><input type="number" min="1" value={draft.max_num_seqs || 1} onChange={(event) => patchNumber('max_num_seqs', event.target.value)} /><small>按每个 DP replica 生效，总上限约为该值 × DP。</small></label><label><span>Data type</span><select value={draft.dtype || 'auto'} onChange={(event) => setDraft({ ...draft, dtype: event.target.value })}><option value="auto">auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option></select></label><label><span>Per-request metrics</span><select value={draft.per_request_metrics_mode || 'auto'} onChange={(event) => setDraft({ ...draft, per_request_metrics_mode: event.target.value })}><option value="auto">Auto · CLI 支持时开启</option><option value="enabled">Enabled · 不支持则拒绝部署</option><option value="disabled">Disabled</option></select><small>控制 vLLM 是否在每次 OpenAI-compatible 响应中附带 queue、TTFT、generation 和 ITL 指标。</small></label><label><span>Prompt token details</span><select value={draft.prompt_tokens_details_mode || 'auto'} onChange={(event) => setDraft({ ...draft, prompt_tokens_details_mode: event.target.value })}><option value="auto">Auto · CLI 支持时开启</option><option value="enabled">Enabled · 不支持则拒绝部署</option><option value="disabled">Disabled</option></select><small>输出每次请求的 <code>usage.prompt_tokens_details.cached_tokens</code>，用于精确记录 Prefix Cache 命中 Token。</small></label></>}
          {(draft.kind === 'api' || draft.kind === 'vllm') && <>
            <label><span>Request timeout mode</span><select value={draft.request_timeout?.mode || 'adaptive'} onChange={(event) => patchRequestTimeout('mode', event.target.value)}><option value="adaptive">Adaptive</option><option value="fixed">Fixed</option></select></label>
            <label><span>Minimum timeout (s)</span><input type="number" min="1" value={draft.request_timeout?.minimum_s ?? 120} onChange={(event) => patchRequestTimeout('minimum_s', event.target.value)} /></label>
            <label><span>Maximum timeout (s)</span><input type="number" min="1" value={draft.request_timeout?.maximum_s ?? 1800} onChange={(event) => patchRequestTimeout('maximum_s', event.target.value)} /></label>
            <label><span>Base timeout (s)</span><input type="number" min="0" value={draft.request_timeout?.base_s ?? 30} onChange={(event) => patchRequestTimeout('base_s', event.target.value)} /></label>
            <label><span>Initial generation speed (tok/s)</span><input type="number" min="0.1" step="0.1" value={draft.request_timeout?.initial_generation_tokens_per_second ?? 20} onChange={(event) => patchRequestTimeout('initial_generation_tokens_per_second', event.target.value)} /></label>
            <label><span>Observed speed ceiling (tok/s)</span><input type="number" min="0.1" step="0.1" value={draft.request_timeout?.observed_tokens_per_second_ceiling ?? 40} onChange={(event) => patchRequestTimeout('observed_tokens_per_second_ceiling', event.target.value)} /></label>
            <label><span>Timeout safety factor</span><input type="number" min="1" step="0.1" value={draft.request_timeout?.safety_factor ?? 1.5} onChange={(event) => patchRequestTimeout('safety_factor', event.target.value)} /></label>
            <label><span>Speed EWMA alpha</span><input type="number" min="0" max="1" step="0.05" value={draft.request_timeout?.ewma_alpha ?? 0.25} onChange={(event) => patchRequestTimeout('ewma_alpha', event.target.value)} /></label>
            <label><span>SDK automatic retries</span><input type="number" min="0" value={draft.max_retries ?? 0} onChange={(event) => patchNumber('max_retries', event.target.value)} /></label>
            <label><span>Client max concurrency</span><input type="number" min="0" value={draft.request_limits?.max_concurrency ?? 0} onChange={(event) => patchRequestLimits('max_concurrency', event.target.value)} /><small>0 表示不在客户端额外限流；vLLM 服务仍按 max sequences 调度。</small></label>
            <label><span>Minimum request interval (s)</span><input type="number" min="0" step="0.1" value={draft.request_limits?.min_interval_s ?? 0} onChange={(event) => patchRequestLimits('min_interval_s', event.target.value)} /></label>
            <label><span>Request limit scope</span><select value={draft.request_limits?.scope || 'process'} onChange={(event) => patchRequestLimits('scope', event.target.value)}><option value="process">process</option><option value="host">host</option></select></label>
          </>}
          {draft.kind === 'vllm' && draft.managed !== false && <div className="wide-field deployment-advanced-settings">
            <div className="panel-heading"><h2>vLLM 高级启动参数</h2><span>均直接映射到当前托管服务命令</span></div>
            <div className="deployment-detail-grid embedded-grid">
              <label><span>Reasoning parser</span><input value={draft.reasoning_parser || ''} placeholder="例如 qwen3" onChange={(event) => setDraft({ ...draft, reasoning_parser: event.target.value || undefined })} /></label>
              <label><span>Tool call parser</span><input value={draft.tool_call_parser || ''} placeholder="例如 qwen3_coder" onChange={(event) => setDraft({ ...draft, tool_call_parser: event.target.value || undefined })} /></label>
              <label><span>Reasoning start string</span><input value={draft.reasoning_config?.reasoning_start_str || ''} placeholder="可选" onChange={(event) => patchReasoningConfig('reasoning_start_str', event.target.value)} /></label>
              <label><span>Reasoning end string</span><input value={draft.reasoning_config?.reasoning_end_str || ''} placeholder="启用 reasoning config 时必填" onChange={(event) => patchReasoningConfig('reasoning_end_str', event.target.value)} /></label>
              <label className="check-field"><input type="checkbox" checked={Boolean(draft.enable_auto_tool_choice)} onChange={(event) => setDraft({ ...draft, enable_auto_tool_choice: event.target.checked })} /><span>Enable auto tool choice</span></label>
              <label className="check-field"><input type="checkbox" checked={Boolean(draft.enable_prefix_caching)} onChange={(event) => setDraft({ ...draft, enable_prefix_caching: event.target.checked })} /><span>Enable prefix caching</span></label>
              <label className="check-field"><input type="checkbox" checked={draft.use_flashinfer_sampler !== false} onChange={(event) => setDraft({ ...draft, use_flashinfer_sampler: event.target.checked })} /><span>Use FlashInfer sampler</span><small>关闭时使用 vLLM 官方后备采样实现，可规避首次 JIT 的 CUDA 工具链不兼容。</small></label>
              <label className="check-field"><input type="checkbox" checked={Boolean(draft.enforce_eager)} onChange={(event) => setDraft({ ...draft, enforce_eager: event.target.checked })} /><span>Enforce eager execution</span></label>
              <label className="wide-field"><span>Additional library paths</span><textarea rows={3} value={libraryPathsText} placeholder="每行一个 LD_LIBRARY_PATH 目录" onChange={(event) => setLibraryPathsText(event.target.value)} /></label>
            </div>
          </div>}
          {(draft.kind === 'api' || draft.kind === 'vllm') && <label className="wide-field"><span>Extra request body JSON</span><textarea rows={5} value={extraBodyText} placeholder="可选；合并到每次 OpenAI-compatible 请求" onChange={(event) => setExtraBodyText(event.target.value)} spellCheck={false} /></label>}
          <label className="wide-field"><span>Capabilities JSON</span><textarea rows={5} value={capabilitiesText} onChange={(event) => setCapabilitiesText(event.target.value)} spellCheck={false} /></label>
        </div>
        <div className="team-contract-note"><strong>Spec 边界</strong><span>这里保存的是可复用配方：后端类型、启动参数和 Source Spec。模型路径、API endpoint、凭据引用来自实例化时选择的 ModelInstance / APIInstance。</span><DetailsDisclosure value={draft} label="展开 DeploymentSpec" /></div>
      </section>

      <aside className="discovered-deployments deployment-instance-pane management-instance-pane">
        <section className="management-instance-builder">
          <div className="section-toolbar"><div><h1>DeploymentInstance</h1><span>为当前 DeploymentSpec 选择一个匹配的资源 Instance</span></div><button className="icon-button" title="刷新并执行健康检查" onClick={() => refresh(true)}><RefreshCw size={15} /></button></div>
          <label><span>Source Instance</span><select value={sourceInstanceId} onChange={(event) => setSourceInstanceId(event.target.value)}><option value="">请选择 {draft.source_spec?.type === 'api' ? 'APIInstance' : 'ModelInstance'}</option>{sourceInstances.map((item: Json) => <option key={item.id} value={item.id}>{item.id} · {draft.source_spec?.type === 'api' ? item.model_id : item.path}</option>)}</select></label>
          <label><span>DeploymentInstance ID</span><input value={instanceId} onChange={(event) => setInstanceId(event.target.value)} /></label>
          <label><span>实际成本 PricingInstance</span><select value={actualPricingInstanceId} onChange={(event) => setActualPricingInstanceId(event.target.value)}><option value="">请选择价格实例</option>{actualPricingCandidates.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.currency} · {item.status || 'registered'}</option>)}</select></label>
          {allowsApiEquivalent && draft.api_equivalent_pricing_spec_id && <label><span>API 等价成本 PricingInstance</span><select value={equivalentPricingInstanceId} onChange={(event) => setEquivalentPricingInstanceId(event.target.value)}><option value="">请选择同模型 API 价格实例</option>{equivalentPricingCandidates.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.currency} · {item.status || 'registered'}</option>)}</select></label>}
          {draft.source_spec?.type === 'api' && <label><span>API key（仅当前 Studio 进程，可选）</span><input type="password" autoComplete="new-password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="也可使用 APIInstance 登记的环境变量" /></label>}
          <div className="reference-notice"><strong>{draft.id}</strong><span>实例化时解析 <code>{draft.source_spec?.type}:{draft.source_spec?.id}</code> 对应的具体资源实例，并执行一次最小模型调用。只有健康检查结果属于 Instance。</span></div>
          <button className="primary" disabled={busy || !sourceInstanceId || !instanceId || !actualPricingInstanceId || Boolean(allowsApiEquivalent && draft.api_equivalent_pricing_spec_id && !equivalentPricingInstanceId)} onClick={instantiate}><ServerCog size={15} />实例化 Deployment</button>
        </section>
        <section className="management-instance-registry">
          <div className="panel-heading"><h2>已注册 DeploymentInstance</h2><span>{(catalog.instances || []).length}</span></div>
          {(catalog.instances || []).map((item: Json) => <article key={item.id} className="process-card">
            <div><span className={`status-pill ${isDeploymentReady(item) ? 'ready' : status(item)}`}>{status(item)}</span><strong>{item.model_id || item.id}</strong><small>{item.id} · {item.kind}</small>{item.base_url && <code>{item.base_url}</code>}{item.model_path && <code>{item.model_path}</code>}</div>
            <dl><dt>DeploymentSpec</dt><dd>{item.deployment_spec_id}</dd><dt>Source Instance</dt><dd>{item.source_instance?.type}:{item.source_instance?.id}</dd><dt>实际成本</dt><dd>{item.actual_pricing_instance_id} · {item.actual_pricing_status?.status}</dd><dt>计价基准</dt><dd>{item.actual_pricing_instance?.pricing_spec?.basis || 'unknown'}</dd><dt>API 等价成本</dt><dd>{item.api_equivalent_pricing_instance_id || '未配置'}</dd><dt>Lifecycle</dt><dd>{item.instance_type || 'unknown'}</dd>{item.probe_latency_s != null && <><dt>Probe latency</dt><dd>{item.probe_latency_s}s</dd></>}<dt>Reasoning token accounting</dt><dd>{item.reasoning_token_accounting || 'inconclusive'}</dd>{item.kind === 'vllm' && <><dt>Parallelism</dt><dd>TP {item.tensor_parallel_size || 1} × DP {item.data_parallel_size || 1}</dd><dt>Sequence capacity</dt><dd>{item.max_num_seqs || 1} / replica</dd><dt>Per-request metrics</dt><dd>{item.per_request_metrics_enabled ? 'enabled' : item.per_request_metrics_supported === false ? 'unsupported' : 'unknown'}</dd><dt>Prompt token details</dt><dd>{item.prompt_tokens_details_enabled ? 'enabled' : item.prompt_tokens_details_supported === false ? 'unsupported' : 'unknown'}</dd><dt>Service metrics</dt><dd>{item.metrics_url || 'auto /metrics'}</dd></>}</dl>
            {item.health_detail && <div className={`instance-diagnostic ${status(item) === 'auth_required' ? 'auth' : ''}`}><strong>健康检查</strong><span>{item.health_detail}</span></div>}
            {item.failure_reason && <div className="instance-failure"><strong>实例化失败</strong><span>{item.failure_reason}</span></div>}
            <DetailsDisclosure value={item} label="展开 DeploymentInstance" className="card-details" />
            <div className="row-actions"><button className="icon-button danger" title="删除 DeploymentInstance" onClick={() => removeInstance(item)}><Trash2 size={15} /></button></div>
          </article>)}
        </section>
      </aside>
    </div>
  </section>
}

export default function StudioView(props: Props) {
  return props.workspace === 'team'
    ? <TeamManagement bootstrap={props.bootstrap} notify={props.notify} />
    : <DeploymentManagement bootstrap={props.bootstrap} environment={props.environment} notify={props.notify} />
}
