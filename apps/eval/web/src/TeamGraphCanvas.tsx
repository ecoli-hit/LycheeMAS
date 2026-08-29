import {
  applyNodeChanges,
  Background,
  BaseEdge,
  ConnectionMode,
  Controls,
  EdgeLabelRenderer,
  getStraightPath,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type EdgeProps,
  type InternalNode,
  type NodeProps,
  useInternalNode,
  useUpdateNodeInternals,
} from '@xyflow/react'
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force'
import { Pause, Play, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  deriveCoordinationSummary,
  edgeCounts,
  EDGE_COLORS,
  EDGE_KINDS,
  EDGE_LABELS,
  projectEdges,
  sharedStateEdges,
  sharedStateNodeId,
  teamEdges,
  type EdgeProjection,
  type TeamGraphEdge,
} from './teamGraph'
import type { AgentNode, Json, TeamEdge } from './types'

interface ForceNode extends SimulationNodeDatum { id: string }
type ForceLink = SimulationLinkDatum<ForceNode>

interface GraphVisibility {
  executionNodes: boolean
  stateNodes: boolean
  controlEdges: boolean
  dataEdges: boolean
}

interface InteractiveProjectionData {
  projection: EdgeProjection
  selectedEdgeId: string
  onSelectEdge: (projectionId: string, edgeId: string) => void
  onSelectState: (stateId: string) => void
}

const NODE_SIZE = 112
const CENTER = { x: 420, y: 280 }
const PERIMETER_HANDLES = 48
const DEFAULT_VISIBILITY: GraphVisibility = {
  executionNodes: true,
  stateNodes: true,
  controlEdges: true,
  dataEdges: true,
}

const displayLabel = (value: unknown): string => String(value || '')
  .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
  .replace(/[_-]+/g, ' ')
  .replace(/\s+/g, ' ')
  .trim()

const initialPositions = (ids: string[]): Map<string, { x: number; y: number }> => {
  const result = new Map<string, { x: number; y: number }>()
  const radius = Math.max(170, Math.min(360, 88 * ids.length))
  ids.forEach((id: string, index: number) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(ids.length, 1)
    result.set(id, {
      x: CENTER.x + Math.cos(angle) * radius - NODE_SIZE / 2,
      y: CENTER.y + Math.sin(angle) * radius - NODE_SIZE / 2,
    })
  })
  return result
}

function graphModel(spec: Json, visibility: GraphVisibility) {
  const records = [...teamEdges(spec), ...sharedStateEdges(spec)]
  const nodeIds = (spec.nodes || []).map((node: Json) => String(node.id))
  const stateIds = (spec.shared_state || []).map((state: Json) => sharedStateNodeId(String(state.id)))
  const positions = initialPositions([...nodeIds, ...stateIds])
  const visibleNodeIds = new Set(
    [
      ...(visibility.executionNodes ? nodeIds : []),
      ...(visibility.stateNodes ? stateIds : []),
    ],
  )
  const visibleRecords = records.filter((edge) => (
    (edge.edgeKind === 'control' ? visibility.controlEdges : visibility.dataEdges)
    && visibleNodeIds.has(edge.source)
    && visibleNodeIds.has(edge.target)
  ))
  const projections = projectEdges(visibleRecords)
  const executionNodes: AgentNode[] = (spec.nodes || []).map((node: Json) => {
    return {
      id: String(node.id),
      type: 'agent',
      hidden: !visibility.executionNodes,
      position: positions.get(String(node.id)) || { x: 60, y: 60 },
      data: {
        label: displayLabel(node.name || node.id),
        role: String(node.name || node.id),
        visual_kind: String(node.kind || 'model_agent'),
        execution_kind: String(node.behavior?.type || node.kind || 'node'),
        node_scope: 'executable',
        system_prompt: String(node.instructions || ''),
        description: String(node.purpose || ''),
        tools: (node.tools || []).map((item: Json) => String(item.id)),
        model_context: node.context || { type: 'unbounded' },
        ports: [],
      },
    }
  })
  const stateNodes: AgentNode[] = (spec.shared_state || []).map((state: Json) => ({
    id: sharedStateNodeId(String(state.id)),
    type: 'agent',
    hidden: !visibility.stateNodes,
    position: positions.get(sharedStateNodeId(String(state.id))) || { x: 60, y: 60 },
    data: {
      label: displayLabel(state.id),
      role: String(state.id),
      visual_kind: String(state.kind || 'structured_state'),
      execution_kind: String(state.kind || 'shared_state'),
      node_scope: 'shared-state',
      system_prompt: '',
      description: String(state.description || ''),
      tools: [],
      ports: [],
    },
  }))
  const nodes = [...executionNodes, ...stateNodes]
  const edges: TeamEdge[] = projections.map((projection) => ({
    id: projection.id,
    source: projection.source,
    target: projection.target,
    type: 'teamEdge',
    data: { projection },
    style: { stroke: '#52615E', strokeWidth: 4.5 },
    ...(projection.forward
      ? { markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#52615E' } }
      : {}),
    ...(projection.reverse
      ? { markerStart: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#52615E' } }
      : {}),
  }))
  return { nodes, edges, projections, records }
}

const centerOf = (node: InternalNode<AgentNode>) => {
  const width = node.measured.width || NODE_SIZE
  const height = node.measured.height || NODE_SIZE
  return {
    x: node.internals.positionAbsolute.x + width / 2,
    y: node.internals.positionAbsolute.y + height / 2,
    radius: Math.min(width, height) / 2,
  }
}

const intersection = (node: InternalNode<AgentNode>, toward: InternalNode<AgentNode>) => {
  const source = centerOf(node)
  const target = centerOf(toward)
  const dx = target.x - source.x
  const dy = target.y - source.y
  const length = Math.hypot(dx, dy) || 1
  return {
    x: source.x + (dx / length) * source.radius,
    y: source.y + (dy / length) * source.radius,
  }
}

function TeamGraphEdgeView({
  id, source, target, markerStart, markerEnd, interactionWidth, selected, data,
}: EdgeProps) {
  const sourceNode = useInternalNode<AgentNode>(source)
  const targetNode = useInternalNode<AgentNode>(target)
  if (!sourceNode || !targetNode) return null
  const sourcePoint = intersection(sourceNode, targetNode)
  const targetPoint = intersection(targetNode, sourceNode)
  const [path, labelX, labelY] = getStraightPath({
    sourceX: sourcePoint.x,
    sourceY: sourcePoint.y,
    targetX: targetPoint.x,
    targetY: targetPoint.y,
  })
  const projectionData = data as unknown as InteractiveProjectionData
  const projection = projectionData?.projection
  const edgeCount = Math.max(projection?.edges.length || 0, 1)
  return <>
    <BaseEdge
      id={`${id}-direction`}
      path={path}
      markerStart={markerStart}
      markerEnd={markerEnd}
      interactionWidth={0}
      style={{ stroke: '#52615E', strokeWidth: 1.2, opacity: 0.42 }}
    />
    <g className={`relation-edge${selected ? ' selected' : ''}`}>
      {(projection?.edges || []).map((edge, index) => {
        const edgeSelected = Boolean(edge.relationId)
          && projectionData.selectedEdgeId === edge.relationId
        const segment = 1 / edgeCount
        return <path
          key={edge.id}
          data-relation-id={edge.id}
          role="button"
          aria-label={`选择 Relation ${edge.id}`}
          className={`relation-edge-segment relation-${edge.edgeKind}${edgeSelected ? ' selected' : ''}${edge.enabled ? '' : ' disabled'}`}
          d={path}
          pathLength={1}
          fill="none"
          stroke={EDGE_COLORS[edge.edgeKind]}
          strokeWidth={edgeSelected ? 7 : 4.5}
          strokeDasharray={edgeCount === 1 ? undefined : `${segment} ${1 - segment}`}
          strokeDashoffset={edgeCount === 1 ? undefined : -index * segment}
          opacity={edge.enabled ? 1 : 0.38}
          pointerEvents="stroke"
          onClick={(event) => {
            event.stopPropagation()
            if (edge.relationId) projectionData.onSelectEdge(id, edge.relationId)
            else if (edge.stateId) projectionData.onSelectState(edge.stateId)
          }}
        />
      })}
      <path d={path} fill="none" stroke="transparent" strokeWidth={interactionWidth || 20} pointerEvents="none" />
    </g>
    {projection?.label && <EdgeLabelRenderer>
      <div
        className={`relation-edge-label${selected ? ' selected' : ''}`}
        style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
      >{projection.label}</div>
    </EdgeLabelRenderer>}
  </>
}

function GraphNodeView({ id, data }: NodeProps<AgentNode>) {
  const updateNodeInternals = useUpdateNodeInternals()
  useEffect(() => updateNodeInternals(id), [id, updateNodeInternals])
  return <div className={`agent-node-inner node-scope-${data.node_scope} node-kind-${data.visual_kind}`}>
    {data.node_scope === 'executable' && Array.from({ length: PERIMETER_HANDLES }, (_, index) => {
      const angle = (index / PERIMETER_HANDLES) * Math.PI * 2
      const radius = NODE_SIZE / 2
      return <Handle
        key={index}
        id={`connect-${index}`}
        type="source"
        position={Position.Top}
        className="graph-perimeter-handle"
        style={{
          left: `${NODE_SIZE / 2 + Math.cos(angle) * radius}px`,
          top: `${NODE_SIZE / 2 + Math.sin(angle) * radius}px`,
        }}
      />
    })}
    <strong title={data.role}>{String(data.label)}</strong>
    <small>{displayLabel(data.node_scope)} · {displayLabel(data.execution_kind)}</small>
  </div>
}

const nodeTypes = { agent: GraphNodeView }
const edgeTypes = { teamEdge: TeamGraphEdgeView }

interface Props {
  spec: Json
  selectedNodeId: string
  selectedEdgeId: string
  onNodeSelect: (nodeId: string) => void
  onEdgeSelect: (projectionId: string, edgeId: string) => void
  onStateSelect: (stateId: string) => void
  onConnect: (source: string, target: string) => void
}

export default function TeamGraphCanvas({
  spec, selectedNodeId, selectedEdgeId, onNodeSelect, onEdgeSelect, onStateSelect, onConnect,
}: Props) {
  const [visibility, setVisibility] = useState<GraphVisibility>(DEFAULT_VISIBILITY)
  const model = useMemo(() => graphModel(spec, visibility), [spec, visibility])
  const counts = useMemo(() => edgeCounts(model.records), [model.records])
  const [nodes, setNodes] = useState(model.nodes)
  const [continuous, setContinuous] = useState(false)
  const [strength, setStrength] = useState(-1200)
  const [distance, setDistance] = useState(250)
  const simulation = useRef<Simulation<ForceNode, ForceLink> | null>(null)
  const strengthRef = useRef(strength)
  const distanceRef = useRef(distance)
  const structureKey = useMemo(() => [
    ...model.nodes.map((node) => node.id),
    ...model.records.map((edge) => `${edge.id}:${edge.source}:${edge.target}`),
  ].join('|'), [model.nodes, model.records])
  const previousStructureKey = useRef('')

  const runLayout = (
    seedNodes: AgentNode[] = nodes,
    keepRunning = continuous,
    nextStrength = strengthRef.current,
    nextDistance = distanceRef.current,
  ) => {
    simulation.current?.stop()
    const forceNodes: ForceNode[] = seedNodes.map((node) => ({
      id: node.id,
      x: node.position.x + NODE_SIZE / 2,
      y: node.position.y + NODE_SIZE / 2,
    }))
    const links: ForceLink[] = model.records.map((edge: TeamGraphEdge) => ({
      source: edge.source,
      target: edge.target,
    }))
    const byId = new Map(forceNodes.map((node) => [node.id, node]))
    const next = forceSimulation(forceNodes)
      .force('charge', forceManyBody().strength(nextStrength))
      .force('link', forceLink<ForceNode, ForceLink>(links).id((node) => node.id).distance(nextDistance).strength(0.7))
      .force('center', forceCenter(CENTER.x, CENTER.y))
      .force('collision', forceCollide(NODE_SIZE / 2 + 20))
      .alpha(1)
      .alphaDecay(keepRunning ? 0 : 0.025)
      .on('tick', () => setNodes((current) => current.map((node) => {
        const forceNode = byId.get(node.id)
        return forceNode ? {
          ...node,
          position: { x: Number(forceNode.x) - NODE_SIZE / 2, y: Number(forceNode.y) - NODE_SIZE / 2 },
        } : node
      })))
    if (keepRunning) next.alphaTarget(0.08).restart()
    simulation.current = next
  }

  useEffect(() => {
    const structureChanged = previousStructureKey.current !== structureKey
    previousStructureKey.current = structureKey
    setNodes((current) => {
      const positions = new Map(current.map((node) => [node.id, node.position]))
      const nextNodes = model.nodes.map((node) => ({
        ...node,
        position: structureChanged ? node.position : (positions.get(node.id) || node.position),
      }))
      if (structureChanged) window.setTimeout(() => runLayout(nextNodes, continuous), 0)
      return nextNodes
    })
    // Visibility only changes `hidden`; the full structure remains in the simulation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.nodes, structureKey])

  useEffect(() => () => { simulation.current?.stop() }, [])
  const summary = deriveCoordinationSummary(spec)
  const setVisible = (key: keyof GraphVisibility, checked: boolean) => {
    setVisibility((current) => ({ ...current, [key]: checked }))
  }

  return <section className="graph-panel">
    <header className="graph-visibility-toolbar">
      <div><strong>有状态团队执行图</strong><small>Stateful Team Execution Graph</small></div>
      <fieldset aria-label="图元素可见性">
        <label><input type="checkbox" checked={visibility.executionNodes} onChange={(event) => setVisible('executionNodes', event.target.checked)} /><i className="node executable" /><span>执行 Node</span><em>{(spec.nodes || []).length}</em></label>
        <label><input type="checkbox" checked={visibility.stateNodes} onChange={(event) => setVisible('stateNodes', event.target.checked)} /><i className="node store" /><span>SharedState</span><em>{(spec.shared_state || []).length}</em></label>
        <label><input type="checkbox" checked={visibility.controlEdges} onChange={(event) => setVisible('controlEdges', event.target.checked)} /><i className="relation-family-dot control" /><span>Control Edge</span><em>{counts.control}</em></label>
        <label><input type="checkbox" checked={visibility.dataEdges} onChange={(event) => setVisible('dataEdges', event.target.checked)} /><i className="relation-family-dot data" /><span>Data Edge</span><em>{counts.data}</em></label>
      </fieldset>
    </header>
    <div className="relation-semantics-summary">
      <span><i className="relation-family-dot control" /><strong>Control</strong> 决定激活、路由与停止</span>
      <span><i className="relation-family-dot data" /><strong>Data</strong> 传递 task、message、state、artifact 与 result</span>
      <small>{summary.topology} · {summary.label} · 开关只改变可见性，不改变 TeamSpec、布局关系或执行语义。</small>
    </div>
    <div className="force-layout-toolbar">
      <button title={continuous ? '暂停持续模拟' : '开启持续模拟'} onClick={() => {
        const next = !continuous
        setContinuous(next)
        if (next) runLayout(nodes, true)
        else simulation.current?.alphaTarget(0).stop()
      }}>{continuous ? <Pause size={14} /> : <Play size={14} />}</button>
      <button title="重置并重新布局" onClick={() => {
        const reset = graphModel(spec, visibility).nodes
        setNodes(reset)
        window.setTimeout(() => runLayout(reset, continuous), 0)
      }}><RefreshCw size={14} /></button>
      <label><span>Strength</span><input aria-label="Strength" type="range" min="-8000" max="-50" step="50" value={strength} onInput={(event) => {
        const value = Number(event.currentTarget.value)
        setStrength(value)
        strengthRef.current = value
        runLayout(nodes, continuous, value, distanceRef.current)
      }} /><em>{strength}</em></label>
      <label><span>Distance</span><input aria-label="Distance" type="range" min="80" max="1400" step="10" value={distance} onInput={(event) => {
        const value = Number(event.currentTarget.value)
        setDistance(value)
        distanceRef.current = value
        runLayout(nodes, continuous, strengthRef.current, value)
      }} /><em>{distance}</em></label>
    </div>
    <div className="team-graph-canvas">
      <ReactFlow
        nodes={nodes.map((node) => ({ ...node, selected: node.id === selectedNodeId }))}
        edges={model.edges.map((edge) => {
          const projection = (edge.data as Json)?.projection as EdgeProjection
          return {
            ...edge,
            selected: projection.edges.some((item) => item.relationId === selectedEdgeId),
            data: {
              ...edge.data,
              selectedEdgeId,
              onSelectEdge: onEdgeSelect,
              onSelectState: onStateSelect,
            },
          }
        })}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionMode={ConnectionMode.Loose}
        onNodesChange={(changes) => setNodes((current) => applyNodeChanges(changes, current))}
        onNodeClick={(_, node) => {
          if (node.id.startsWith('shared-state::')) onStateSelect(node.id.slice('shared-state::'.length))
          else onNodeSelect(node.id)
        }}
        onEdgeClick={(_, edge) => {
          const projection = (edge.data as Json)?.projection as EdgeProjection
          const record = projection?.edges[0]
          if (record?.relationId) onEdgeSelect(edge.id, record.relationId)
          else if (record?.stateId) onStateSelect(record.stateId)
        }}
        onConnect={(connection) => {
          if (connection.source && connection.target
            && !connection.source.startsWith('shared-state::')
            && !connection.target.startsWith('shared-state::')
            && connection.source !== connection.target) {
            onConnect(connection.source, connection.target)
          }
        }}
        fitView
        fitViewOptions={{ padding: 0.2, includeHiddenNodes: true }}
        minZoom={0.15}
        maxZoom={2}
      >
        <Background gap={24} size={1} color="#d9e1df" />
        <Controls />
      </ReactFlow>
      <div className="team-graph-legend">
        <span><i className="node-executable" />Executable Node</span>
        <span><i className="node-store" />SharedState</span>
        {EDGE_KINDS.map((kind) => <span key={kind}><i className={kind} />{EDGE_LABELS[kind]} Edge</span>)}
      </div>
    </div>
  </section>
}
