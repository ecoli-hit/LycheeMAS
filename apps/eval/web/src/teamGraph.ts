import type { Json } from './types'

export type EdgeKind = 'control' | 'data'

export interface TeamGraphEdge {
  id: string
  relationId: string
  source: string
  sourcePort: string
  target: string
  targetPort: string
  edgeKind: EdgeKind
  semanticType: string
  operation: string
  edgeIndex: number
  enabled: boolean
  description: string
  policy: Json
  stateId?: string
  derived?: boolean
}

export interface EdgeProjection {
  id: string
  source: string
  target: string
  edges: TeamGraphEdge[]
  kinds: EdgeKind[]
  forward: boolean
  reverse: boolean
  label: string
}

export const EDGE_KINDS: EdgeKind[] = ['control', 'data']
export const EDGE_COLORS: Record<EdgeKind, string> = { control: '#2563EB', data: '#D97706' }
export const EDGE_LABELS: Record<EdgeKind, string> = { control: 'Control', data: 'Data' }

const pair = (source: string, target: string): [string, string] => (
  source.localeCompare(target) <= 0 ? [source, target] : [target, source]
)

const edgeLabel = (edges: TeamGraphEdge[]): string => EDGE_KINDS.map((kind) => {
  const count = edges.filter((edge) => edge.edgeKind === kind).length
  return count ? `${kind === 'control' ? 'C' : 'D'}×${count}` : ''
}).filter(Boolean).join(' · ')

export function teamEdges(spec: Json): TeamGraphEdge[] {
  const nodeIds = new Set((spec.nodes || []).map((node: Json) => String(node.id)))
  const values: TeamGraphEdge[] = []
  for (const [index, relation] of (spec.relations || []).entries()) {
    const source = String(relation.from || '')
    const target = String(relation.to || '')
    if (!nodeIds.has(source) || !nodeIds.has(target)) continue
    if (relation.control) {
      values.push({
        id: `${relation.id}:control`, relationId: String(relation.id), source, target,
        sourcePort: String(relation.control.trigger || 'completed'),
        targetPort: String(relation.control.action || 'activate'), edgeKind: 'control',
        semanticType: 'control', operation: String(relation.control.action || 'activate'),
        edgeIndex: index, enabled: true, description: '', policy: relation.control,
      })
    }
    if (relation.data) {
      const transfers = relation.data.transfers || []
      values.push({
        id: `${relation.id}:data`, relationId: String(relation.id), source, target,
        sourcePort: transfers.map((item: Json) => item.source).join(', '),
        targetPort: transfers.map((item: Json) => item.target).join(', '), edgeKind: 'data',
        semanticType: transfers.map((item: Json) => item.target).join(', ') || 'data',
        operation: 'transfer', edgeIndex: index, enabled: true, description: '',
        policy: { transfers },
      })
    }
  }
  return values
}

const stateNodeId = (stateId: string): string => `shared-state::${stateId}`

export function sharedStateEdges(spec: Json): TeamGraphEdge[] {
  const nodeIds = new Set((spec.nodes || []).map((node: Json) => String(node.id)))
  const values: TeamGraphEdge[] = []
  for (const state of spec.shared_state || []) {
    const stateId = String(state.id)
    const visualId = stateNodeId(stateId)
    for (const writer of state.writers || []) {
      if (!nodeIds.has(String(writer))) continue
      values.push({
        id: `state:${stateId}:write:${writer}`, relationId: '', source: String(writer),
        target: visualId, sourcePort: 'output', targetPort: 'write', edgeKind: 'data',
        semanticType: 'state_write', operation: String(state.update || 'append'),
        edgeIndex: values.length, enabled: true, description: `write ${stateId}`,
        policy: { state_id: stateId, access: 'write' }, stateId, derived: true,
      })
    }
    for (const reader of state.readers || []) {
      if (!nodeIds.has(String(reader))) continue
      values.push({
        id: `state:${stateId}:read:${reader}`, relationId: '', source: visualId,
        target: String(reader), sourcePort: 'read', targetPort: 'input', edgeKind: 'data',
        semanticType: 'state_read', operation: 'read', edgeIndex: values.length,
        enabled: true, description: `read ${stateId}`,
        policy: { state_id: stateId, access: 'read' }, stateId, derived: true,
      })
    }
  }
  return values
}

export const sharedStateNodeId = stateNodeId

export function projectEdges(edges: TeamGraphEdge[]): EdgeProjection[] {
  const grouped = new Map<string, TeamGraphEdge[]>()
  for (const edge of edges) {
    const [source, target] = pair(edge.source, edge.target)
    const key = `relations:${encodeURIComponent(source)}::${encodeURIComponent(target)}`
    grouped.set(key, [...(grouped.get(key) || []), edge])
  }
  return [...grouped.entries()].map(([id, group]) => {
    const [source, target] = pair(group[0].source, group[0].target)
    return {
      id, source, target, edges: group,
      kinds: EDGE_KINDS.filter((kind) => group.some((edge) => edge.edgeKind === kind)),
      forward: group.some((edge) => edge.source === source && edge.target === target),
      reverse: group.some((edge) => edge.source === target && edge.target === source),
      label: edgeLabel(group),
    }
  })
}

export const edgeCounts = (edges: TeamGraphEdge[]): Record<EdgeKind, number> => ({
  control: edges.filter((edge) => edge.edgeKind === 'control').length,
  data: edges.filter((edge) => edge.edgeKind === 'data').length,
})

export const executableNodeIds = (spec?: Json): string[] => (spec?.nodes || [])
  .map((node: Json) => String(node.id))

export const controlSemanticsByNode = (spec?: Json): Map<string, string[]> => {
  const result = new Map<string, string[]>()
  for (const node of spec?.nodes || []) {
    const operations = (node.operations || []).map((item: Json) => String(item.type))
    if (operations.length) result.set(String(node.id), operations)
  }
  return result
}

export interface CoordinationSummary {
  topology: 'independent' | 'sequential' | 'centralized' | 'decentralized'
  label: string
  operationNames: string[]
  executableCount: number
  storeCount: number
  controlCount: number
  dataCount: number
  isDependencyChain: boolean
  hasHandoff: boolean
}

const isLinearControlGraph = (members: string[], edges: TeamGraphEdge[]): boolean => {
  if (members.length < 2 || edges.length !== members.length - 1) return false
  const memberSet = new Set(members)
  const outgoing = new Map(members.map((id): [string, string[]] => [id, []]))
  const indegree = new Map(members.map((id): [string, number] => [id, 0]))
  for (const edge of edges) {
    if (!memberSet.has(edge.source) || !memberSet.has(edge.target)) return false
    outgoing.set(edge.source, [...(outgoing.get(edge.source) || []), edge.target])
    indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1)
  }
  return [...outgoing.values()].every((targets) => targets.length <= 1)
    && [...indegree.values()].every((degree) => degree <= 1)
    && members.filter((id) => indegree.get(id) === 0).length === 1
}

export const memberNodeIds = (spec?: Json): string[] => {
  const centralIds = new Set(
    [...controlSemanticsByNode(spec).entries()]
      .filter(([, operations]) => operations.some((value) => [
        'select_next', 'plan', 'delegate', 'monitor_progress', 'detect_stall',
        'replan', 'aggregate',
      ].includes(value)))
      .map(([id]) => id),
  )
  return executableNodeIds(spec).filter((id) => !centralIds.has(id))
}

export const deriveCoordinationSummary = (spec?: Json): CoordinationSummary => {
  const allEdges = teamEdges(spec || {})
  const executable = executableNodeIds(spec)
  const workers = memberNodeIds(spec)
  const centralCount = executable.length - workers.length
  const control = allEdges.filter((edge) => edge.edgeKind === 'control')
  const workerControl = control.filter((edge) => workers.includes(edge.source) && workers.includes(edge.target))
  const hasHandoff = control.some((edge) => edge.operation === 'handoff')
  const isDependencyChain = isLinearControlGraph(workers, workerControl)
  const topology: CoordinationSummary['topology'] = centralCount
    ? 'centralized'
    : workers.length <= 1
      ? 'independent'
      : isDependencyChain
        ? 'sequential'
        : 'decentralized'
  const counts = edgeCounts(allEdges)
  const operations = [...controlSemanticsByNode(spec).values()].flat()
  const stateCount = (spec?.shared_state || []).length
  return {
    topology,
    label: `${executable.length} Nodes · ${stateCount} SharedState · C${counts.control} D${counts.data}${hasHandoff ? ' · handoff' : ''}`,
    operationNames: operations,
    executableCount: executable.length,
    storeCount: stateCount,
    controlCount: counts.control,
    dataCount: counts.data,
    isDependencyChain,
    hasHandoff,
  }
}

export const editableEdges = (projection?: EdgeProjection): TeamGraphEdge[] => projection?.edges || []
