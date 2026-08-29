import type { Edge, Node } from '@xyflow/react'

export type Json = Record<string, any>

export interface AgentData extends Record<string, unknown> {
  label: string
  role: string
  visual_kind: string
  execution_kind?: string
  node_scope?: string
  system_prompt: string
  description?: string
  tools: string[]
  model_context?: Json
  ports?: Array<{
    id: string
    type: 'source' | 'target'
    side: 'left' | 'right' | 'top' | 'bottom'
    offset: number
    edge_type: string
    color: string
  }>
}

export type AgentNode = Node<AgentData>
export type TeamEdge = Edge

export interface NodeResourceBinding {
  node_id: string
  requirement: string
  resource_instance_type: string
  resource_instance_id: string
  generation_overrides?: Json
}

export interface Bootstrap {
  paths: Json
  teams: Json
  team_instances: Json
  benchmarks: Json[]
  benchmark_registry: Json
  model_registry: Json
  api_registry: Json
  deployments: Json
  pricing_registry: Json
  metric_registry: Json
  evaluation_profiles: Json
  runs: Json[]
  tmux_sessions: Json[]
  environments: Json
  installation_profiles: Json[]
  experiments: Json
  default_experiment_spec: Json
}

export interface PlanResponse {
  launch_id: string
  launch_dir: string
  run_dir: string
  project: Json
  files: Record<string, string>
  commands: Record<string, string>
}
