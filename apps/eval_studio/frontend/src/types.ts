import type { Edge, Node } from '@xyflow/react'

export type Json = Record<string, any>

export interface AgentData extends Record<string, unknown> {
  label: string
  role: string
  agent_type: string
  system_prompt: string
  description?: string
  tools: string[]
  model_context?: Json
}

export type AgentNode = Node<AgentData>
export type TeamEdge = Edge

export interface InferenceDeploymentBinding {
  slot_id: string
  deployment_instance_id: string
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
