import type { Bootstrap, Json, PlanResponse } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail || response.statusText)
  }
  return response.json()
}

export const api = {
  bootstrap: () => request<Bootstrap>('/api/bootstrap'),
  workspace: (name: string) => request<Partial<Bootstrap>>(`/api/workspaces/${encodeURIComponent(name)}`),
  refreshBenchmarks: () => request<Json[]>('/api/benchmarks'),
  instantiateModel: (id: string, payload: Json) =>
    request<Json>(`/api/model-instances/${encodeURIComponent(id)}/instantiate`, {
      method: 'POST', body: JSON.stringify(payload),
    }),
  scanModelInstances: (modelsRoot: string) =>
    request<Json>('/api/model-instances/scan', {
      method: 'POST', body: JSON.stringify({ models_root: modelsRoot }),
    }),
  deleteModelInstance: (id: string) =>
    request<Json>(`/api/model-instances/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  instantiateApi: (id: string, payload: Json) =>
    request<Json>(`/api/api-instances/${encodeURIComponent(id)}/instantiate`, {
      method: 'POST', body: JSON.stringify(payload),
    }),
  deleteApiInstance: (id: string) =>
    request<Json>(`/api/api-instances/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  instantiateBenchmark: (id: string, payload: Json) =>
    request<Json>(`/api/benchmark-instances/${encodeURIComponent(id)}/instantiate`, {
      method: 'POST', body: JSON.stringify(payload),
    }),
  scanBenchmarkInstances: (rawRoot: string, preparedRoot: string) =>
    request<Json>('/api/benchmark-instances/scan', {
      method: 'POST', body: JSON.stringify({ raw_root: rawRoot, prepared_root: preparedRoot }),
    }),
  checkBenchmarkInstance: (id: string) =>
    request<Json>(`/api/benchmark-instances/${encodeURIComponent(id)}/check`, { method: 'POST' }),
  deleteBenchmarkInstance: (id: string) =>
    request<Json>(`/api/benchmark-instances/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  refreshRuns: () => request<Json[]>('/api/runs'),
  runPage: (query = '', offset = 0, limit = 100) => {
    const params = new URLSearchParams({ query, offset: String(offset), limit: String(limit) })
    return request<Json>(`/api/run-page?${params.toString()}`)
  },
  resources: (rawRoot: string, preparedRoot: string, modelsRoot: string) =>
    request<Json>(`/api/resources/catalog?raw_root=${encodeURIComponent(rawRoot)}&prepared_root=${encodeURIComponent(preparedRoot)}&models_root=${encodeURIComponent(modelsRoot)}`),
  checkEnvironment: (payload: Json) =>
    request<Json>('/api/environments/check', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deployments: (probe = false) => request<Json>(`/api/deployments?probe=${probe ? 'true' : 'false'}`),
  pricing: () => request<Json>('/api/pricing'),
  savePricingSpec: (id: string, payload: Json) =>
    request<Json>(`/api/pricing-specs/${encodeURIComponent(id)}`, {
      method: 'PUT', body: JSON.stringify(payload),
    }),
  deletePricingSpec: (id: string) =>
    request<Json>(`/api/pricing-specs/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  instantiatePricing: (specId: string, payload: Json) =>
    request<Json>(`/api/pricing-specs/${encodeURIComponent(specId)}/instances`, {
      method: 'POST', body: JSON.stringify(payload),
    }),
  deletePricingInstance: (id: string) =>
    request<Json>(`/api/pricing-instances/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  teamSpecs: () => request<Json>('/api/team-specs'),
  teamBindingReport: (runtimeFramework: string, teamSpec: Json) =>
    request<Json>('/api/team-binding-report', {
      method: 'POST',
      body: JSON.stringify({ runtime_framework: runtimeFramework, team_spec: teamSpec }),
    }),
  teamInstances: (probe = false) => request<Json>(`/api/team-instances?probe=${probe ? 'true' : 'false'}`),
  saveTeamSpec: (id: string, payload: Json) =>
    request<Json>(`/api/team-specs/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteTeamSpec: (id: string) =>
    request<Json>(`/api/team-specs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  saveTeamInstance: (id: string, payload: Json) =>
    request<Json>(`/api/team-instances/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteTeamInstance: (id: string) =>
    request<Json>(`/api/team-instances/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  saveDeployment: (id: string, payload: Json) =>
    request<Json>(`/api/deployments/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteDeployment: (id: string) =>
    request<Json>(`/api/deployments/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  deployDeployment: (id: string, payload: Json) =>
    request<Json>(`/api/deployments/${encodeURIComponent(id)}/deploy`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteDeploymentInstance: (id: string) =>
    request<Json>(`/api/deployment-instances/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  experimentSpecs: () => request<Json[]>('/api/experiment-specs'),
  experimentDashboard: (includeSpecs = false) =>
    request<Json>(`/api/experiment-dashboard?include_specs=${includeSpecs ? 'true' : 'false'}`),
  experimentInstances: () => request<Json[]>('/api/experiment-instances'),
  experimentOrphans: () => request<Json[]>('/api/experiment-orphans'),
  experimentInstanceProgress: (id: string, tail = 40) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/progress?tail=${tail}`),
  saveExperimentSpec: (id: string, project: Json) =>
    request<Json>(`/api/experiment-specs/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(project),
    }),
  deleteExperimentSpec: (id: string) =>
    request<Json>(`/api/experiment-specs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  createExperimentInstance: (payload: Json) =>
    request<Json>('/api/experiment-instances', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateExperimentInstance: (id: string, payload: Json) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  planExperimentInstance: (id: string) =>
    request<PlanResponse>(`/api/experiment-instances/${encodeURIComponent(id)}/plan`, {
      method: 'POST',
    }),
  launchExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/launch`, {
      method: 'POST',
    }),
  enqueueExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/enqueue`, {
      method: 'POST',
    }),
  dequeueExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/dequeue`, {
      method: 'POST',
    }),
  stopExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/stop`, {
      method: 'POST',
    }),
  resumeExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/resume`, {
      method: 'POST',
    }),
  drainExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}/drain`, {
      method: 'POST',
    }),
  stopExperimentOrphan: (id: string) =>
    request<Json>(`/api/experiment-orphans/${encodeURIComponent(id)}/stop`, {
      method: 'POST',
    }),
  recoverExperimentOrphan: (id: string) =>
    request<Json>(`/api/experiment-orphans/${encodeURIComponent(id)}/recover`, {
      method: 'POST',
    }),
  deleteExperimentInstance: (id: string) =>
    request<Json>(`/api/experiment-instances/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  experimentQueue: () => request<Json>('/api/experiment-queue'),
  startExperimentQueue: (maxParallelInstances = 16, maxRunningTrials = 64) =>
    request<Json>('/api/experiment-queue/start', {
      method: 'POST',
      body: JSON.stringify({
        max_parallel_instances: maxParallelInstances,
        max_running_trials: maxRunningTrials,
      }),
    }),
  stopExperimentQueueAfterCurrent: () =>
    request<Json>('/api/experiment-queue/stop-after-current', { method: 'POST' }),
  prepareDataset: (payload: Json) =>
    request<Json>('/api/datasets/prepare', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  launchProgress: (launchId: string) => request<Json>(`/api/launches/${launchId}/progress`),
  runEvents: (runId: string, startLine = 0, limit = 1000) =>
    request<Json>(`/api/runs/${runId}/run-events?start_line=${startLine}&limit=${limit}`),
  executionTrace: (runId: string, startNode = 0, limit = 1000, caseId = '', trialIndex?: number, filters: string[] = []) => {
    const query = new URLSearchParams({ start_node: String(startNode), limit: String(limit) })
    if (caseId) query.set('case_id', caseId)
    if (trialIndex !== undefined) query.set('trial_index', String(trialIndex))
    if (filters.length) query.set('filters', filters.join(','))
    return request<Json>(`/api/runs/${runId}/execution-trace?${query.toString()}`)
  },
  results: (runId: string, startTrial = 0, limit = 100) =>
    request<Json>(`/api/runs/${runId}/results?start_trial=${startTrial}&limit=${limit}`),
  evidence: (runId: string, startLine = 0, limit = 1000) =>
    request<Json>(`/api/runs/${runId}/evidence?start_line=${startLine}&limit=${limit}`),
  metricObservations: (runId: string, startLine = 0, limit = 1000) =>
    request<Json>(`/api/runs/${runId}/metric-observations?start_line=${startLine}&limit=${limit}`),
  metricTrials: (runId: string, startTrial = 0, limit = 100) =>
    request<Json>(`/api/runs/${runId}/metric-trials?start_trial=${startTrial}&limit=${limit}`),
  evidenceCoverage: (runId: string) =>
    request<Json>(`/api/runs/${runId}/evidence-coverage`),
  metricContracts: () => request<Json>('/api/metric-contracts'),
  evaluationProfiles: () => request<Json>('/api/evaluation-profiles'),
  metricApplicability: (runId: string) =>
    request<Json>(`/api/runs/${runId}/metric-applicability`),
  metricEvaluation: (runId: string, profileId = 'core') =>
    request<Json>(`/api/runs/${runId}/metric-evaluation?profile_id=${encodeURIComponent(profileId)}`),
  analyzeRun: (runId: string, profileId = 'core') =>
    request<Json>(`/api/runs/${runId}/analyze?profile_id=${encodeURIComponent(profileId)}`, {
      method: 'POST',
    }),
}
