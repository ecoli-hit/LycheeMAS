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
  startExperimentQueue: (maxParallelInstances = 8) =>
    request<Json>('/api/experiment-queue/start', {
      method: 'POST',
      body: JSON.stringify({ max_parallel_instances: maxParallelInstances }),
    }),
  stopExperimentQueueAfterCurrent: () =>
    request<Json>('/api/experiment-queue/stop-after-current', { method: 'POST' }),
  prepareDataset: (payload: Json) =>
    request<Json>('/api/datasets/prepare', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  launchProgress: (launchId: string) => request<Json>(`/api/launches/${launchId}/progress`),
  events: (runId: string, startLine = 0, limit = 1000) =>
    request<Json>(`/api/runs/${runId}/events?start_line=${startLine}&limit=${limit}`),
  groupChat: (runId: string, startLine = 0, limit = 1000) =>
    request<Json>(`/api/runs/${runId}/group-chat?start_line=${startLine}&limit=${limit}`),
  evidence: (runId: string, startLine = 0, limit = 1000) =>
    request<Json>(`/api/runs/${runId}/evidence?start_line=${startLine}&limit=${limit}`),
  evidenceCoverage: (runId: string) =>
    request<Json>(`/api/runs/${runId}/evidence-coverage`),
}
