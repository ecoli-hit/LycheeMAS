import {
  Plus,
  RefreshCw,
  Save,
  ServerCog,
  Trash2,
} from 'lucide-react'
import { useState } from 'react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import ResourcePressure, { type PressureEntry } from './ResourcePressure'
import SpecSelector from './SpecSelector'
import TeamManagementV6 from './TeamManagement'
import type { Bootstrap, Json } from './types'

interface Props {
  bootstrap: Bootstrap
  environment: Json
  notify: (message: string) => void
  workspace: 'team' | 'deployment'
}

const status = (value: Json) => String(value.observed_status || value.status || 'unknown')
const isDeploymentReady = (value: Json) => (
  ['running', 'ready_on_run'].includes(status(value))
)

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
    schema_version: 3,
    id: `deployment-${Date.now().toString(36)}`,
    kind: 'hf',
    model_spec_id: modelSpec?.id || '',
    source_spec: { type: 'model', id: modelSpec?.id || '' },
    actual_pricing_spec_id: actualPricingSpec?.id || '',
    api_equivalent_pricing_spec_id: equivalentPricingSpec?.id || null,
    python: bootstrap.environments?.default?.python || '',
    device: 'cuda:0',
    cuda_visible_devices: '0',
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

const resolvedModelId = (modelSpec?: Json, apiSpec?: Json): string => {
  if (!modelSpec) return ''
  if (!apiSpec) return String(modelSpec.name || '')
  return String(modelSpec.provider_model_ids?.[apiSpec.provider_key] || '')
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
  const selectedModelSpec = modelSpecs.find((item: Json) => item.id === draft.model_spec_id)
  const compatibleModelSpecs = draft.source_spec?.type === 'api'
    ? modelSpecs.filter((item: Json) => (selectedSourceSpec?.allowed_model_spec_ids || []).includes(item.id))
    : modelSpecs.filter((item: Json) => item.id === draft.source_spec?.id)
  const pricingModelId = resolvedModelId(
    selectedModelSpec,
    draft.source_spec?.type === 'api' ? selectedSourceSpec : undefined,
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
  const deploymentPressureEntries: PressureEntry[] = Object.entries(
    catalog.pressure?.deployments || {},
  ).map(([deploymentId, pressure]: [string, any]) => {
    const instance = (catalog.instances || []).find((item: Json) => item.id === deploymentId)
    return {
      id: deploymentId,
      kind: instance?.kind === 'api' ? 'api' : 'vllm',
      pressure,
      title: `${instance?.kind === 'api' ? 'API' : 'vLLM'} · ${deploymentId}`,
    }
  })
  if (catalog.pressure?.gpu) {
    deploymentPressureEntries.push({
      id: 'deployment-gpu',
      kind: 'gpu',
      pressure: catalog.pressure.gpu,
      title: 'Deployment 使用的 GPU',
    })
  }

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
    setExtraBodyText(spec.extra_body ? JSON.stringify(spec.extra_body, null, 2) : '')
    setLibraryPathsText((spec.library_paths || []).join('\n'))
    setSourceInstanceId('')
    setInstanceId(`instance-${spec.id}`)
    setApiKey('')
    setActualPricingInstanceId(
      existing?.actual_pricing_instance_id
        || pricingInstances.find((item) => item.pricing_spec_id === spec.actual_pricing_spec_id
          && pricingAppliesToModel(item, resolvedModelId(
            modelSpecs.find((model: Json) => model.id === spec.model_spec_id),
            spec.source_spec?.type === 'api'
              ? apiSpecs.find((source: Json) => source.id === spec.source_spec?.id)
              : undefined,
          ))
          && (actualPricingBasis(spec) !== 'token_usage'
            || pricingAppliesToSource(item, String(spec.source_spec?.id || ''))))?.id
        || '',
    )
    setEquivalentPricingInstanceId(
      existing?.api_equivalent_pricing_instance_id
        || pricingInstances.find((item) => item.pricing_spec_id === spec.api_equivalent_pricing_spec_id
          && pricingAppliesToModel(item, resolvedModelId(
            modelSpecs.find((model: Json) => model.id === spec.model_spec_id),
            spec.source_spec?.type === 'api'
              ? apiSpecs.find((source: Json) => source.id === spec.source_spec?.id)
              : undefined,
          )))?.id
        || '',
    )
  }

  const deploymentPayload = () => {
    let extraBody: Json | undefined
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
      ...(extraBody ? { extra_body: extraBody } : { extra_body: undefined }),
      ...(libraryPaths.length ? { library_paths: libraryPaths } : { library_paths: undefined }),
      ...(Object.keys(reasoningConfig).length ? { reasoning_config: reasoningConfig } : { reasoning_config: undefined }),
      ...(!allowsApiEquivalent ? { api_equivalent_pricing_spec_id: null } : {}),
      schema_version: 3,
    }
  }

  const chooseKind = (kind: string) => {
    const managed = kind !== 'api'
    const sourceType = kind === 'api' ? 'api' : 'model'
    const candidates = sourceType === 'api' ? apiSpecs : modelSpecs
    const selectedSource = candidates[0]
    const allowedModelIds = sourceType === 'api'
      ? selectedSource?.allowed_model_spec_ids || []
      : [selectedSource?.id]
    const selectedModel = modelSpecs.find((item: Json) => allowedModelIds.includes(item.id))
      || modelSpecs[0]
    const modelId = resolvedModelId(
      selectedModel,
      sourceType === 'api' ? selectedSource : undefined,
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
    for (const field of ['tensor_parallel_size', 'data_parallel_size', 'max_num_seqs', 'admission_control', 'per_request_metrics_mode', 'prompt_tokens_details_mode', 'reasoning_parser', 'reasoning_config', 'tool_call_parser', 'enable_auto_tool_choice', 'enable_prefix_caching', 'use_flashinfer_sampler', 'enforce_eager']) {
      delete next[field]
    }
    setDraft({
      ...next,
      kind,
      managed: kind !== 'api',
      ...(kind === 'vllm' ? { shared: false, host: '127.0.0.1', port: 8000, tensor_parallel_size: 1, data_parallel_size: 1, max_num_seqs: 16, admission_control: { mode: 'auto', minimum: 4, initial: 16, maximum: 16, increase_step: 2, decrease_factor: 0.75, healthy_polls_required: 3 }, per_request_metrics_mode: 'auto', prompt_tokens_details_mode: 'auto' } : {}),
      model_spec_id: selectedModel?.id || '',
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
    const allowedModelIds = draft.source_spec?.type === 'api'
      ? sourceSpec?.allowed_model_spec_ids || []
      : [sourceSpec?.id]
    const nextModel = modelSpecs.find((item: Json) => item.id === draft.model_spec_id
      && allowedModelIds.includes(item.id))
      || modelSpecs.find((item: Json) => allowedModelIds.includes(item.id))
    const modelId = resolvedModelId(
      nextModel,
      draft.source_spec?.type === 'api' ? sourceSpec : undefined,
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
      model_spec_id: nextModel?.id || '',
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
    const allowedModelIds = sourceType === 'api'
      ? selectedSource?.allowed_model_spec_ids || []
      : [selectedSource?.id]
    const selectedModel = modelSpecs.find((item: Json) => allowedModelIds.includes(item.id))
      || modelSpecs[0]
    const modelId = resolvedModelId(
      selectedModel,
      sourceType === 'api' ? selectedSource : undefined,
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
      model_spec_id: selectedModel?.id || '',
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

  const chooseModelSpec = (modelSpecId: string) => {
    const modelSpec = compatibleModelSpecs.find((item: Json) => item.id === modelSpecId)
    const modelId = resolvedModelId(
      modelSpec,
      draft.source_spec?.type === 'api' ? selectedSourceSpec : undefined,
    )
    const actualSpec = preferredActualPricingSpec(
      pricingCatalog.specs || [],
      actualPricingBasis(draft),
      modelId,
      String(draft.source_spec?.id || ''),
    )
    const equivalentSpec = allowsApiEquivalent ? preferredActualPricingSpec(
      pricingCatalog.specs || [],
      'token_usage',
      modelId,
    ) : null
    setDraft({
      ...draft,
      model_spec_id: modelSpecId,
      actual_pricing_spec_id: actualSpec?.id || '',
      api_equivalent_pricing_spec_id: equivalentSpec?.id || null,
    })
    setActualPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === actualSpec?.id
        && pricingAppliesToModel(item, modelId)
        && (actualSpec?.basis !== 'token_usage'
          || pricingAppliesToSource(item, String(draft.source_spec?.id || ''))))?.id || '',
    )
    setEquivalentPricingInstanceId(
      pricingInstances.find((item) => item.pricing_spec_id === equivalentSpec?.id
        && pricingAppliesToModel(item, modelId))?.id || '',
    )
  }

  const saveSpec = async () => {
    setBusy(true)
    try {
      const saved = await api.saveDeployment(draft.id, deploymentPayload())
      await refresh()
      setDraft(saved)
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

  const patchAdmissionControl = (field: string, raw: string | number) => setDraft({
    ...draft,
    admission_control: {
      ...(draft.admission_control || {}),
      [field]: field === 'mode' ? raw : Number(raw),
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
          <label><span>{draft.source_spec?.type === 'api' ? 'APISpec' : 'ModelSpec source'}</span><select value={draft.source_spec?.id || ''} onChange={(event) => chooseSourceSpec(event.target.value)}>{sourceSpecs.map((item: Json) => <option key={item.id} value={item.id}>{item.name || item.id} · {item.id}</option>)}</select></label>
          <label><span>ModelSpec</span><select disabled={draft.source_spec?.type !== 'api'} value={draft.model_spec_id || ''} onChange={(event) => chooseModelSpec(event.target.value)}>{compatibleModelSpecs.map((item: Json) => <option key={item.id} value={item.id}>{item.name || item.id} · {item.id}</option>)}</select><small>{draft.source_spec?.type === 'api' ? 'API 访问通道与模型产品分别选择。' : '本地部署的 ModelSpec 与模型资源来源一致。'}</small></label>
          <label><span>实际成本 PricingSpec</span><select value={draft.actual_pricing_spec_id || ''} onChange={(event) => { const id = event.target.value; setDraft({ ...draft, actual_pricing_spec_id: id }); setActualPricingInstanceId(pricingInstances.find((item) => item.pricing_spec_id === id)?.id || '') }}>{actualPricingSpecs.map((item: Json) => <option key={item.id} value={item.id}>{item.id} · {item.billing_mode}</option>)}</select><small>{allowsApiEquivalent ? '本地算力按运行墙钟时间 × 实际分配 GPU 数计费。' : '外部服务按 provider 返回的 token usage 计费。'}</small></label>
          {allowsApiEquivalent && <label><span>API 等价成本 PricingSpec</span><select value={draft.api_equivalent_pricing_spec_id || ''} onChange={(event) => { const id = event.target.value; setDraft({ ...draft, api_equivalent_pricing_spec_id: id || null }); setEquivalentPricingInstanceId(pricingInstances.find((item) => item.pricing_spec_id === id)?.id || '') }}><option value="">不计算 API 等价成本</option>{(pricingCatalog.specs || []).filter((item: Json) => item.basis === 'token_usage').map((item: Json) => <option key={item.id} value={item.id}>{item.id}</option>)}</select><small>可选：用同一次运行的 token usage 估算对标云 API 费用。</small></label>}
          {(draft.kind === 'hf' || (draft.kind === 'vllm' && draft.managed !== false)) && <><label className="wide-field"><span>Python executable</span><input value={draft.python || environment.python || ''} onChange={(event) => setDraft({ ...draft, python: event.target.value })} /></label><label><span>{draft.kind === 'hf' ? 'Physical CUDA device' : 'CUDA visible devices'}</span><input value={draft.cuda_visible_devices || '0'} onChange={(event) => setDraft({ ...draft, cuda_visible_devices: event.target.value })} /><small>{draft.kind === 'hf' ? '每个 Local HF DeploymentInstance 分配一张物理 GPU；实例化团队后自动映射为进程内 cuda:N。' : '按 TP × DP 提供本地 vLLM 服务所需物理 GPU。'}</small></label></>}
          {draft.kind === 'vllm' && draft.managed !== false && <><label><span>Host</span><input value={draft.host || '127.0.0.1'} onChange={(event) => setDraft({ ...draft, host: event.target.value })} /></label><label><span>Port</span><input type="number" min="1" max="65535" value={draft.port || 8000} onChange={(event) => patchNumber('port', event.target.value)} /></label><label><span>Tensor parallel</span><input type="number" min="1" value={draft.tensor_parallel_size || 1} onChange={(event) => patchNumber('tensor_parallel_size', event.target.value)} /><small>每个模型副本共同使用的 GPU 数。</small></label><label><span>Data parallel</span><input type="number" min="1" value={draft.data_parallel_size || 1} onChange={(event) => patchNumber('data_parallel_size', event.target.value)} /><small>完整模型副本数；至少需要 TP × DP 张可见 GPU。</small></label><label><span>Max model length</span><input type="number" min="1" value={draft.max_model_len || 32768} onChange={(event) => patchNumber('max_model_len', event.target.value)} /></label><label><span>GPU memory utilization</span><input type="number" min="0.01" max="1" step="0.01" value={draft.gpu_memory_utilization || 0.9} onChange={(event) => patchNumber('gpu_memory_utilization', event.target.value)} /></label><label><span>Max concurrent sequences / replica</span><input type="number" min="1" value={draft.max_num_seqs || 1} onChange={(event) => patchNumber('max_num_seqs', event.target.value)} /><small>按每个 DP replica 生效，总上限约为该值 × DP。</small></label><label><span>Data type</span><select value={draft.dtype || 'auto'} onChange={(event) => setDraft({ ...draft, dtype: event.target.value })}><option value="auto">auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option></select></label><label><span>Per-request metrics</span><select value={draft.per_request_metrics_mode || 'auto'} onChange={(event) => setDraft({ ...draft, per_request_metrics_mode: event.target.value })}><option value="auto">Auto · CLI 支持时开启</option><option value="enabled">Enabled · 不支持则拒绝部署</option><option value="disabled">Disabled</option></select><small>控制 vLLM 是否在每次 OpenAI-compatible 响应中附带 queue、TTFT、generation 和 ITL 指标。</small></label><label><span>Prompt token details</span><select value={draft.prompt_tokens_details_mode || 'auto'} onChange={(event) => setDraft({ ...draft, prompt_tokens_details_mode: event.target.value })}><option value="auto">Auto · CLI 支持时开启</option><option value="enabled">Enabled · 不支持则拒绝部署</option><option value="disabled">Disabled</option></select><small>输出每次请求的 <code>usage.prompt_tokens_details.cached_tokens</code>，用于精确记录 Prefix Cache 命中 Token。</small></label></>}
          {(draft.kind === 'api' || draft.kind === 'vllm') && <>
            <label><span>Request timeout mode</span><select value={draft.request_timeout?.mode || 'adaptive'} onChange={(event) => patchRequestTimeout('mode', event.target.value)}><option value="adaptive">Adaptive</option><option value="fixed">Fixed</option></select></label>
            <label><span>Minimum timeout (s)</span><input type="number" min="1" value={draft.request_timeout?.minimum_s ?? 120} onChange={(event) => patchRequestTimeout('minimum_s', event.target.value)} /></label>
            <label><span>Maximum timeout (s)</span><input type="number" min="1" value={draft.request_timeout?.maximum_s ?? 1800} onChange={(event) => patchRequestTimeout('maximum_s', event.target.value)} /></label>
            <label><span>Base timeout (s)</span><input type="number" min="0" value={draft.request_timeout?.base_s ?? 30} onChange={(event) => patchRequestTimeout('base_s', event.target.value)} /></label>
            <label><span>Initial generation speed (tok/s)</span><input type="number" min="0.1" step="0.1" value={draft.request_timeout?.initial_generation_tokens_per_second ?? 20} onChange={(event) => patchRequestTimeout('initial_generation_tokens_per_second', event.target.value)} /></label>
            <label><span>Planning speed upper bound (tok/s)</span><input type="number" min="0.1" step="0.1" value={draft.request_timeout?.observed_tokens_per_second_ceiling ?? 40} onChange={(event) => patchRequestTimeout('observed_tokens_per_second_ceiling', event.target.value)} /><small>规划取冷启动基线、历史 EWMA 与该上界中的最慢值；快样本不会缩短保守超时。</small></label>
            <label><span>Timeout safety factor</span><input type="number" min="1" step="0.1" value={draft.request_timeout?.safety_factor ?? 1.5} onChange={(event) => patchRequestTimeout('safety_factor', event.target.value)} /></label>
            <label><span>Speed EWMA alpha</span><input type="number" min="0" max="1" step="0.05" value={draft.request_timeout?.ewma_alpha ?? 0.25} onChange={(event) => patchRequestTimeout('ewma_alpha', event.target.value)} /></label>
            <label><span>SDK automatic retries</span><input type="number" min="0" value={draft.max_retries ?? 0} onChange={(event) => patchNumber('max_retries', event.target.value)} /></label>
            <label><span>Client max concurrency</span><input type="number" min="0" value={draft.request_limits?.max_concurrency ?? 0} onChange={(event) => patchRequestLimits('max_concurrency', event.target.value)} /><small>0 表示不在客户端额外限流；vLLM 服务仍按 max sequences 调度。</small></label>
            <label><span>Minimum request interval (s)</span><input type="number" min="0" step="0.1" value={draft.request_limits?.min_interval_s ?? 0} onChange={(event) => patchRequestLimits('min_interval_s', event.target.value)} /></label>
            <label><span>Request limit scope</span><select value={draft.request_limits?.scope || 'process'} onChange={(event) => patchRequestLimits('scope', event.target.value)}><option value="process">process</option><option value="host">host</option></select></label>
            {draft.kind === 'vllm' && <><label><span>运行池准入模式</span><select value={draft.admission_control?.mode || 'auto'} onChange={(event) => patchAdmissionControl('mode', event.target.value)}><option value="auto">Auto · 多信号 AIMD</option><option value="fixed">Fixed · 固定容量</option></select><small>不重启 vLLM；只控制共享运行池实际放入多少个 Case。</small></label><label><span>初始准入容量</span><input type="number" min="1" value={draft.admission_control?.initial ?? draft.request_limits?.max_concurrency ?? 1} onChange={(event) => patchAdmissionControl('initial', event.target.value)} /></label><label><span>最小准入容量</span><input type="number" min="1" value={draft.admission_control?.minimum ?? 1} onChange={(event) => patchAdmissionControl('minimum', event.target.value)} /></label><label><span>最大准入容量</span><input type="number" min="1" value={draft.admission_control?.maximum ?? draft.request_limits?.max_concurrency ?? 1} onChange={(event) => patchAdmissionControl('maximum', event.target.value)} /><small>不能超过 Client max concurrency。</small></label><label><span>健康窗口扩容步长</span><input type="number" min="1" value={draft.admission_control?.increase_step ?? 2} onChange={(event) => patchAdmissionControl('increase_step', event.target.value)} /></label><label><span>过载收缩系数</span><input type="number" min="0.1" max="0.95" step="0.05" value={draft.admission_control?.decrease_factor ?? 0.75} onChange={(event) => patchAdmissionControl('decrease_factor', event.target.value)} /></label><label><span>连续健康采样数</span><input type="number" min="1" value={draft.admission_control?.healthy_polls_required ?? 3} onChange={(event) => patchAdmissionControl('healthy_polls_required', event.target.value)} /></label></>}
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
          <div className="wide-field reference-notice"><strong>模型能力来自 ModelSpec</strong><span>{selectedModelSpec?.name || '未选择模型'} · thinking protocol: {selectedModelSpec?.thinking_protocol || 'none'} · {Object.keys(selectedModelSpec?.capabilities || {}).join(', ') || '未声明能力'}</span></div>
        </div>
        <div className="team-contract-note"><strong>Spec 边界</strong><span>这里保存后端启动与请求策略，并分别引用 ModelSpec 和资源访问 Spec。模型能力属于 ModelSpec；模型路径、API endpoint 和凭据引用来自实例化时选择的资源 Instance。</span><DetailsDisclosure value={draft} label="展开 DeploymentSpec" /></div>
      </section>

      <aside className="discovered-deployments deployment-instance-pane management-instance-pane">
        <section className="management-instance-builder">
          <div className="section-toolbar"><div><h1>DeploymentInstance</h1><span>为当前 DeploymentSpec 选择一个匹配的资源 Instance</span></div><button className="icon-button" title="刷新并执行健康检查" onClick={() => refresh(true)}><RefreshCw size={15} /></button></div>
          <label><span>Source Instance</span><select value={sourceInstanceId} onChange={(event) => setSourceInstanceId(event.target.value)}><option value="">请选择 {draft.source_spec?.type === 'api' ? 'APIInstance' : 'ModelInstance'}</option>{sourceInstances.map((item: Json) => <option key={item.id} value={item.id}>{item.id} · {draft.source_spec?.type === 'api' ? item.base_url : item.path}</option>)}</select></label>
          <label><span>DeploymentInstance ID</span><input value={instanceId} onChange={(event) => setInstanceId(event.target.value)} /></label>
          <label><span>实际成本 PricingInstance</span><select value={actualPricingInstanceId} onChange={(event) => setActualPricingInstanceId(event.target.value)}><option value="">请选择价格实例</option>{actualPricingCandidates.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.currency} · {item.status || 'registered'}</option>)}</select></label>
          {allowsApiEquivalent && draft.api_equivalent_pricing_spec_id && <label><span>API 等价成本 PricingInstance</span><select value={equivalentPricingInstanceId} onChange={(event) => setEquivalentPricingInstanceId(event.target.value)}><option value="">请选择同模型 API 价格实例</option>{equivalentPricingCandidates.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.currency} · {item.status || 'registered'}</option>)}</select></label>}
          {draft.source_spec?.type === 'api' && <label><span>API key（仅当前 Studio 进程，可选）</span><input type="password" autoComplete="new-password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="也可使用 APIInstance 登记的环境变量" /></label>}
          <div className="reference-notice"><strong>{draft.id}</strong><span>实例化时解析 <code>{draft.source_spec?.type}:{draft.source_spec?.id}</code> 对应的具体资源实例，并执行一次最小模型调用。只有健康检查结果属于 Instance。</span></div>
          <button className="primary" disabled={busy || !sourceInstanceId || !instanceId || !actualPricingInstanceId || Boolean(allowsApiEquivalent && draft.api_equivalent_pricing_spec_id && !equivalentPricingInstanceId)} onClick={instantiate}><ServerCog size={15} />实例化 Deployment</button>
        </section>
        <section className="management-instance-registry">
          <div className="panel-heading"><h2>已注册 DeploymentInstance</h2><span>{(catalog.instances || []).length}</span></div>
          <ResourcePressure
            title="推理后端压力"
            subtitle="Deployment monitor 直接采集服务与 GPU 事实；这里不包含实验优先级或 Scheduler 准入决策"
            entries={deploymentPressureEntries}
            emptyText="尚无活动 Deployment 的实时压力样本。"
          />
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
    ? <TeamManagementV6 bootstrap={props.bootstrap} notify={props.notify} />
    : <DeploymentManagement bootstrap={props.bootstrap} environment={props.environment} notify={props.notify} />
}
