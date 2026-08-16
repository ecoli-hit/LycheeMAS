import { Check, Plus, Save, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import SpecSelector from './SpecSelector'
import type { Bootstrap, Json } from './types'

interface Props {
  bootstrap: Bootstrap
  notify: (message: string, tone?: 'success' | 'error') => void
  onRefresh: () => Promise<void>
}

const gpuPricing = (billingMode: string): Json => ({
  basis: 'allocated_gpu_time',
  billing_mode: billingMode,
  rate_card_mode: 'flat',
  rate_unit: 'per_gpu_hour',
  supported_deployment_kinds: ['hf', 'vllm'],
  required_rates: ['gpu_hour'],
  optional_rates: [],
  currency: 'CNY',
  rates: { gpu_hour: null },
  rate_tiers: [],
  metadata: {},
  ...(billingMode === 'monthly_node_amortized'
    ? { amortization_policy: 'allocated_gpu_share', amortization_hours_per_month: 730 }
    : {}),
})

const pricingContract = (basis: string): Json => basis === 'allocated_gpu_time'
  ? gpuPricing('internal_gpu_hour')
  : {
      basis: 'token_usage',
      billing_mode: 'provider_token_usage',
      rate_card_mode: 'flat_or_input_token_tiers',
      rate_unit: 'per_million_tokens',
      supported_deployment_kinds: ['api', 'vllm'],
      required_rates: ['input', 'output'],
      optional_rates: ['cached_input', 'reasoning_output', 'thinking_output'],
      currency: 'CNY',
      rates: {
        input: null,
        output: null,
        cached_input: null,
        reasoning_output: null,
        thinking_output: null,
      },
      rate_tiers: [],
      metadata: { model_ids: [], source_spec_ids: [] },
    }

const newSpec = (): Json => ({
  schema_version: 2,
  id: `pricing-${Date.now().toString(36)}`,
  ...pricingContract('token_usage'),
  notes: '',
})

const newInstance = (spec: Json): Json => ({
  id: `price-${Date.now().toString(36)}`,
  pricing_spec_id: spec.id || '',
})

const listText = (value: unknown): string => Array.isArray(value) ? value.join(', ') : ''
const parseList = (value: string): string[] => value.split(',').map((item) => item.trim()).filter(Boolean)
const numberOrNull = (value: string): number | null => value === '' ? null : Number(value)

export default function CostView({ bootstrap, notify, onRefresh }: Props) {
  const initialCatalog = bootstrap.pricing_registry || { specs: [], instances: [] }
  const initialSpec = initialCatalog.specs?.[0] || newSpec()
  const [catalog, setCatalog] = useState<Json>(initialCatalog)
  const [specDraft, setSpecDraft] = useState<Json>(structuredClone(initialSpec))
  const [instanceSpecId, setInstanceSpecId] = useState<string>(initialSpec.id || '')
  const [instanceDraft, setInstanceDraft] = useState<Json>(newInstance(initialSpec))
  const [busy, setBusy] = useState(false)

  const specs: Json[] = catalog.specs || []
  const instances: Json[] = catalog.instances || []
  const selectedInstanceSpec = specs.find((item) => item.id === instanceSpecId) || initialSpec
  const isGpu = specDraft.basis === 'allocated_gpu_time'
  const billingMode = specDraft.billing_mode || 'internal_gpu_hour'
  const rateKeys = [...(specDraft.required_rates || []), ...(specDraft.optional_rates || [])]
  const tiered = Boolean(specDraft.rate_tiers?.length)

  const refresh = async () => {
    const value = await api.pricing()
    setCatalog(value)
    return value
  }

  const loadSpec = (item: Json) => {
    setSpecDraft(structuredClone(item))
    setInstanceSpecId(item.id)
    setInstanceDraft(newInstance(item))
  }

  const startSpec = () => setSpecDraft(newSpec())

  const changeBasis = (basis: string) => setSpecDraft({
    schema_version: 2,
    id: specDraft.id,
    ...pricingContract(basis),
    notes: specDraft.notes || '',
  })

  const changeBillingMode = (mode: string) => setSpecDraft({
    schema_version: 2,
    id: specDraft.id,
    ...gpuPricing(mode),
    notes: specDraft.notes || '',
  })

  const setRate = (key: string, raw: string) => setSpecDraft({
    ...specDraft,
    rates: { ...(specDraft.rates || {}), [key]: numberOrNull(raw) },
  })

  const setMetadata = (key: string, value: unknown) => setSpecDraft({
    ...specDraft,
    metadata: { ...(specDraft.metadata || {}), [key]: value },
  })

  const setTiered = (enabled: boolean) => setSpecDraft({
    ...specDraft,
    rate_tiers: enabled ? [{
      up_to_input_tokens: 128000,
      rates: Object.fromEntries(rateKeys.map((key) => [key, specDraft.rates?.[key] ?? null])),
    }] : [],
  })

  const patchTier = (index: number, patch: Json) => setSpecDraft({
    ...specDraft,
    rate_tiers: (specDraft.rate_tiers || []).map(
      (tier: Json, position: number) => position === index ? { ...tier, ...patch } : tier,
    ),
  })

  const addTier = () => {
    const tiers = [...(specDraft.rate_tiers || [])]
    const previous = tiers[tiers.length - 1]
    tiers.push({
      up_to_input_tokens: Math.max(1, Number(previous?.up_to_input_tokens || 65536) * 2),
      rates: Object.fromEntries(rateKeys.map((key) => [key, previous?.rates?.[key] ?? null])),
    })
    setSpecDraft({ ...specDraft, rate_tiers: tiers })
  }

  const setTierRate = (index: number, key: string, raw: string) => {
    const tier = (specDraft.rate_tiers || [])[index] || {}
    patchTier(index, { rates: { ...(tier.rates || {}), [key]: numberOrNull(raw) } })
  }

  const removeTier = (index: number) => setSpecDraft({
    ...specDraft,
    rate_tiers: (specDraft.rate_tiers || []).filter(
      (_: Json, position: number) => position !== index,
    ),
  })

  const setGpuQuote = (key: string, raw: string) => {
    const value = numberOrNull(raw)
    const metadata = { ...(specDraft.metadata || {}), [key]: value }
    const rates = { ...(specDraft.rates || {}) }
    if (billingMode === 'on_demand_gpu_hour') {
      rates.gpu_hour = value
    } else if (billingMode === 'monthly_node_amortized') {
      const nodeMonth = Number(metadata.quoted_node_month)
      const gpuCount = Number(metadata.node_gpu_count)
      const monthHours = Number(specDraft.amortization_hours_per_month)
      const derived = nodeMonth > 0 && gpuCount > 0 && monthHours > 0
        ? nodeMonth / gpuCount / monthHours
        : null
      metadata.derived_gpu_hour = derived
      rates.gpu_hour = derived
    }
    setSpecDraft({ ...specDraft, metadata, rates })
  }

  const saveSpec = async () => {
    if (!specDraft.id) return
    setBusy(true)
    try {
      const metadata = { ...(specDraft.metadata || {}) }
      if (!metadata.source_spec_ids?.length) delete metadata.source_spec_ids
      const saved = await api.savePricingSpec(specDraft.id, { ...specDraft, metadata })
      await refresh()
      await onRefresh()
      setSpecDraft(saved)
      setInstanceSpecId(saved.id)
      setInstanceDraft(newInstance(saved))
      notify(`PricingSpec ${saved.id} 已保存`)
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause), 'error')
    } finally {
      setBusy(false)
    }
  }

  const removeSpec = async () => {
    if (!specs.some((item) => item.id === specDraft.id)) return
    if (!window.confirm(`确认删除 PricingSpec ${specDraft.id}？`)) return
    try {
      await api.deletePricingSpec(specDraft.id)
      const value = await refresh()
      await onRefresh()
      const next = value.specs?.[0] || newSpec()
      setSpecDraft(structuredClone(next))
      setInstanceSpecId(next.id || '')
      setInstanceDraft(newInstance(next))
      notify(`PricingSpec ${specDraft.id} 已删除`)
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause), 'error')
    }
  }

  const chooseInstanceSpec = (specId: string) => {
    const item = specs.find((candidate) => candidate.id === specId) || {}
    setInstanceSpecId(specId)
    setInstanceDraft(newInstance(item))
  }

  const instantiate = async () => {
    if (!instanceDraft.id || !instanceSpecId) return
    setBusy(true)
    try {
      const saved = await api.instantiatePricing(instanceSpecId, { id: instanceDraft.id })
      await refresh()
      await onRefresh()
      setInstanceDraft(newInstance(selectedInstanceSpec))
      notify(`PricingInstance ${saved.id} 已实例化`)
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause), 'error')
    } finally {
      setBusy(false)
    }
  }

  const removeInstance = async (item: Json) => {
    if (!window.confirm(`确认删除 PricingInstance ${item.id}？`)) return
    try {
      await api.deletePricingInstance(item.id)
      await refresh()
      await onRefresh()
      notify(`PricingInstance ${item.id} 已删除`)
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause), 'error')
    }
  }

  return <section className="studio-view">
    <div className="cost-workspace management-workspace">
      <section className="cost-spec-pane management-spec-pane">
        <header className="management-spec-header">
          <div className="section-toolbar">
            <div><h1>PricingSpec</h1><span>定义一张完整、可版本化的价格表</span></div>
            <div className="toolbar-actions">
              <button className="icon-button" title="新建 PricingSpec" onClick={startSpec}><Plus size={16} /></button>
              <button className="secondary" disabled={busy || !specDraft.id} onClick={saveSpec}><Save size={15} />保存</button>
              <button className="icon-button danger" title="删除 PricingSpec" disabled={!specs.some((item) => item.id === specDraft.id)} onClick={removeSpec}><Trash2 size={15} /></button>
            </div>
          </div>
          <SpecSelector kind="PricingSpec" specs={specs} draftId={specDraft.id || ''} optionLabel={(item) => `${item.id} · ${item.billing_mode}`} onLoad={loadSpec} />
        </header>

        <div className="cost-spec-fields deployment-detail-grid">
          <label><span>PricingSpec ID</span><input value={specDraft.id || ''} onChange={(event) => setSpecDraft({ ...specDraft, id: event.target.value })} /></label>
          <label><span>Basis</span><select value={specDraft.basis || 'token_usage'} onChange={(event) => changeBasis(event.target.value)}><option value="token_usage">Token usage</option><option value="allocated_gpu_time">Allocated GPU time</option></select></label>
          {isGpu
            ? <label><span>Billing mode</span><select value={billingMode} onChange={(event) => changeBillingMode(event.target.value)}><option value="on_demand_gpu_hour">按量 GPU 卡时</option><option value="monthly_node_amortized">整机包月摊销</option><option value="internal_gpu_hour">内部 GPU 卡时</option></select></label>
            : <label><span>Billing mode</span><input disabled value="provider_token_usage" /></label>}
          <label><span>Currency</span><input maxLength={3} value={specDraft.currency || 'CNY'} onChange={(event) => setSpecDraft({ ...specDraft, currency: event.target.value.toUpperCase() })} /></label>
          <label><span>Rate-card mode</span><input disabled value={specDraft.rate_card_mode || ''} /></label>
          <label><span>Rate unit</span><input disabled value={specDraft.rate_unit || ''} /></label>
          <label><span>Supported deployments</span><input disabled value={(specDraft.supported_deployment_kinds || []).join(', ')} /></label>
          <label><span>Required rates</span><input disabled value={(specDraft.required_rates || []).join(', ')} /></label>
          <label><span>Optional rates</span><input disabled value={(specDraft.optional_rates || []).join(', ') || 'none'} /></label>
          {billingMode === 'monthly_node_amortized' && <>
            <label><span>Amortization policy</span><input disabled value={specDraft.amortization_policy || 'allocated_gpu_share'} /></label>
            <label><span>每月摊销小时数</span><input type="number" min="1" step="1" value={specDraft.amortization_hours_per_month || 730} onChange={(event) => setSpecDraft({ ...specDraft, amortization_hours_per_month: Number(event.target.value) })} /></label>
          </>}
        </div>

        {!isGpu && <label className="toggle-row cost-tier-toggle"><input type="checkbox" checked={tiered} onChange={(event) => setTiered(event.target.checked)} /><span>按单次请求输入 Token 数阶梯计价</span></label>}
        {!isGpu && !tiered && <div className="cost-rate-grid">{rateKeys.map((key) => <label key={key}><span>{key}{(specDraft.required_rates || []).includes(key) ? ' *' : ''}</span><input type="number" min="0" step="any" value={specDraft.rates?.[key] ?? ''} placeholder="未填写" onChange={(event) => setRate(key, event.target.value)} /></label>)}</div>}
        {!isGpu && tiered && <div className="cost-tier-editor">
          <div className="cost-tier-heading"><strong>Rate tiers</strong><button className="secondary compact-button" onClick={addTier}><Plus size={14} />添加阶梯</button></div>
          {(specDraft.rate_tiers || []).map((tier: Json, index: number) => <section className="cost-tier-row" key={`${index}-${tier.up_to_input_tokens}`}>
            <div className="cost-tier-limit"><label><span>输入 Token 上限</span><input type="number" min="1" step="1" value={tier.up_to_input_tokens || ''} onChange={(event) => patchTier(index, { up_to_input_tokens: Number(event.target.value) })} /></label><button className="icon-button danger" title="删除阶梯" onClick={() => removeTier(index)}><Trash2 size={14} /></button></div>
            <div className="cost-rate-grid">{rateKeys.map((key) => <label key={key}><span>{key}{(specDraft.required_rates || []).includes(key) ? ' *' : ''}</span><input type="number" min="0" step="any" value={tier.rates?.[key] ?? ''} placeholder="未填写" onChange={(event) => setTierRate(index, key, event.target.value)} /></label>)}</div>
          </section>)}
        </div>}

        {isGpu && <div className="deployment-detail-grid">
          <label><span>GPU 型号</span><input value={specDraft.metadata?.accelerator || ''} placeholder="例如 NVIDIA A800" onChange={(event) => setMetadata('accelerator', event.target.value)} /></label>
          <label><span>平台 / 提供方</span><input value={specDraft.metadata?.provider || ''} onChange={(event) => setMetadata('provider', event.target.value)} /></label>
          {billingMode === 'on_demand_gpu_hour' && <label><span>按量价格（元 / 卡 / 小时）</span><input type="number" min="0" step="any" value={specDraft.metadata?.quoted_gpu_hour ?? ''} onChange={(event) => setGpuQuote('quoted_gpu_hour', event.target.value)} /></label>}
          {billingMode === 'monthly_node_amortized' && <>
            <label><span>整机包月价格（元 / 台 / 月）</span><input type="number" min="0" step="any" value={specDraft.metadata?.quoted_node_month ?? ''} onChange={(event) => setGpuQuote('quoted_node_month', event.target.value)} /></label>
            <label><span>整机 GPU 数</span><input type="number" min="1" step="1" value={specDraft.metadata?.node_gpu_count ?? ''} onChange={(event) => setGpuQuote('node_gpu_count', event.target.value)} /></label>
            <label><span>折算价格（元 / 卡 / 小时）</span><input disabled value={specDraft.rates?.gpu_hour ?? ''} /></label>
          </>}
          {billingMode === 'internal_gpu_hour' && <label><span>内部价格（元 / 卡 / 小时）</span><input type="number" min="0" step="any" value={specDraft.rates?.gpu_hour ?? ''} onChange={(event) => setRate('gpu_hour', event.target.value)} /></label>}
        </div>}

        {!isGpu && <div className="deployment-detail-grid">
          <label><span>Provider</span><input value={specDraft.metadata?.provider || ''} onChange={(event) => setMetadata('provider', event.target.value)} /></label>
          <label><span>Region</span><input value={specDraft.metadata?.region || ''} onChange={(event) => setMetadata('region', event.target.value)} /></label>
          <label className="wide-field"><span>Applicable Model IDs</span><input value={listText(specDraft.metadata?.model_ids)} placeholder="多个 model_id 用逗号分隔" onChange={(event) => setMetadata('model_ids', parseList(event.target.value))} /></label>
          <label className="wide-field"><span>Applicable Source Spec IDs</span><input value={listText(specDraft.metadata?.source_spec_ids)} placeholder="实际 API 价格填写 APISpec ID" onChange={(event) => setMetadata('source_spec_ids', parseList(event.target.value))} /></label>
        </div>}
        <label><span>Notes</span><textarea rows={4} value={specDraft.notes || ''} onChange={(event) => setSpecDraft({ ...specDraft, notes: event.target.value })} /></label>
        <div className="team-contract-note cost-contract-note"><strong>Spec 边界</strong><span>币种、费率、阶梯、适用模型、平台报价与摊销规则全部属于 PricingSpec；价格变化时创建新的 Spec ID。</span><DetailsDisclosure value={specDraft} label="展开 PricingSpec" /></div>
      </section>

      <aside className="cost-instance-pane management-instance-pane">
        <section className="management-instance-builder">
          <div className="section-toolbar"><div><h1>PricingInstance</h1><span>将完整 PricingSpec 的当前版本实例化为可绑定对象</span></div></div>
          <label><span>PricingSpec</span><select value={instanceSpecId} onChange={(event) => chooseInstanceSpec(event.target.value)}>{specs.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.billing_mode}</option>)}</select></label>
          <label><span>PricingInstance ID</span><input value={instanceDraft.id || ''} onChange={(event) => setInstanceDraft({ ...instanceDraft, id: event.target.value })} /></label>
          <div className="reference-notice"><strong>{selectedInstanceSpec.id}</strong><span>{selectedInstanceSpec.currency || '---'} · {selectedInstanceSpec.status || 'unknown'}。实例会自动记录 Spec 指纹和创建时间，不再重复填写报价。</span></div>
          <button className="primary" disabled={busy || !instanceDraft.id || !instanceSpecId} onClick={instantiate}><Check size={15} />实例化 Pricing</button>
          <DetailsDisclosure value={instanceDraft} label="展开 PricingInstance 草稿" />
        </section>

        <section className="management-instance-registry">
          <div className="panel-heading"><h2>已注册 PricingInstance</h2><span>{instances.length}</span></div>
          {instances.map((item) => <article key={item.id} className="process-card">
            <div><span className={`status-pill ${item.status === 'ready' ? 'ready' : 'incomplete'}`}>{item.status || 'unknown'}</span><strong>{item.id}</strong><small>{item.pricing_spec_id} · {item.currency}</small></div>
            <dl>
              <dt>Created</dt><dd>{item.created_at_utc || 'unknown'}</dd>
              <dt>Billing</dt><dd>{item.billing_mode}</dd>
              <dt>Basis</dt><dd>{item.basis}</dd>
              {item.basis === 'allocated_gpu_time' ? <><dt>GPU</dt><dd>{item.metadata?.accelerator || 'unspecified'}</dd><dt>GPU-hour</dt><dd>{item.rates?.gpu_hour ?? 'not configured'} {item.currency}</dd></> : <><dt>Rate tiers</dt><dd>{item.rate_tiers?.length ? `${item.rate_tiers.length} tiers` : 'flat'}</dd><dt>Model IDs</dt><dd>{listText(item.metadata?.model_ids) || 'not configured'}</dd></>}
            </dl>
            <DetailsDisclosure value={item} label="展开 PricingInstance" className="card-details" />
            <div className="row-actions"><button className="icon-button danger" title="删除 PricingInstance" onClick={() => removeInstance(item)}><Trash2 size={14} /></button></div>
          </article>)}
        </section>
      </aside>
    </div>
  </section>
}
