import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { CheckCircle2, Cloud, Cpu, Database, Folder, KeyRound, ListChecks, Network, Plus, RefreshCw, Search, Settings2, ShieldCheck, Trash2, XCircle } from 'lucide-react'
import { api } from './api'
import DetailsDisclosure from './DetailsDisclosure'
import SpecSelector from './SpecSelector'
import type { Bootstrap, Json } from './types'

interface Props {
  bootstrap: Bootstrap
  environment: Json
  notify: (message: string) => void
  onRefresh: () => void
}

const terminal = new Set(['completed', 'failed', 'stopped', 'finished_after_server_restart'])

const bytes = (value?: number) => {
  if (!value) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = value
  let index = 0
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1 }
  return `${size.toFixed(index > 1 ? 1 : 0)} ${units[index]}`
}

const registeredBenchmarkProviders = (sources: Json = {}) => {
  const standardProviders = new Set(['modelscope', 'huggingface', 'github'])
  return Object.entries(sources)
    .filter(([provider, source]) => standardProviders.has(provider) && Array.isArray((source as Json)?.default_ids) && (source as Json).default_ids.length > 0)
    .map(([provider]) => providerLabel(provider))
    .join(', ') || 'none'
}

export default function ResourcesView({ bootstrap, environment, notify, onRefresh }: Props) {
  const [assetKind, setAssetKind] = useState<'benchmark' | 'model' | 'api'>('benchmark')
  const [rawRoot, setRawRoot] = useState(bootstrap.paths.raw_root)
  const [preparedRoot, setPreparedRoot] = useState(bootstrap.paths.prepared_root)
  const [modelsRoot, setModelsRoot] = useState(bootstrap.paths.models_root)
  const [catalog, setCatalog] = useState<Json>({ benchmarks: bootstrap.benchmarks, benchmark_specs: bootstrap.benchmark_registry?.specs || [], benchmark_instances: bootstrap.benchmark_registry?.instances || [], model_specs: bootstrap.model_registry?.specs || [], model_instances: bootstrap.model_registry?.instances || [], api_registry: bootstrap.api_registry || { specs: [], instances: [] } })
  const [jobs, setJobs] = useState<Record<string, Json>>({})
  const [scanBusy, setScanBusy] = useState<'benchmark' | 'model' | ''>('')
  const [checkingBenchmark, setCheckingBenchmark] = useState('')
  const [benchmarkScanReport, setBenchmarkScanReport] = useState<Json | null>(null)
  const [modelScanReport, setModelScanReport] = useState<Json | null>(null)
  const [showProxy, setShowProxy] = useState(false)
  const [proxy, setProxy] = useState<Json>({ enabled: false, http_proxy: '', https_proxy: '', all_proxy: '', no_proxy: '127.0.0.1,localhost' })
  const firstBenchmarkSpec = bootstrap.benchmark_registry?.specs?.[0]
  const [benchmarkForm, setBenchmarkForm] = useState<Json>({ category: firstBenchmarkSpec?.category || '', benchmark_spec_id: firstBenchmarkSpec?.id || '', instance_id: firstBenchmarkSpec ? `${firstBenchmarkSpec.id}-instance` : '', mode: 'download', source: 'auto', source_path: '', stage: 'prepared' })
  const firstModelSpec = bootstrap.model_registry?.specs?.[0]
  const [modelForm, setModelForm] = useState<Json>({ organization: firstModelSpec?.organization || '', model_spec_id: firstModelSpec?.id || '', instance_id: firstModelSpec ? `${firstModelSpec.id}-instance` : '', mode: 'download', source: 'auto', source_path: '', revision: '' })
  const firstApiSpec = bootstrap.api_registry?.specs?.[0]
  const [apiForm, setApiForm] = useState<Json>(apiFormForSpec(firstApiSpec))
  const [benchmarkSpecId, setBenchmarkSpecId] = useState(firstBenchmarkSpec?.id || '')
  const [modelSpecId, setModelSpecId] = useState(firstModelSpec?.id || '')
  const [apiSpecId, setApiSpecId] = useState(firstApiSpec?.id || '')

  const refreshCatalog = async () => {
    try {
      setCatalog(await api.resources(rawRoot, preparedRoot, modelsRoot))
      onRefresh()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  useEffect(() => {
    const active = Object.entries(jobs).filter(([, value]) => !terminal.has(value.state?.status || value.status || 'running'))
    if (!active.length) return
    const timer = window.setInterval(async () => {
      const next = { ...jobs }
      let completed = false
      await Promise.all(active.map(async ([key, value]) => {
        try {
          const progress = await api.launchProgress(value.launch_id)
          next[key] = { ...value, ...progress }
          if (terminal.has(progress.state?.status)) completed = true
        } catch { /* the next refresh can recover */ }
      }))
      setJobs(next)
      if (completed) void refreshCatalog()
    }, 900)
    return () => window.clearInterval(timer)
  }, [jobs, rawRoot, preparedRoot, modelsRoot])

  const registerJob = (key: string, value: Json) => {
    setJobs((current) => ({ ...current, [key]: { ...value, percent: 0, message: 'starting' } }))
    notify(`资源任务已启动：${value.launch_id}`)
  }

  const instantiateBenchmark = async () => {
    try {
      if (!benchmarkForm.instance_id) throw new Error('请填写 BenchmarkInstance ID')
      const value = await api.instantiateBenchmark(benchmarkForm.instance_id, { ...benchmarkForm, raw_root: rawRoot, prepared_root: preparedRoot, python: environment.python, proxy })
      if (value.job) registerJob(`benchmark:${benchmarkForm.instance_id}`, value.job)
      notify(`BenchmarkInstance ${value.instance.id} 已创建`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const instantiateModel = async () => {
    try {
      const value = await api.instantiateModel(modelForm.instance_id, { ...modelForm, models_root: modelsRoot, python: environment.python, proxy })
      if (value.job) registerJob(`model:${modelForm.instance_id}`, value.job)
      notify(`ModelInstance ${value.instance.id} 已创建`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const instantiateApi = async () => {
    try {
      const value = await api.instantiateApi(apiForm.instance_id, apiForm)
      notify(`APIInstance ${value.id} 已创建`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const scanBenchmarks = async () => {
    setScanBusy('benchmark')
    try {
      const result = await api.scanBenchmarkInstances(rawRoot, preparedRoot)
      setBenchmarkScanReport(result)
      const summary = result.summary || {}
      notify(`Benchmark 扫描完成：新建 ${summary.created_instances || 0}，已存在 ${summary.existing_instances || 0}，仅 Raw ${summary.raw_only || 0}`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setScanBusy('') }
  }

  const scanModels = async () => {
    setScanBusy('model')
    try {
      const result = await api.scanModelInstances(modelsRoot)
      setModelScanReport(result)
      const summary = result.summary || {}
      notify(`模型扫描完成：新建 ${summary.created_instances || 0}，已存在 ${summary.existing_instances || 0}，未匹配 ${summary.unmatched_complete_models || 0}`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setScanBusy('') }
  }

  const removeInstance = async (kind: 'benchmark' | 'model' | 'api', item: Json) => {
    const label = kind === 'benchmark' ? 'BenchmarkInstance' : kind === 'model' ? 'ModelInstance' : 'APIInstance'
    if (!window.confirm(`确认注销 ${label} ${item.id}？\n真实数据或模型文件会保留。`)) return
    try {
      if (kind === 'benchmark') await api.deleteBenchmarkInstance(item.id)
      if (kind === 'model') await api.deleteModelInstance(item.id)
      if (kind === 'api') await api.deleteApiInstance(item.id)
      notify(`${label} ${item.id} 已注销`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
  }

  const removeBenchmarkInstance = async (item: Json) => {
    await removeInstance('benchmark', item)
  }

  const checkBenchmarkInstance = async (item: Json) => {
    setCheckingBenchmark(item.id)
    try {
      const checked = await api.checkBenchmarkInstance(item.id)
      const failed = (checked.loader_check?.tasks || []).filter((task: Json) => task.status !== 'ready')
      notify(checked.loader_status === 'ready'
        ? `BenchmarkInstance ${item.id} loader 检测通过`
        : `BenchmarkInstance ${item.id} loader 检测失败${failed.length ? `：${failed.map((task: Json) => task.task).join(', ')}` : ''}`)
      await refreshCatalog()
    } catch (cause) { notify(cause instanceof Error ? cause.message : String(cause)) }
    finally { setCheckingBenchmark('') }
  }

  const activeJobs = useMemo(() => Object.values(jobs).filter((item) => !terminal.has(item.state?.status)), [jobs])
  const benchmarkSpecs: Json[] = catalog.benchmark_specs || []
  const benchmarkInstances: Json[] = catalog.benchmark_instances || []
  const modelSpecs: Json[] = catalog.model_specs || []
  const modelInstances: Json[] = catalog.model_instances || []
  const apiSpecs: Json[] = catalog.api_registry?.specs || []
  const apiInstances: Json[] = catalog.api_registry?.instances || []
  const selectedBenchmarkSpec = benchmarkSpecs.find((item: Json) => item.id === benchmarkForm.benchmark_spec_id)
  const selectedBenchmarkCatalog = catalog.benchmarks.find((item: Json) => item.full_prepare_target === selectedBenchmarkSpec?.prepare_target)
  const selectedModelSpec = modelSpecs.find((item: Json) => item.id === modelForm.model_spec_id)
  const selectedApiSpec = apiSpecs.find((item: Json) => item.id === apiForm.api_spec_id)
  const browsedBenchmarkSpec = benchmarkSpecs.find((item: Json) => item.id === benchmarkSpecId) || benchmarkSpecs[0]
  const browsedModelSpec = modelSpecs.find((item: Json) => item.id === modelSpecId) || modelSpecs[0]
  const browsedApiSpec = apiSpecs.find((item: Json) => item.id === apiSpecId) || apiSpecs[0]

  const loadBenchmarkSpec = (spec: Json) => {
    setBenchmarkSpecId(spec.id)
    setBenchmarkForm({ category: spec.category || '', benchmark_spec_id: spec.id, instance_id: `${spec.id}-instance`, mode: 'download', source: 'auto', source_path: '', stage: 'prepared' })
  }

  const loadModelSpec = (spec: Json) => {
    const hasSources = Boolean(Object.keys(spec?.sources || {}).length)
    setModelForm({ organization: spec?.organization || '', model_spec_id: spec?.id || '', instance_id: spec ? `${spec.id}-instance` : '', mode: hasSources ? 'download' : 'local', source: 'auto', source_path: '', revision: '' })
    setModelSpecId(spec.id)
  }

  const loadApiSpec = (spec: Json) => {
    setApiSpecId(spec.id)
    setApiForm(apiFormForSpec(spec))
  }

  return <section className="resource-view">
    <div className="section-toolbar"><div><h1>资源中心</h1><span>Spec 定义平台支持目录，Instance 表示已显式实例化的资源 · {activeJobs.length} active jobs</span></div><div className="toolbar-actions"><button className={`secondary ${proxy.enabled ? 'enabled' : ''}`} onClick={() => setShowProxy(true)}><Network size={15} />下载代理</button><button className="secondary" onClick={refreshCatalog}><RefreshCw size={15} />重新检测</button></div></div>

    {Object.keys(jobs).length > 0 && <div className="resource-jobs"><div className="panel-heading"><h2>资源任务</h2><span>关闭页面不会停止服务器子进程</span></div>{Object.entries(jobs).map(([key, job]) => <div className="resource-job-row" key={key}><strong>{key}</strong><code>{job.launch_id}</code><JobProgress job={job} /></div>)}</div>}

    <nav className="resource-kind-tabs" aria-label="资源类型">
      <button className={assetKind === 'benchmark' ? 'active' : ''} onClick={() => setAssetKind('benchmark')}><Database size={17} /><span>Benchmark<small>{benchmarkInstances.length} instances</small></span></button>
      <button className={assetKind === 'model' ? 'active' : ''} onClick={() => setAssetKind('model')}><Cpu size={17} /><span>模型<small>{modelInstances.length} instances</small></span></button>
      <button className={assetKind === 'api' ? 'active' : ''} onClick={() => setAssetKind('api')}><Cloud size={17} /><span>API<small>{apiInstances.length} instances</small></span></button>
    </nav>

    <div className="resource-workspace management-workspace">
      <section className="resource-spec-pane management-spec-pane">
        {assetKind === 'benchmark' && <>
          <header className="management-spec-header"><div className="section-toolbar"><div><h1>BenchmarkSpec</h1><span>基础身份信息与对应 Benchmark 的完整运行定义</span></div></div><SpecSelector kind="BenchmarkSpec" specs={benchmarkSpecs} draftId={browsedBenchmarkSpec?.id || ''} optionLabel={(item) => `${item.id} · ${item.category}`} onLoad={loadBenchmarkSpec} /></header>
          <div className="resource-spec-body">
            <div className="resource-paths"><PathField label="Benchmark raw root" value={rawRoot} onChange={setRawRoot} /><PathField label="Benchmark prepared root" value={preparedRoot} onChange={setPreparedRoot} /><button className="secondary scan-button" disabled={Boolean(scanBusy)} onClick={scanBenchmarks}><Search size={15} />{scanBusy === 'benchmark' ? '扫描中' : '扫描并注册'}</button></div>
            {benchmarkScanReport && <DetailsDisclosure value={benchmarkScanReport} label="展开最近一次 Benchmark 扫描报告" className="scan-report" />}
            {browsedBenchmarkSpec && <>
              <div className="resource-spec-fields deployment-detail-grid">
                <ReadOnlyField label="BenchmarkSpec ID" value={browsedBenchmarkSpec.id} />
                <ReadOnlyField label="Schema version" value={browsedBenchmarkSpec.schema_version} />
                <ReadOnlyField label="名称" value={browsedBenchmarkSpec.name} />
                <ReadOnlyField label="类别" value={browsedBenchmarkSpec.category} />
              </div>
              <BenchmarkContractView spec={browsedBenchmarkSpec} />
            </>}
          </div>
        </>}
        {assetKind === 'model' && <>
          <header className="management-spec-header"><div className="section-toolbar"><div><h1>ModelSpec</h1><span>平台登记的模型产品、来源与能力边界</span></div></div><SpecSelector kind="ModelSpec" specs={modelSpecs} draftId={browsedModelSpec?.id || ''} optionLabel={(item) => `${item.id} · ${item.organization}`} onLoad={loadModelSpec} /></header>
          <div className="resource-spec-body">
            <div className="resource-paths"><PathField label="Model root" value={modelsRoot} onChange={setModelsRoot} /><button className="secondary scan-button" disabled={Boolean(scanBusy)} onClick={scanModels}><Search size={15} />{scanBusy === 'model' ? '扫描中' : '扫描并注册'}</button></div>
            {modelScanReport && <DetailsDisclosure value={modelScanReport} label="展开最近一次模型扫描报告" className="scan-report" />}
            {browsedModelSpec && <><div className="resource-spec-fields deployment-detail-grid"><ReadOnlyField label="ModelSpec ID" value={browsedModelSpec.id} /><ReadOnlyField label="名称" value={browsedModelSpec.name} /><ReadOnlyField label="公司 / 机构" value={browsedModelSpec.organization} /><ReadOnlyField label="模型家族" value={browsedModelSpec.family} /><ReadOnlyField label="参数规模" value={browsedModelSpec.parameter_size} /><ReadOnlyField label="许可证" value={browsedModelSpec.license || 'unspecified'} /><ReadOnlyField label="支持的后端" value={(browsedModelSpec.supported_backends || []).join(', ')} /><ReadOnlyField label="Thinking 协议" value={browsedModelSpec.thinking_protocol || 'none'} /><ReadOnlyField label="能力" value={Object.entries(browsedModelSpec.capabilities || {}).filter(([, enabled]) => Boolean(enabled)).map(([name]) => name).join(', ')} wide /></div><div className="resource-spec-summary"><dl><dt>Gated</dt><dd>{browsedModelSpec.gated ? 'yes' : 'no'}</dd><dt>ModelScope sources</dt><dd>{browsedModelSpec.sources?.modelscope?.length || 0}</dd><dt>HuggingFace sources</dt><dd>{browsedModelSpec.sources?.huggingface?.length || 0}</dd></dl><DetailsDisclosure value={browsedModelSpec} label="展开 ModelSpec" /></div></>}
          </div>
        </>}
        {assetKind === 'api' && <>
          <header className="management-spec-header"><div className="section-toolbar"><div><h1>APISpec</h1><span>平台登记的 API 产品、协议与默认连接约束</span></div></div><SpecSelector kind="APISpec" specs={apiSpecs} draftId={browsedApiSpec?.id || ''} optionLabel={(item) => `${item.id} · ${item.provider}`} onLoad={loadApiSpec} /></header>
          <div className="resource-spec-body">
            {browsedApiSpec && <><div className="resource-spec-fields deployment-detail-grid"><ReadOnlyField label="APISpec ID" value={browsedApiSpec.id} /><ReadOnlyField label="名称" value={browsedApiSpec.name} /><ReadOnlyField label="提供商" value={browsedApiSpec.provider} /><ReadOnlyField label="Provider key" value={browsedApiSpec.provider_key} /><ReadOnlyField label="公司 / 机构" value={browsedApiSpec.organization} /><ReadOnlyField label="协议" value={browsedApiSpec.protocol} /><ReadOnlyField label="Deployment kind" value={browsedApiSpec.deployment_kind} /><ReadOnlyField label="鉴权模式" value={(browsedApiSpec.auth_modes || []).join(', ')} /><ReadOnlyField label="允许的 ModelSpec" value={(browsedApiSpec.allowed_model_spec_ids || []).join(', ')} wide /><ReadOnlyField label="默认 Base URL" value={browsedApiSpec.default_base_url} wide /></div><div className="resource-spec-summary"><dl><dt>Shared</dt><dd>{browsedApiSpec.shared ? 'yes' : 'no'}</dd><dt>Max concurrency</dt><dd>{browsedApiSpec.request_limits?.max_concurrency || 'unlimited'}</dd><dt>Protocol capabilities</dt><dd>{Object.keys(browsedApiSpec.protocol_capabilities || {}).join(', ') || 'none'}</dd></dl><DetailsDisclosure value={browsedApiSpec} label="展开 APISpec" /></div></>}
          </div>
        </>}
      </section>

      <aside className="resource-instance-pane management-instance-pane">
        <section className="management-instance-builder">
          {assetKind === 'benchmark' && <><div className="section-toolbar"><div><h1>BenchmarkInstance</h1><span>为左侧载入的 BenchmarkSpec 配置数据实例</span></div></div><ReadOnlyField label="当前 BenchmarkSpec" value={selectedBenchmarkSpec?.id || ''} /><div className="two-fields"><label><span>数据获取方式</span><select value={benchmarkForm.mode} onChange={(event) => setBenchmarkForm({ ...benchmarkForm, mode: event.target.value })}><option value="download">通过注册渠道下载</option><option value="local">本地导入</option></select></label><label><span>BenchmarkInstance ID</span><input value={benchmarkForm.instance_id} onChange={(event) => setBenchmarkForm({ ...benchmarkForm, instance_id: event.target.value })} /></label></div>{benchmarkForm.mode === 'download' ? <><label><span>下载渠道</span><ProviderSelect value={benchmarkForm.source} sources={selectedBenchmarkCatalog?.download_sources || {}} onChange={(source) => setBenchmarkForm({ ...benchmarkForm, source })} /></label><SourceList sources={selectedBenchmarkCatalog?.download_sources || {}} /></> : <><label><span>服务器本地数据路径</span><input value={benchmarkForm.source_path} onChange={(event) => setBenchmarkForm({ ...benchmarkForm, source_path: event.target.value })} placeholder="/data/shared/benchmark" /></label><label><span>本地数据阶段</span><select value={benchmarkForm.stage} onChange={(event) => setBenchmarkForm({ ...benchmarkForm, stage: event.target.value })}><option value="prepared">Prepared（可直接供 loader 使用）</option><option value="raw">Raw（实例化时转换为 Prepared）</option></select></label></>}<div className="reference-notice"><strong>{selectedBenchmarkSpec?.name || '未载入 BenchmarkSpec'}</strong><span>使用完整准备目标 <code>{selectedBenchmarkSpec?.prepare_target || '—'}</code>；实例会固定当前 Spec 指纹。</span></div><button className="primary" disabled={!benchmarkForm.benchmark_spec_id || !benchmarkForm.instance_id || (benchmarkForm.mode === 'local' && !benchmarkForm.source_path)} onClick={instantiateBenchmark}><Plus size={15} />实例化 Benchmark</button></>}
          {assetKind === 'model' && <><div className="section-toolbar"><div><h1>ModelInstance</h1><span>为左侧载入的 ModelSpec 配置模型实例</span></div></div><ReadOnlyField label="当前 ModelSpec" value={selectedModelSpec?.id || ''} /><div className="two-fields"><label><span>模型获取方式</span><select value={modelForm.mode} onChange={(event) => setModelForm({ ...modelForm, mode: event.target.value })}><option value="download" disabled={!Object.keys(selectedModelSpec?.sources || {}).length}>通过注册渠道下载</option><option value="local">本地导入</option></select></label><label><span>ModelInstance ID</span><input value={modelForm.instance_id} onChange={(event) => setModelForm({ ...modelForm, instance_id: event.target.value })} /></label></div>{modelForm.mode === 'download' ? <><label><span>下载渠道</span><ProviderSelect value={modelForm.source} sources={selectedModelSpec?.sources || {}} onChange={(source) => setModelForm({ ...modelForm, source })} /></label><SourceList sources={selectedModelSpec?.sources || {}} /><label><span>Revision（可选）</span><input value={modelForm.revision} onChange={(event) => setModelForm({ ...modelForm, revision: event.target.value })} /></label></> : <label><span>服务器本地模型路径</span><input value={modelForm.source_path} onChange={(event) => setModelForm({ ...modelForm, source_path: event.target.value })} placeholder="/data/shared/model" /></label>}<div className="reference-notice"><strong>{selectedModelSpec?.name || '未载入 ModelSpec'}</strong><span>下载写入 <code>{modelsRoot}/{modelForm.instance_id || '&lt;instance-id&gt;'}</code>；本地导入只记录路径。</span></div><button className="primary" disabled={!modelForm.model_spec_id || !modelForm.instance_id || (modelForm.mode === 'local' && !modelForm.source_path) || (modelForm.mode === 'download' && !Object.keys(selectedModelSpec?.sources || {}).length)} onClick={instantiateModel}><Plus size={15} />实例化模型</button></>}
          {assetKind === 'api' && <><div className="section-toolbar"><div><h1>APIInstance</h1><span>为左侧载入的 APISpec 配置连接实例；模型在 DeploymentSpec 中单独选择</span></div></div><ReadOnlyField label="当前 APISpec" value={selectedApiSpec?.id || ''} /><div className="two-fields"><label><span>鉴权模式</span><select value={apiForm.auth_mode} onChange={(event) => setApiForm({ ...apiForm, auth_mode: event.target.value })}>{(selectedApiSpec?.auth_modes || []).map((mode: string) => <option key={mode} value={mode}>{mode === 'none' ? '无需鉴权' : 'API key'}</option>)}</select></label><label><span>APIInstance ID</span><input value={apiForm.instance_id} onChange={(event) => setApiForm({ ...apiForm, instance_id: event.target.value })} /></label></div><label><span>Base URL</span><input disabled={!selectedApiSpec?.base_url_editable} value={apiForm.base_url} onChange={(event) => setApiForm({ ...apiForm, base_url: event.target.value })} /></label>{apiForm.auth_mode === 'env' && <><label><span>API key env</span><input value={apiForm.api_key_env} onChange={(event) => setApiForm({ ...apiForm, api_key_env: event.target.value })} /></label><label><span>API key（仅当前 Studio 会话，可选）</span><div className="field-icon"><KeyRound size={14} /><input type="password" autoComplete="new-password" value={apiForm.api_key} onChange={(event) => setApiForm({ ...apiForm, api_key: event.target.value })} /></div></label></>}<div className="two-fields"><label><span>最大并发</span><input type="number" min="0" value={apiForm.request_limits?.max_concurrency || 0} onChange={(event) => setApiForm({ ...apiForm, request_limits: { ...(apiForm.request_limits || {}), max_concurrency: Number(event.target.value) } })} /></label><label><span>最小请求间隔（秒）</span><input type="number" min="0" step="0.1" value={apiForm.request_limits?.min_interval_s || 0} onChange={(event) => setApiForm({ ...apiForm, request_limits: { ...(apiForm.request_limits || {}), min_interval_s: Number(event.target.value) } })} /></label></div><div className="reference-notice"><strong>{selectedApiSpec?.provider || '未载入 APISpec'} · {selectedApiSpec?.name || ''}</strong><span>密钥明文不会写入配置文件；实例只保存访问通道，不复制模型能力。</span></div><button className="primary" disabled={!apiForm.api_spec_id || !apiForm.instance_id || !apiForm.base_url || (apiForm.auth_mode === 'env' && !apiForm.api_key_env && !apiForm.api_key)} onClick={instantiateApi}><Plus size={15} />实例化 API</button></>}
        </section>

        <section className="management-instance-registry resource-instance-registry">
          <div className="panel-heading"><h2>已注册 {assetKind === 'benchmark' ? 'BenchmarkInstance' : assetKind === 'model' ? 'ModelInstance' : 'APIInstance'}</h2><span>{assetKind === 'benchmark' ? benchmarkInstances.length : assetKind === 'model' ? modelInstances.length : apiInstances.length}</span></div>
          {assetKind === 'benchmark' && benchmarkInstances.map((item: Json) => {
            const job = jobs[`benchmark:${item.id}`]
            const rawPath = item.raw_path || item.provenance?.find((source: Json) => source.raw_path || (source.stage === 'raw' && source.path))?.raw_path || item.provenance?.find((source: Json) => source.stage === 'raw')?.path
            return <article className="process-card resource-instance-card" key={item.id}><div><Status value={item.observed_status || item.status} /><strong>{item.id}</strong><small>{item.benchmark_spec_id} · {acquisitionLabel(item)}</small></div><dl><dt>Raw</dt><dd title={rawPath}>{rawPath || 'not registered'}</dd><dt>Prepared</dt><dd title={item.prepared_path}>{item.prepared_path}</dd><dt>Size</dt><dd>{bytes(item.integrity?.size_bytes)}</dd><dt>Loader</dt><dd>{item.loader_status || 'unchecked'}</dd></dl><JobProgress job={job} /><DetailsDisclosure value={item} label="展开 BenchmarkInstance" className="card-details" /><div className="row-actions"><button className="icon-button" disabled={checkingBenchmark === item.id || item.status !== 'ready'} title="检测 runnable task loaders" onClick={() => checkBenchmarkInstance(item)}><RefreshCw size={14} /></button><button className="icon-button danger" title="注销 BenchmarkInstance，保留数据文件" onClick={() => removeBenchmarkInstance(item)}><Trash2 size={14} /></button></div></article>
          })}
          {assetKind === 'model' && modelInstances.map((item: Json) => <article className="process-card resource-instance-card" key={item.id}><div><Status value={item.observed_status || item.status} /><strong>{item.id}</strong><small>{item.model_spec_id} · {acquisitionLabel(item)}</small></div><dl><dt>Path</dt><dd title={item.path}>{item.path}</dd><dt>Size</dt><dd>{bytes(item.integrity?.size_bytes)}</dd><dt>Weights</dt><dd>{item.integrity?.weight_files || 0} files</dd><dt>Lifecycle</dt><dd>{item.configuration_status || 'unknown'}</dd></dl><JobProgress job={jobs[`model:${item.id}`]} /><DetailsDisclosure value={item} label="展开 ModelInstance" className="card-details" /><div className="row-actions"><button className="icon-button danger" title="注销 ModelInstance，保留模型文件" onClick={() => removeInstance('model', item)}><Trash2 size={14} /></button></div></article>)}
          {assetKind === 'api' && apiInstances.map((item: Json) => <article className="process-card resource-instance-card" key={item.id}><div><Status value={item.configuration_status} /><strong>{item.id}</strong><small>{item.api_spec_id} · access channel</small></div><dl><dt>Endpoint</dt><dd title={item.base_url}>{item.base_url}</dd><dt>Auth</dt><dd>{item.auth_mode === 'none' ? 'none' : item.api_key_env}</dd><dt>Concurrency</dt><dd>{item.request_limits?.max_concurrency || 'unlimited'}</dd><dt>Lifecycle</dt><dd>{item.validation_error ? 'invalid' : 'current'}</dd></dl><DetailsDisclosure value={item} label="展开 APIInstance" className="card-details" /><div className="row-actions"><button className="icon-button danger" title="注销 APIInstance" onClick={() => removeInstance('api', item)}><Trash2 size={14} /></button></div></article>)}
          {assetKind === 'benchmark' && !benchmarkInstances.length && <div className="empty-state">尚未创建 BenchmarkInstance</div>}
          {assetKind === 'model' && !modelInstances.length && <div className="empty-state">尚未创建 ModelInstance</div>}
          {assetKind === 'api' && !apiInstances.length && <div className="empty-state">尚未创建 APIInstance</div>}
        </section>
      </aside>
    </div>

    {showProxy && <Modal title="下载代理" onClose={() => setShowProxy(false)}><label className="check-field"><input type="checkbox" checked={proxy.enabled} onChange={(event) => setProxy({ ...proxy, enabled: event.target.checked })} /><span>仅为后续下载子进程启用代理</span></label><label><span>HTTP proxy</span><input value={proxy.http_proxy} onChange={(event) => setProxy({ ...proxy, http_proxy: event.target.value })} placeholder="http://127.0.0.1:7890" /></label><label><span>HTTPS proxy</span><input value={proxy.https_proxy} onChange={(event) => setProxy({ ...proxy, https_proxy: event.target.value })} placeholder="http://127.0.0.1:7890" /></label><label><span>ALL proxy</span><input value={proxy.all_proxy} onChange={(event) => setProxy({ ...proxy, all_proxy: event.target.value })} placeholder="socks5://127.0.0.1:7890" /></label><label><span>NO proxy</span><input value={proxy.no_proxy} onChange={(event) => setProxy({ ...proxy, no_proxy: event.target.value })} /></label><div className="modal-actions"><button className="primary" onClick={() => setShowProxy(false)}>完成</button></div></Modal>}



  </section>
}

function PathField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label><span>{label}</span><div className="input-with-icon"><Folder size={15} /><input value={value} onChange={(event) => onChange(event.target.value)} /></div></label>
}

function ReadOnlyField({ label, value, wide = false }: { label: string; value?: string | number | boolean | null; wide?: boolean }) {
  const displayValue = value === undefined || value === null || value === '' ? '—' : String(value)
  return <label className={wide ? 'wide-field' : ''}><span>{label}</span><input disabled value={displayValue} /></label>
}

function BenchmarkContractView({ spec }: { spec: Json }) {
  const contract = spec.implementation || spec
  const tasks = Object.entries(contract.task_contracts || {}) as Array<[string, Json]>
  const sources = ['modelscope', 'huggingface', 'github'].map((provider) => ({
    provider,
    config: (contract.sources || {})[provider] || {},
  })).filter(({ config }) => Array.isArray(config.default_ids) && config.default_ids.length > 0)
  const otherSources: Json[] = contract.sources?.other_defaults || []
  const fallbackFiles: Json[] = contract.sources?.fallback_files || []
  const requiredCapabilities: string[] = contract.capabilities?.required || []
  const capabilityFlags = Object.entries(contract.capabilities || {}).filter(([key]) => key !== 'required')
  const sandboxes = Object.entries(contract.sandbox_profiles || {}) as Array<[string, Json]>
  const runtime = Object.entries(contract.runtime_defaults || {})
  const network = contract.network_defaults || {}
  const networkTargets = Object.entries(network.targets || {})

  return <section className="benchmark-contract-view">
    <header className="benchmark-contract-header">
      <div><strong>Benchmark 运行定义</strong><span>只读</span></div>
      <code>Benchmark ID: {contract.id}</code>
    </header>

    <section className="benchmark-contract-section wide data-section">
      <div className="benchmark-contract-title"><Database size={16} /><div><h3>数据准备与来源</h3><small>准备目标、别名与注册下载源</small></div></div>
      <div className="benchmark-contract-kv compact"><div><span>完整准备目标</span><code>{contract.prepare_target || '—'}</code></div><div><span>标准平台</span><strong>{registeredBenchmarkProviders(contract.sources)}</strong></div></div>
      <h4>准备目标与别名</h4>
      <div className="contract-tags">{(contract.prepare_targets || []).map((target: string) => <code className={target === contract.prepare_target ? 'contract-tag primary' : 'contract-tag'} key={target}>{target}</code>)}</div>
      <div className="contract-source-list">
        {sources.map(({ provider, config }) => <div className="contract-source-row" key={provider}><strong>{providerLabel(provider)}</strong><div>{config.default_ids.map((sourceId: string) => <code key={sourceId}>{sourceId}</code>)}</div>{config.env && <small>override: {config.env}</small>}</div>)}
        {!sources.length && <span className="contract-empty">没有已注册的标准下载源</span>}
      </div>
      {(otherSources.length > 0 || fallbackFiles.length > 0) && <div className="benchmark-contract-kv compact"><div><span>其它候选来源</span><strong>{otherSources.length}</strong></div><div><span>文件级后备来源</span><strong>{fallbackFiles.length}</strong></div></div>}
    </section>

    <section className="benchmark-contract-section wide task-section">
      <div className="benchmark-contract-title"><ListChecks size={16} /><div><h3>任务与评分</h3><small>可运行任务、答案提取与官方评分</small></div></div>
      <div className="contract-table-wrap"><table className="contract-table"><thead><tr><th>Runnable task</th><th>Kind</th><th>Extractor</th><th>默认评分口径</th><th>Scorer</th></tr></thead><tbody>{tasks.map(([task, value]) => {
        const scoring = value.scoring || {}
        const defaultProfile = scoring.default_profile || '—'
        const scorer = scoring.profiles?.[defaultProfile]?.scorer_id || '—'
        return <tr key={task}><td><code>{task}</code></td><td>{value.kind || '—'}</td><td>{value.extractor || '—'}</td><td>{defaultProfile}</td><td>{scorer}</td></tr>
      })}</tbody></table></div>
    </section>

    <section className="benchmark-contract-section capability-section">
      <div className="benchmark-contract-title"><ShieldCheck size={16} /><div><h3>能力与沙盒</h3><small>运行所需能力与隔离环境</small></div></div>
      <h4>必需能力</h4>
      <div className="contract-tags">{requiredCapabilities.length ? requiredCapabilities.map((capability) => <span className="contract-tag capability" key={capability}>{humanizeKey(capability)}</span>) : <span className="contract-empty">无特殊能力要求</span>}</div>
      {capabilityFlags.length > 0 && <div className="benchmark-contract-kv">{capabilityFlags.map(([key, value]) => <div key={key}><span>{humanizeKey(key)}</span><strong className={value === true ? 'positive' : ''}>{contractValue(value)}</strong></div>)}</div>}
      <h4>沙盒配置</h4>
      <div className="contract-sandbox-list">{sandboxes.length ? sandboxes.map(([id, value]) => <div key={id}><strong>{value.label || id}</strong><code>{value.docker_image || 'no image'}</code>{value.description && <small>{value.description}</small>}</div>) : <span className="contract-empty">使用通用 Python sandbox</span>}</div>
    </section>

    <section className="benchmark-contract-section runtime-section">
      <div className="benchmark-contract-title"><Settings2 size={16} /><div><h3>运行与网络</h3><small>默认运行参数与网络策略</small></div></div>
      <div className="contract-subsection"><h4>默认运行参数</h4><div className="benchmark-contract-kv">{runtime.map(([key, value]) => <div key={key}><span>{humanizeKey(key)}</span><strong>{contractValue(value)}</strong></div>)}</div></div>
      <div className="contract-subsection"><h4>网络策略</h4><div className="benchmark-contract-kv"><div><span>Access</span><strong>{network.access || 'optional'}</strong></div><div><span>Mode</span><strong>{network.mode || 'direct'}</strong></div>{network.proxy_url && <div><span>Proxy</span><code>{network.proxy_url}</code></div>}</div>{networkTargets.length > 0 && <div className="contract-targets">{networkTargets.map(([target, enabled]) => <span className={enabled ? 'enabled' : 'disabled'} key={target}>{humanizeKey(target)} · {enabled ? 'on' : 'off'}</span>)}</div>}</div>
    </section>

    <footer className="benchmark-contract-footer">
      <span>污染审计默认：<strong>{contract.contamination_audit_default ? '开启' : '关闭'}</strong></span>
      <div><DetailsDisclosure value={{ schema_version: spec.schema_version, id: spec.id, name: spec.name, category: spec.category }} label="展开 BenchmarkSpec JSON" /><DetailsDisclosure value={contract} label="展开原始运行定义 JSON" /></div>
    </footer>
  </section>
}

function humanizeKey(value: string) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function providerLabel(provider: string) {
  if (provider === 'modelscope') return 'ModelScope'
  if (provider === 'huggingface') return 'HuggingFace'
  if (provider === 'github') return 'GitHub'
  return provider
}

function contractValue(value: unknown) {
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (value === undefined || value === null || value === '') return '—'
  if (Array.isArray(value)) return value.join(', ') || '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}


function ProviderSelect({ value, sources, onChange }: { value: string; sources: Json; onChange: (value: string) => void }) {
  const providers = Object.entries(sources || {}).filter(([, items]) => Array.isArray(items) && items.length).map(([provider]) => provider)
  return <select value={value} onChange={(event) => onChange(event.target.value)}><option value="auto">auto（按注册优先级回退）</option>{providers.map((provider) => <option key={provider} value={provider}>{provider === 'modelscope' ? 'ModelScope' : provider === 'huggingface' ? 'HuggingFace' : provider === 'github' ? 'GitHub' : provider}</option>)}</select>
}

function SourceList({ sources }: { sources: Json }) {
  const rows = Object.entries(sources || {}).flatMap(([provider, items]) => (items as Array<Json | string>).map((item) => `${provider === 'modelscope' ? 'MS' : provider === 'huggingface' ? 'HF' : provider === 'github' ? 'GH' : provider} · ${typeof item === 'string' ? item : item.id || item.repo_id || 'registered source'}`))
  return <div className="source-list">{rows.length ? rows.map((row) => <small key={row} title={row}>{row}</small>) : <small>无已确认平台源，只能本地导入</small>}</div>
}

function acquisitionLabel(item: Json) {
  if (item.acquisition?.source === 'registered_scan' || String(item.id || '').includes('-scan-')) return '目录扫描注册'
  return item.acquisition?.mode === 'download' ? '注册渠道下载' : '本地导入'
}

function apiFormForSpec(spec?: Json): Json {
  return { provider: spec?.provider || '', api_spec_id: spec?.id || '', instance_id: spec ? `${spec.id}-instance` : '', base_url: spec?.default_base_url || '', auth_mode: spec?.auth_modes?.[0] || 'env', api_key_env: spec?.default_api_key_env || '', api_key: '', trust_env: spec?.trust_env !== false, shared: Boolean(spec?.shared), request_limits: spec?.request_limits || {} }
}

function JobProgress({ job }: { job?: Json }) {
  if (!job) return <span className="muted">—</span>
  const status = job.state?.status || job.status || 'running'
  return <div className="job-progress"><div><span style={{ width: `${job.percent || 0}%` }} /></div><small>{job.percent || 0}% · {status}</small><em title={job.message}>{job.message || 'starting'}</em></div>
}

function Status({ value }: { value: string }) {
  const ok = value === 'ready' || value === 'present' || value === 'configured'
  const pending = value === 'unchecked' || value === 'preparing' || value === 'raw_only'
  return <span className={`status-pill ${ok ? 'ready' : value}`}>{ok ? <CheckCircle2 size={13} /> : pending ? <RefreshCw size={13} /> : <XCircle size={13} />}{value}</span>
}

function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}><div className="modal-dialog resource-modal" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(event) => event.stopPropagation()}><div className="panel-heading"><h2>{title}</h2><button className="icon-button" title="关闭" onClick={onClose}>×</button></div>{children}</div></div>
}
