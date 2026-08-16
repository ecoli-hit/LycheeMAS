import { useMemo, useState } from 'react'
import { Check, CircleAlert, Clipboard, ClipboardCheck, Cpu, FolderCog, Network, RefreshCw, Terminal } from 'lucide-react'
import { api } from './api'
import type { Bootstrap, Json } from './types'

interface Props {
  bootstrap: Bootstrap
  activeEnvironment: Json
  onUse: (environment: Json) => void
  notify: (message: string) => void
}

const shellQuote = (value: string) => `'${String(value).replaceAll("'", `'"'"'`)}'`

export default function EnvironmentView({ bootstrap, activeEnvironment, onUse, notify }: Props) {
  const environments = bootstrap.environments.environments as Json[]
  const [selectedId, setSelectedId] = useState(activeEnvironment.id || 'readme-default')
  const initial = environments.find((item) => item.id === selectedId) || activeEnvironment
  const [environment, setEnvironment] = useState<Json>({ ...initial })
  const [result, setResult] = useState<Json | null>(null)
  const [busy, setBusy] = useState(false)
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>(['eval'])
  const [showNetwork, setShowNetwork] = useState(false)
  const [proxy, setProxy] = useState<Json>({ enabled: false, http_proxy: '', https_proxy: '', all_proxy: '', no_proxy: '127.0.0.1,localhost' })
  const [mirrors, setMirrors] = useState<Json>({ enabled: false, pip_index_url: 'https://pypi.tuna.tsinghua.edu.cn/simple', pip_extra_index_url: '', conda_channels: 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge,defaults' })
  const isActive = activeEnvironment.python === environment.python
  const packageGroups = useMemo(() => Object.entries(result?.groups || {}), [result])

  const installationCommand = useMemo(() => {
    const profiles = (bootstrap.installation_profiles || []).filter((profile: Json) => selectedProfiles.includes(profile.id))
    const extras = Array.from(new Set(profiles.flatMap((profile: Json) => profile.extras || [])))
    const target = extras.length ? `.[${extras.join(',')}]` : '.'
    const lines = [`cd ${shellQuote(bootstrap.paths.repo_root)}`]
    if (proxy.enabled) {
      for (const [field, name] of [['http_proxy', 'HTTP_PROXY'], ['https_proxy', 'HTTPS_PROXY'], ['all_proxy', 'ALL_PROXY'], ['no_proxy', 'NO_PROXY']]) {
        if (proxy[field]) lines.push(`export ${name}=${shellQuote(proxy[field])}`)
      }
    }
    if (mirrors.enabled) {
      if (mirrors.pip_index_url) {
        lines.push(`export PIP_INDEX_URL=${shellQuote(mirrors.pip_index_url)}`)
        lines.push(`export UV_DEFAULT_INDEX=${shellQuote(mirrors.pip_index_url)}`)
      }
      if (mirrors.pip_extra_index_url) lines.push(`export PIP_EXTRA_INDEX_URL=${shellQuote(mirrors.pip_extra_index_url)}`)
    }
    if (environment.is_default && !environment.exists) {
      if (environment.conda_sh) lines.push(`source ${shellQuote(environment.conda_sh)}`)
      const channels = mirrors.enabled
        ? String(mirrors.conda_channels || '').split(',').map((item) => item.trim()).filter(Boolean).map((item) => ` -c ${shellQuote(item)}`).join('')
        : ''
      lines.push(`conda create -n ${shellQuote(environment.conda_env || 'LycheeMAS')} python=3.12 -y${channels}`)
      lines.push(`conda activate ${shellQuote(environment.conda_env || 'LycheeMAS')}`)
      lines.push('python -m pip install uv')
      lines.push('uv venv .venv --python 3.12 --seed')
      lines.push('source .venv/bin/activate')
      lines.push(`uv pip install -e ${shellQuote(target)}`)
      return lines.join('\n')
    }
    if (environment.conda_sh && environment.conda_env) {
      lines.push(`source ${shellQuote(environment.conda_sh)}`)
      lines.push(`conda activate ${shellQuote(environment.conda_env)}`)
    }
    lines.push(`${shellQuote(environment.python || 'python')} -m pip install -e ${shellQuote(target)}`)
    return lines.join('\n')
  }, [bootstrap.installation_profiles, bootstrap.paths.repo_root, environment, mirrors, proxy, selectedProfiles])

  const select = (id: string) => {
    setSelectedId(id)
    const item = environments.find((row) => row.id === id)
    if (item) setEnvironment({ ...item })
    setResult(null)
  }

  const check = async () => {
    setBusy(true)
    try {
      const value = await api.checkEnvironment({ environment, raw_root: bootstrap.paths.raw_root, prepared_root: bootstrap.paths.prepared_root })
      setResult(value)
      notify(value.status === 'ready' ? '环境基础检查通过' : '环境检查完成，请查看缺失项')
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }

  const copyCommand = async () => {
    await navigator.clipboard.writeText(installationCommand)
    notify('安装命令已复制')
  }

  return <section className="environment-view">
    <div className="section-toolbar">
      <div><h1>运行环境</h1><span>默认遵循 README：Conda LycheeMAS + 项目 .venv</span></div>
      <div className="toolbar-actions"><button className={`secondary ${proxy.enabled || mirrors.enabled ? 'enabled' : ''}`} onClick={() => setShowNetwork(true)}><Network size={15} />下载网络</button><button className="secondary" disabled={busy} onClick={check}><RefreshCw size={15} />检测所选环境</button></div>
    </div>

    <div className="environment-layout">
      <aside className="environment-list">
        <div className="panel-heading"><h2>可用环境</h2></div>
        {environments.map((item) => <button key={item.id} className={selectedId === item.id ? 'selected' : ''} onClick={() => select(item.id)}><span className={`environment-dot ${item.exists ? 'exists' : 'missing'}`} /><span><strong>{item.name}</strong><small>{item.python}</small></span>{item.is_default && <em>默认</em>}</button>)}
        <button className={selectedId === 'custom' ? 'selected' : ''} onClick={() => { setSelectedId('custom'); setEnvironment({ id: 'custom', name: 'Custom environment', python: '', conda_env: '', conda_sh: '' }); setResult(null) }}><span className="environment-dot" /><span><strong>其他环境</strong><small>指定 Python executable</small></span></button>
      </aside>

      <div className="environment-detail">
        <div className="environment-form">
          <label><span>环境名称</span><input value={environment.name || ''} onChange={(event) => setEnvironment({ ...environment, name: event.target.value })} /></label>
          <label><span>Python executable</span><div className="input-with-icon"><Terminal size={15} /><input value={environment.python || ''} onChange={(event) => setEnvironment({ ...environment, python: event.target.value })} /></div></label>
          <div className="two-fields"><label><span>Conda environment</span><input value={environment.conda_env || ''} onChange={(event) => setEnvironment({ ...environment, conda_env: event.target.value })} /></label><label><span>Conda initialization</span><input value={environment.conda_sh || ''} onChange={(event) => setEnvironment({ ...environment, conda_sh: event.target.value })} /></label></div>
          <div className="environment-actions"><button className="primary" onClick={() => { onUse(environment); notify('已设为实验和资源任务环境') }}><Check size={15} />{isActive ? '当前实验环境' : '设为实验环境'}</button><span>{environment.is_default ? 'README 标准环境' : '自定义/兼容环境'}</span></div>
        </div>

        {!environment.exists && environment.is_default && !result && <div className="setup-panel"><CircleAlert size={18} /><div><strong>README 默认环境尚未创建</strong><span>下方命令会同时创建 Conda 环境和项目 .venv。</span></div></div>}

        <div className="environment-install-panel">
          <div className="panel-heading"><h2>按模块生成安装命令</h2><span>Studio 不代执行环境修改</span></div>
          <div className="install-profile-grid">{(bootstrap.installation_profiles || []).map((profile: Json) => <label key={profile.id} className={`install-profile ${selectedProfiles.includes(profile.id) ? 'selected' : ''}`}><input type="checkbox" checked={selectedProfiles.includes(profile.id)} onChange={(event) => setSelectedProfiles((current) => event.target.checked ? [...current, profile.id] : current.filter((item) => item !== profile.id))} /><span><strong>{profile.label}</strong><small>{profile.description}</small><code>{profile.extras.join(', ')}</code></span></label>)}</div>
          <div className="command-preview environment-command"><div className="panel-heading"><h2>完整安装命令</h2><button className="icon-button" title="复制安装命令" disabled={!selectedProfiles.length} onClick={copyCommand}><Clipboard size={15} /></button></div><pre><code>{selectedProfiles.length ? installationCommand : '# 请选择至少一个安装模块'}</code></pre></div>
        </div>

        {result && <><div className="environment-summary"><div><ClipboardCheck size={18} /><span>总体状态</span><strong className={`check-${result.status}`}>{result.status}</strong></div><div><Terminal size={18} /><span>Python</span><strong>{result.python_version || '—'}</strong></div><div><Cpu size={18} /><span>CUDA</span><strong>{result.cuda?.available ? `${result.cuda.device_count} GPU` : 'unavailable'}</strong></div><div><FolderCog size={18} /><span>Prefix</span><strong title={result.prefix}>{result.prefix || '—'}</strong></div></div><div className="check-grid"><section><div className="panel-heading"><h2>基础功能</h2></div>{(result.checks || []).map((item: Json) => <CheckRow key={item.id} item={item} />)}</section>{packageGroups.map(([name, items]) => <section key={name}><div className="panel-heading"><h2>{name} packages</h2></div>{(items as Json[]).map((item) => <CheckRow key={item.name} item={{ ...item, status: item.available ? 'pass' : 'fail', detail: item.name }} />)}</section>)}<section><div className="panel-heading"><h2>外部工具</h2></div>{(result.external || []).map((item: Json) => <CheckRow key={item.name} item={{ ...item, status: item.available ? 'pass' : 'warn', detail: item.detail || item.path || item.name }} />)}</section></div></>}
      </div>
    </div>

    {showNetwork && <div className="modal-backdrop" role="presentation" onMouseDown={() => setShowNetwork(false)}><div className="modal-dialog environment-network-dialog" role="dialog" aria-modal="true" aria-label="下载网络" onMouseDown={(event) => event.stopPropagation()}><div className="panel-heading"><h2>下载网络</h2><button className="icon-button" title="关闭" onClick={() => setShowNetwork(false)}>×</button></div><section><label className="check-field"><input type="checkbox" checked={proxy.enabled} onChange={(event) => setProxy({ ...proxy, enabled: event.target.checked })} /><span>启用下载代理</span></label><div className="two-fields"><label><span>HTTP proxy</span><input value={proxy.http_proxy} onChange={(event) => setProxy({ ...proxy, http_proxy: event.target.value })} placeholder="http://127.0.0.1:7890" /></label><label><span>HTTPS proxy</span><input value={proxy.https_proxy} onChange={(event) => setProxy({ ...proxy, https_proxy: event.target.value })} placeholder="http://127.0.0.1:7890" /></label></div><div className="two-fields"><label><span>ALL proxy</span><input value={proxy.all_proxy} onChange={(event) => setProxy({ ...proxy, all_proxy: event.target.value })} placeholder="socks5://127.0.0.1:7890" /></label><label><span>NO proxy</span><input value={proxy.no_proxy} onChange={(event) => setProxy({ ...proxy, no_proxy: event.target.value })} /></label></div></section><section><label className="check-field"><input type="checkbox" checked={mirrors.enabled} onChange={(event) => setMirrors({ ...mirrors, enabled: event.target.checked })} /><span>启用下载镜像</span></label><label><span>Python package index</span><input value={mirrors.pip_index_url} onChange={(event) => setMirrors({ ...mirrors, pip_index_url: event.target.value })} /></label><label><span>Python extra index</span><input value={mirrors.pip_extra_index_url} onChange={(event) => setMirrors({ ...mirrors, pip_extra_index_url: event.target.value })} placeholder="可选" /></label><label><span>Conda channels（逗号分隔）</span><input value={mirrors.conda_channels} onChange={(event) => setMirrors({ ...mirrors, conda_channels: event.target.value })} /></label></section><div className="modal-actions"><button className="primary" onClick={() => setShowNetwork(false)}>完成</button></div></div></div>}
  </section>
}

function CheckRow({ item }: { item: Json }) {
  const status = item.status || (item.available ? 'pass' : 'fail')
  return <div className="check-row"><span className={`check-mark ${status}`} /> <strong>{item.id || item.name}</strong><small title={item.detail}>{item.detail || '—'}</small></div>
}
