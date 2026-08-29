import { lazy, Suspense, useEffect, useState } from 'react'
import { Activity, Boxes, CircleDollarSign, Database, FlaskConical, Gauge, Play, RefreshCw, ServerCog, Workflow } from 'lucide-react'
import { api } from './api'
import EnvironmentView from './EnvironmentView'
import type { Bootstrap, Json } from './types'

const CostView = lazy(() => import('./CostView'))
const ExperimentView = lazy(() => import('./ExperimentView'))
const ResourcesView = lazy(() => import('./ResourcesView'))
const RunsView = lazy(() => import('./RunsView'))
const StudioView = lazy(() => import('./StudioView'))

type Tab = 'environment' | 'resources' | 'cost' | 'team' | 'deployment' | 'experiment' | 'runs'

export default function App() {
  const [tab, setTab] = useState<Tab>('environment')
  const [data, setData] = useState<Bootstrap | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState<{ message: string, tone: 'success' | 'error' } | null>(null)
  const [activeEnvironment, setActiveEnvironment] = useState<Json | null>(null)
  const [hydratedTabs, setHydratedTabs] = useState<Set<Tab>>(() => new Set(['environment']))

  const reload = async () => {
    setError('')
    try {
      const core = await api.bootstrap()
      const workspace = tab === 'environment' ? {} : await api.workspace(tab)
      const value = { ...core, ...workspace }
      setData(value)
      setActiveEnvironment((current) => current || value.environments.default)
      setHydratedTabs(new Set(['environment', tab]))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  useEffect(() => void reload(), [])

  const notify = (message: string, tone: 'success' | 'error' = 'success') => {
    setNotice({ message, tone })
    window.setTimeout(() => setNotice(null), 4000)
  }

  const openWorkspace = async (next: Tab) => {
    setTab(next)
    if (hydratedTabs.has(next)) return
    setError('')
    try {
      const workspace = await api.workspace(next)
      setData((current) => current ? { ...current, ...workspace } : current)
      setHydratedTabs((current) => new Set([...current, next]))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  if (!data) {
    return (
      <main className="loading-screen">
        <Activity className="spin" size={22} />
        <span>{error || '连接 Eval Studio…'}</span>
        {error && <button onClick={reload}>重试</button>}
      </main>
    )
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <FlaskConical size={20} />
          <strong>LycheeMAS Eval</strong>
          <span>Studio</span>
        </div>
        <nav className="primary-nav" aria-label="主导航">
          <button className={tab === 'environment' ? 'active' : ''} onClick={() => void openWorkspace('environment')}>
            <Gauge size={16} />运行环境
          </button>
          <button className={tab === 'resources' ? 'active' : ''} onClick={() => void openWorkspace('resources')}>
            <Database size={16} />资源中心
          </button>
          <button className={tab === 'cost' ? 'active' : ''} onClick={() => void openWorkspace('cost')} title="Pricing">
            <CircleDollarSign size={16} />成本管理
          </button>
          <button className={tab === 'deployment' ? 'active' : ''} onClick={() => void openWorkspace('deployment')} title="Deployment">
            <ServerCog size={16} />部署管理
          </button>
          <button className={tab === 'team' ? 'active' : ''} onClick={() => void openWorkspace('team')} title="Team">
            <Boxes size={16} />团队管理
          </button>
          <button className={tab === 'experiment' ? 'active' : ''} onClick={() => void openWorkspace('experiment')} title="Experiment">
            <Workflow size={16} />实验管理
          </button>
          <button className={tab === 'runs' ? 'active' : ''} onClick={() => void openWorkspace('runs')}>
            <Play size={16} />运行记录
          </button>
        </nav>
        <button className="icon-button" title="刷新所有状态" onClick={reload}>
          <RefreshCw size={17} />
        </button>
      </header>

      <main className="app-main">
        <Suspense fallback={<div className="workspace-loading"><Activity className="spin" size={20} /><span>正在载入工作区</span></div>}>
          {!hydratedTabs.has(tab) && <div className="workspace-loading"><Activity className="spin" size={20} /><span>正在载入工作区数据</span></div>}
          {hydratedTabs.has(tab) && tab === 'environment' && <EnvironmentView bootstrap={data} activeEnvironment={activeEnvironment || data.environments.default} onUse={setActiveEnvironment} notify={notify} />}
          {hydratedTabs.has(tab) && tab === 'resources' && (
            <ResourcesView bootstrap={data} environment={activeEnvironment || data.environments.default} notify={notify} onRefresh={reload} />
          )}
          {hydratedTabs.has(tab) && tab === 'cost' && <CostView bootstrap={data} notify={notify} onRefresh={reload} />}
          {hydratedTabs.has(tab) && (tab === 'team' || tab === 'deployment') && (
            <StudioView
              bootstrap={data}
              environment={activeEnvironment || data.environments.default}
              notify={notify}
              workspace={tab}
            />
          )}
          {hydratedTabs.has(tab) && tab === 'experiment' && <ExperimentView bootstrap={data} environment={activeEnvironment || data.environments.default} notify={notify} onRefresh={reload} />}
          {hydratedTabs.has(tab) && tab === 'runs' && <RunsView initialRuns={data.runs} />}
        </Suspense>
      </main>
      {notice && <div className={`toast ${notice.tone}`}>{notice.message}</div>}
      {error && <div className="toast error">{error}</div>}
    </div>
  )
}
