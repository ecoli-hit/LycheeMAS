import { useEffect, useState } from 'react'
import { Activity, Boxes, CircleDollarSign, Database, FlaskConical, Gauge, Play, RefreshCw, ServerCog, Workflow } from 'lucide-react'
import { api } from './api'
import CostView from './CostView'
import EnvironmentView from './EnvironmentView'
import ExperimentView from './ExperimentView'
import ResourcesView from './ResourcesView'
import RunsView from './RunsView'
import StudioView from './StudioView'
import type { Bootstrap, Json } from './types'

type Tab = 'environment' | 'resources' | 'cost' | 'team' | 'deployment' | 'experiment' | 'runs'

export default function App() {
  const [tab, setTab] = useState<Tab>('environment')
  const [data, setData] = useState<Bootstrap | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState<{ message: string, tone: 'success' | 'error' } | null>(null)
  const [activeEnvironment, setActiveEnvironment] = useState<Json | null>(null)

  const reload = async () => {
    setError('')
    try {
      const value = await api.bootstrap()
      setData(value)
      setActiveEnvironment((current) => current || value.environments.default)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  useEffect(() => void reload(), [])

  const notify = (message: string, tone: 'success' | 'error' = 'success') => {
    setNotice({ message, tone })
    window.setTimeout(() => setNotice(null), 4000)
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
          <button className={tab === 'environment' ? 'active' : ''} onClick={() => setTab('environment')}>
            <Gauge size={16} />运行环境
          </button>
          <button className={tab === 'resources' ? 'active' : ''} onClick={() => setTab('resources')}>
            <Database size={16} />资源中心
          </button>
          <button className={tab === 'cost' ? 'active' : ''} onClick={() => setTab('cost')} title="Pricing">
            <CircleDollarSign size={16} />成本管理
          </button>
          <button className={tab === 'deployment' ? 'active' : ''} onClick={() => setTab('deployment')} title="Deployment">
            <ServerCog size={16} />部署管理
          </button>
          <button className={tab === 'team' ? 'active' : ''} onClick={() => setTab('team')} title="Team">
            <Boxes size={16} />团队管理
          </button>
          <button className={tab === 'experiment' ? 'active' : ''} onClick={() => setTab('experiment')} title="Experiment">
            <Workflow size={16} />实验管理
          </button>
          <button className={tab === 'runs' ? 'active' : ''} onClick={() => setTab('runs')}>
            <Play size={16} />运行记录
          </button>
        </nav>
        <button className="icon-button" title="刷新所有状态" onClick={reload}>
          <RefreshCw size={17} />
        </button>
      </header>

      <main className="app-main">
        {tab === 'environment' && <EnvironmentView bootstrap={data} activeEnvironment={activeEnvironment || data.environments.default} onUse={setActiveEnvironment} notify={notify} />}
        {tab === 'resources' && (
          <ResourcesView bootstrap={data} environment={activeEnvironment || data.environments.default} notify={notify} onRefresh={reload} />
        )}
        {tab === 'cost' && <CostView bootstrap={data} notify={notify} onRefresh={reload} />}
        {(tab === 'team' || tab === 'deployment') && (
          <StudioView
            bootstrap={data}
            environment={activeEnvironment || data.environments.default}
            notify={notify}
            workspace={tab}
          />
        )}
        {tab === 'experiment' && <ExperimentView bootstrap={data} environment={activeEnvironment || data.environments.default} notify={notify} onRefresh={reload} />}
        {tab === 'runs' && <RunsView initialRuns={data.runs} />}
      </main>
      {notice && <div className={`toast ${notice.tone}`}>{notice.message}</div>}
      {error && <div className="toast error">{error}</div>}
    </div>
  )
}
