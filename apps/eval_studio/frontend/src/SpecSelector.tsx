import type { Json } from './types'

interface Props {
  kind: 'DeploymentSpec' | 'TeamSpec' | 'ExperimentSpec' | 'PricingSpec' | 'BenchmarkSpec' | 'ModelSpec' | 'APISpec'
  specs: Json[]
  draftId: string
  optionLabel: (spec: Json) => string
  onLoad: (spec: Json) => void
}

export default function SpecSelector({ kind, specs, draftId, optionLabel, onLoad }: Props) {
  const saved = specs.some((item) => item.id === draftId)
  return <div className="spec-switcher">
    <label>
      <span>载入 {kind}</span>
      <select
        value={saved ? draftId : ''}
        onChange={(event) => {
          const spec = specs.find((item) => item.id === event.target.value)
          if (spec) onLoad(spec)
        }}
      >
        <option value="" disabled>选择已保存 {kind}</option>
        {specs.map((spec) => <option key={spec.id} value={spec.id}>{optionLabel(spec)}</option>)}
      </select>
    </label>
    {!saved && <div className="draft-indicator"><strong>新建草稿</strong><span>{draftId || '尚未填写 ID'} · 保存后进入载入列表</span></div>}
  </div>
}
