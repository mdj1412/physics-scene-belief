"""Decompose speed-induced gate decisions by the matched contact's paired slip state."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
GROUPS=('always_stick','stick_to_slip','slip_to_stick','always_slip')
METRICS=('mass','changed_gate_mass','observed_contribution','common_action_contribution','common_benefit_of_selected_correction','initial_gate_minus_shrink','fast_gate_minus_shrink')
SCOPES={'local_equal':[0,1,2],'weak_null':[1,2],'strong':[0],'weak':[1],'null':[2],'global_null':[3]}
def main():
    cfg=json.loads((ROOT/'design37.json').read_text())
    panel=np.zeros((4,8,4,2,4,len(METRICS)))
    for c,cell in enumerate(cfg['cells']):
        for b in range(8):
            directory=ROOT/'results'/cell/f'block_{b:03d}'
            for i,r in enumerate(cfg['regimes']):
                signs=['plus'] if i==3 else ['plus','minus']
                for s in signs:
                    with np.load(directory/(r+'_'+s+'.npz')) as d:
                        risk=d['risk']; slip=~d['matched_stick'].astype(bool)
                        g=d['accept4'].astype(float)
                        for k,j in enumerate((1,2)):
                            code=2*slip[0].astype(int)+slip[j].astype(int)
                            dd=(risk[j,3]-risk[j,2])-(risk[0,3]-risk[0,2])
                            dd_shared=(g[j]-g[0])*(risk[0,1]-risk[0,0])
                            for v in range(4):
                                valid=code==v
                                cols=np.stack([valid,valid*(g[j]!=g[0]),valid*dd,valid*dd_shared,
                                    valid*(risk[0,1]-risk[0,0]), valid*(risk[0,3]-risk[0,2]),
                                    valid*(risk[j,3]-risk[j,2])],-1)
                                panel[c,b,i,k,v]+=cols.mean(0)/len(signs)
                            assert np.all(np.isin(code,range(4)))
    assert np.allclose(panel[...,0].sum(-1),1.)
    rng=np.random.default_rng(370171)
    ids=(2*rng.integers(4,size=(12000,4,4,2))+np.arange(2)).reshape(-1,4,8)
    out=dict(group=GROUPS,metrics=METRICS,contrasts=['0.75−0.50','1.10−0.50'],scopes={})
    for key,idx in SCOPES.items():
        x=np.take(panel,idx,axis=2).mean(2)
        mean=x.mean((0,1)); boots=x[np.arange(4)[None,:,None],ids].mean((1,2))
        ci=np.moveaxis(np.quantile(boots,[.025,.975],axis=0),0,-1)
        rows=[]
        for k,sp in enumerate((.75,1.10)):
            per={}
            for v,gn in enumerate(GROUPS):
                mass,flip,actual,paired,davg,initial,fast=mean[k,v]
                per[gn]=dict(share=mass,gate_changed_given_group=flip/mass if mass else None,
                        actual_contribution=actual,actual_ci=ci[k,v,2].tolist(),
                        common_contribution=paired,common_ci=ci[k,v,3].tolist(),
                        common_within_group=paired/mass if mass else None,
                        mean_always_minus_original=davg/mass if mass else None,
                        initial_gate_minus_shrink=initial/mass if mass else None,
                        fast_gate_minus_shrink=fast/mass if mass else None)
            per['total']=dict(share=1.,actual_contribution=mean[k,:,2].sum(),
                    common_contribution=mean[k,:,3].sum(),
                    common_ci=np.quantile(boots[:,k,:,3].sum(-1),[.025,.975]).tolist())
            rows.append(per)
        out['scopes'][key]=rows
    (ROOT/'slip37.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    rows=['## 네 번째 물체가 실제로 slip으로 바뀌었을 때의 분해','',
          '같은 첫 세 접촉, 같은 네 번째 물체와 noise를 사용한 속도 쌍을 latent contact state로 분할했다. 각 행에서 실현 비중과 그 환경 중 gate가 달라진 비율을 보인다. 기여는 전체 local 평균 Δ Brier(빠른 속도 − +0.50)에 대한 각 행의 가중합 항이다. 원래 비교는 각 속도의 gate − 고정 shrinkage 차이에 대한 속도 효과다. 공통 action 비교는 +0.50의 deployment probability·signed proposal을 고정하므로 gate 선택만 바뀐다. 음수가 빠른 속도의 개선이다.','',
          '| 속도 | +0.50에서 → 빠른 속도에서 | 비중 | 해당 그룹의 gate 변경률 | 해당 그룹의 Gate−shrink (기준 → 빠른 속도) | 실제 경로 기여 | 공통 action 기여 [95%] |',
          '|---|---|---:|---:|---:|---:|---:|']
    for speed,result in zip((.75,1.10),out['scopes']['local_equal']):
        for name in GROUPS:
            v=result[name];lo,hi=v['common_ci']
            f=v['gate_changed_given_group'];flip='—' if f is None else f'{f*100:.1f}%'
            sub='—' if not v['share'] else f'{v["initial_gate_minus_shrink"]:+.5f} → {v["fast_gate_minus_shrink"]:+.5f}'
            rows.append(f'| {speed:.2f} | {name} | {v["share"]*100:.1f}% | {flip} | {sub} | {v["actual_contribution"]:+.5f} | {v["common_contribution"]:+.5f} [{lo:+.5f}, {hi:+.5f}] |')
    rows+=['','처음부터 slip인 물체에서도 속도에 따라 normal impulse와 friction modulation이 달라질 수 있다. 전환 그룹 안에서는 물체의 slope와 마찰 분포가 전체와 다를 수 있다. 각 기여는 관측된 효과의 합을 설명하며 slip의 독립적인 causal mediation 비율은 추정하지 않는다. [세부 수치](slip37.json) · [사전 설계](SLIP_PLAN_KO.md)']
    (ROOT/'SLIP_KO.md').write_text('\n'.join(rows)+'\n')
if __name__=='__main__':main()
