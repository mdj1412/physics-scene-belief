"""Post-primary sensitivity: evaluate identical frozen forecasts under three discretizations."""
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
import forecast_eval44 as f
ROOT=f.ROOT
ARMS=('primary','halfdt','halfmesh')
DIRS=('labels44','labels44_full_halfdt','labels44_full_halfmesh')


def ci(x,num):
    mean=x.mean(0)
    err=t.ppf(1-.025/num,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
    return np.stack([mean,mean-err,mean+err],axis=-1)


def main():
    p=ROOT/'forecast44.json'
    forecasts=json.loads(p.read_text())['rows']
    shape=(3,12,2,2,3,2,2,3,2)
    risks=np.zeros(shape)
    flips=np.zeros((3,12,2),int)
    valid_count=np.zeros(3,int)
    for panel,i,h,v,surf,law in product(range(12),range(2),range(2),range(3),('inlay','woven'),('rational','gaussian_rate')):
        var=('base','hollow','bulge')[v]
        path=Path(f'panel_{panel:02d}')/f'{f.SHAPES[i]}_{surf}_{law}_{var}_h{f.H[h]:.0f}.npz'
        with np.load(ROOT/'results'/path) as a:
            pred=a['pred'][:,0,:,:8]
            eq=pred.mean(-1); eq2=(pred**2).mean(-1)
        for arm,folder in enumerate(DIRS):
            with np.load(ROOT/folder/path) as a:
                y=a['labels'];valid=a['valid']
            if arm==0:
                ref=y.copy();refok=valid.copy()
            common=refok&valid
            flips[arm,panel,0]+=np.sum((y!=ref)&common)
            flips[arm,panel,1]+=common.sum()
            valid_count[arm]+=valid.sum()
            assert np.all(valid.sum(axis=(-1,-2))>0)
            vals=eq2[:,None,None,None]-2*eq[:,None,None,None]*y+y
            risks[arm,panel,i,h,v]+=np.moveaxis((vals*valid).sum(axis=(-2,-1))/valid.sum(axis=(-2,-1)),0,-1)/4
    out=dict(arms=ARMS,scopes=f.SCOPES,models=['original','privileged'],summaries={},changed_cells=[],
             flip_fraction_paired_ci=ci(flips[:,:,0].T/flips[:,:,1].T,2).tolist(),flip_counts=flips.sum(1).tolist(),
             total_valid=valid_count.tolist(),cell_summary_note='original and privileged zero correction only')
    rel=[]
    for scope,sl in (('all',slice(None)),('deformed',slice(1,None))):
        risk=risks[:,:,:,:,sl].mean(axis=4)
        D=risk[...,1]-risk[...,0]
        source=[r for r in forecasts if r['scope']==scope]
        src=np.array([[r['pred'],r['carry']] for r in source]).reshape(2,2,2,2,3,2)
        s={}
        out['summaries'][scope]=s
        for arm,tag in enumerate(ARMS):
            fine=f.calc(D[arm],src[...,0],src[...,1])
            design_mean=D[arm].mean(0)
            changed=np.flatnonzero(np.sign(src[...,0].ravel())!=np.sign(src[...,1].ravel()))
            comp=np.where(np.sign(D[0].mean(0))!=np.sign(design_mean))
            signs=np.sign(design_mean)
            both_sig=(np.array(fine['sim_lo'])*np.array(fine['sim_hi'])>0).reshape(signs.shape)
            if arm==0: old_sig=both_sig
            per=[]
            for idx in changed:
                r=source[idx]
                per.append(dict(**r,observed=fine['observed'][idx],sim95=[fine['sim_lo'][idx],fine['sim_hi'][idx]]))
            if scope=='all':out['changed_cells'].append(per)
            paired_risk = risk[arm].mean(axis=(2,3,4))  # n,sh,d,model
            strata = np.stack((paired_risk[:,:,1],paired_risk[:,:,[0,2]].mean(axis=2)),axis=2)
            diff=strata[...,1]-strata[...,0]
            s[tag]=dict(score=fine,
              sign_changed_from_primary=len(comp[0]),primary_and_new_resolved_changed=int((old_sig[comp]&both_sig[comp]).sum()),
              stable_primary_and_new_48sim_signs=int((old_sig&both_sig&(np.sign(D[0].mean(0))==signs)).sum()),
              shape_strata_model_loss=strata.mean(axis=0).tolist(),
              shape_strata_D_4sim_ci=ci(diff,4).tolist(),
              changed_design_rows=[source[np.ravel_multi_index(idx,signs.shape)] for idx in zip(*comp)])
        if scope=='all':
            s0=s['primary']['score']
            ref=json.loads((ROOT/'forecast_eval44.json').read_text())
            real=np.array([r['observed'] for r in ref['cells'] if r['scope']==scope])
            np.testing.assert_allclose(s0['observed'],real,atol=2e-12,rtol=0)
            rel=sorted([x for x in product(*([range(2)]*4),range(3)) if len({int(np.sign(D[arm].mean(0)[x])) for arm in range(3)})>1])
    out['affected_cells_all']=[list(x) for x in rel]
    (ROOT/'numerical_forecast44.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# 전체 충돌을 다시 생성한 수치 민감도 검증','',
           'Frozen forecast를 바꾸지 않고 12개 catalog 전부를 동일 물체·위치·장치에서 다시 측정했다. Primary는 dt 20 μs와 mesh 간격 0.10 mm, halfdt는 dt만 10 μs, halfmesh는 mesh만 0.05 mm다. 확률과 GPU 연산은 세 조건에서 동일하다. 독립 검증 표본 수는 12개 catalog이며 반복된 충돌은 paired control이다. 두 수치 설정은 결과를 본 뒤 추가한 민감도 대조다.',
           '', '| 설정 | 전체 label 변화율 vs primary | All 부호 적중(support / carry / 항상 손해) | Deformed 부호 적중 | All에서 동시 95%로 확정한 칸과 그 안의 support 적중 | Primary 대비 all 부호 변경 |',
           '|---|---:|---:|---:|---:|---:|']
    for a,tag in enumerate(ARMS):
        s=out['summaries']['all'][tag];x=s['score'];z=out['summaries']['deformed'][tag]['score']; n,den=out['flip_counts'][a]
        lines.append(f'| {tag} | {n:,}/{den:,} ({100*n/den:.2f}%) | {" / ".join(map(str,x["correct"]))} | {" / ".join(map(str,z["correct"]))} | {x["confidence_resolved"]} · {x["resolved_correct"][0]} 적중 | {s["sign_changed_from_primary"]}/48 |')
    lines+=['','All의 Brier는 base·hollow·bulge와 모든 조합을 같은 가중으로 평균했다. Nominal은 위치가 정확할 때, shifted는 ±0.5 mm의 균등 평균이다. 마지막 열은 같은 설정에서 네 shape×stratum을 함께 보정한 paired 95% 구간이다.','',
           '| 설정 | Shape | 위치 | Original Brier | Privileged Brier | Privileged−original [동시 95%] |','|---|---|---|---:|---:|---:|']
    for tag,sh,dd in product(ARMS,range(2),range(2)):
        s=out['summaries']['all'][tag]
        op,pp=s['shape_strata_model_loss'][sh][dd];d,lo,hi=s['shape_strata_D_4sim_ci'][sh][dd]
        lines.append(f'| {tag} | {f.SHAPES[sh]} | {("nominal","shifted")[dd]} | {op:.6f} | {pp:.6f} | {d:+.6f} [{lo:+.6f}, {hi:+.6f}] |')
    lines+=['','V43 carry와 사전 support forecast가 다르게 예측한 5개 칸의 관측된 projection−original Brier다. 세 측정의 확률 입력은 동일하다.','',
            '| Shape/h, R/L, offset (mm) | Carry 예상 | Support 예상 | Primary | Halfdt | Halfmesh |','|---|---:|---:|---:|---:|---:|']
    for trio in zip(*out['changed_cells']):
        a=trio[0]
        lines.append(f'| {a["shape"]}/{a["height_mm"]:g}, {a["r_mm"]:g}/{a["span_mm"]:g}, {a["offset_mm"]:+g} | {a["carry"]:+.6f} | {a["pred"]:+.6f} | '+' | '.join(f'{r["observed"]:+.6f}' for r in trio)+' |')
    lines+=['','각 설정의 48개 칸별 D와 동시 구간, sign agreement·MAE와 부호가 달라진 셀은 [기계 판독 결과](numerical_forecast44.json)에 보관했다. [분석 코드](numerical_forecast44.py) · [사전 예측](forecast44.json)']
    (ROOT/'NUMERICAL_FORECAST_KO.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
