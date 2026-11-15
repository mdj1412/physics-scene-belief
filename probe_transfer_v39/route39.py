"""Create observation-only forecasts, then evaluate paired losses by independent catalog."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from forecast39 import ROOT, design_matrix


def make_masks(out,src,cells,coef):
    predicted=np.empty((len(cells),12,2,2,3,8))
    # Only access these two sensor-derived arrays in the npz container.
    # We commit every forecast to disk before decoding any Brier or label.
    for ci,cell in enumerate(cells):
        for k in range(12):
            for b in range(2):
                with np.load(src/cell/f'catalog_{k:02d}'/f'block_{b:02d}.npz') as f:
                    predicted[ci,k,b]=design_matrix(f['z'][:,None,:],f['weights'])@coef
    np.save(out,predicted)


def score(y,p):
    win=p<0
    # Reduce each independent catalog to means of numerators and denominators.
    return np.stack([win.mean(axis=(-4,-3,-2,-1)),(win*y).mean(axis=(-4,-3,-2,-1)),
                     ((~win)*y).mean(axis=(-4,-3,-2,-1)),y.mean(axis=(-4,-3,-2,-1)),
                     ((y-p)**2).mean(axis=(-4,-3,-2,-1)),p.mean(axis=(-4,-3,-2,-1))],axis=-1)


def estimates(s):
    rate,win,lose,total,sqerr,pred=s.T
    condwin,condloss=win/rate,lose/(1-rate)
    return np.stack([rate,condwin,condloss,condwin-condloss,win,-lose,total,np.sqrt(sqerr),pred],axis=-1)


def main():
    src=ROOT/'results'
    meta=json.loads((src/'completion.json').read_text())
    assert meta['blocks']==96 and meta['catalogs']==12 and meta['blocks_per_catalog']==2
    preds=json.loads((ROOT/'forecast39.json').read_text())
    cells=meta['cells']
    assert list(preds['cells'])==cells
    path=ROOT/'blind_scores39.npy'
    make_masks(path,src,cells,np.array(preds['observable']['coefficients']))
    pred=np.load(path)
    y=np.empty_like(pred)
    for ci,cell in enumerate(cells):
        for k in range(12):
            for b in range(2):
                with np.load(src/cell/f'catalog_{k:02d}'/f'block_{b:02d}.npz') as a:
                    np.testing.assert_array_equal(a['methods'],['fixed','optimizer'])
                    risk=a['risk']
                    y[ci,k,b]=risk[:,:,1]-risk[:,:,0]
    vals=score(y,pred)
    cols=['optimizer_rate','optimizer_minus_fixed_predwin','optimizer_minus_fixed_predloss',
          'predwin_minus_predloss','route_minus_fixed','route_minus_optimizer',
          'optimizer_minus_fixed','rmse','mean_predicted_delta']
    rng=np.random.default_rng(393040)
    ids=rng.integers(12,size=(20000,len(cells),12))
    boot=vals[np.arange(len(cells))[None,:,None],ids]
    p=estimates(vals.mean((0,1))[None])[0]
    rep=estimates(boot.mean((1,2)))
    # Joint interval for route - each conventional alternative.
    sd=rep[:,4:6].std(0,ddof=1)
    width=np.quantile((abs(rep[:,4:6]-p[4:6])/sd).max(1),.95)*sd
    table={k:[v,*np.quantile(rep[:,i],[.025,.975])] for i,(k,v) in enumerate(zip(cols,p))}
    strata={c:dict(zip(cols,estimates(vals[i].mean(0)[None])[0])) for i,c in enumerate(cells)}
    # The two forecast groups need not contain the same mixture of shapes.
    # Standardize the group contrast by giving every planned cell a quarter weight.
    cell_gap=np.stack([estimates(boot[:,i].mean(axis=1))[:,3] for i in range(len(cells))],axis=1)
    gap_point=np.mean([row[cols[3]] for row in strata.values()])
    ans=dict(method='frozen z,w-only ridge; unweighted cell mean with catalog bootstrap, no p90 measured for routed policy',
             coefficient_source_hash=hashlib.sha256((ROOT/'forecast39.json').read_bytes()).hexdigest(),
             forecasts_hash=hashlib.sha256(path.read_bytes()).hexdigest(),n_criterion_rows=pred.size,
             summaries=table,route_contrasts_simultaneous=np.stack((p[4:6]-width,p[4:6]+width),1),
             equal_cell_standardized_group_gap=[gap_point,*np.quantile(cell_gap.mean(1),[.025,.975])],
             strata=strata,pooled_descriptive_spearman=float(spearmanr(y.ravel(),pred.ravel()).statistic))
    (ROOT/'route39.json').write_text(json.dumps(ans,indent=2,allow_nan=False,default=lambda a:a.tolist())+'\n')
    def f(k):
        v,l,h=table[k]
        return f'{v:+.6f} [{l:+.6f}, {h:+.6f}]'
    text=['# 네 번째 관측 후 어느 환경에서 optimizer가 유리했나','',
          '예측식은 v38에서만 학습한 z, w 기반 ridge regression이며 v39의 test label로 보정하지 않았다. 원자료의 z와 w만 읽어 전 환경의 점수를 먼저 저장한 후 두 방법의 실제 paired Brier 차이를 읽었다. 이 순서는 동일한 실행으로 얻은 sensor record를 사용한 blind replay다. 새로운 물체의 후속 충돌 정답은 보조 정책의 입력에 없다. [사전 예측](PREDICTIONS_KO.md) · [예측 결과와 hash](route39.json)','',
          f'Optimizer 사용을 예측한 비율은 {p[0]*100:.1f}%였다. 이 집단의 실제 optimizer − fixed Brier는 {f(cols[1])}이고, fixed를 선택한 집단에서는 {f(cols[2])}였다. 두 집단 간 차이는 {f(cols[3])}였다. Interval은 4개 고정 cell 내부에서 catalog를 재표집한 pointwise 95% 구간이다. 이 비교는 보정 선택에 사용할 수 있는 정보의 정렬 능력을 평가하며 두 집단 간 causal effect를 추정하지 않는다.','',
          '| 비교 | Brier 차이 | 동시 95% interval |','|---|---:|---|']
    for j,method in enumerate(('Routing − fixed','Routing − optimizer')):
        l,h=ans['route_contrasts_simultaneous'][j]
        text.append(f'| {method} | {p[4+j]:+.6f} | [{l:+.6f}, {h:+.6f}] |')
    text+=['','| 형상과 law | Optimizer 선택률 | 예측 win 집단 실제 Δ | 예측 loss 집단 실제 Δ | Routing − fixed | Routing − optimizer |',
           '|---|---:|---:|---:|---:|---:|']
    for c,row in strata.items():
        text.append(f'| {c} | {row[cols[0]]*100:.1f}% | {row[cols[1]]:+.6f} | {row[cols[2]]:+.6f} | {row[cols[4]]:+.6f} | {row[cols[5]]:+.6f} |')
    ge,gl,gh=ans['equal_cell_standardized_group_gap']
    text+=['',f'두 집단의 형상과 law 구성비가 달랐다. 두 shelf에서는 합산 결과와 이득의 순서가 반대이고 두 ridge에서는 같은 방향이었다. 결과 이후 이를 점검하기 위해 각 cell 안에서 먼저 구한 집단 차이를 각 25%로 표준화하면 {ge:+.6f} [{gl:+.6f}, {gh:+.6f}]였다. 이 보조 분석은 모든 형상에 공통인 환경별 판별력을 확정하지 못한다.','',f'환경별 Δ의 RMSE는 {p[7]:.6f}, 전체 descriptive Spearman correlation은 {ans["pooled_descriptive_spearman"]:.3f}이었다. 평가 단위는 {pred.size:,}개 environment-regime-orientation 행이며 서로 노이즈와 물체를 공유하는 행들이 포함된다. 독립 단위는 48개 cell-specific catalog다. Router의 GPU p90은 별도로 측정하지 않았다.']
    (ROOT/'ROUTING_KO.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    main()
