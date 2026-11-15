"""Condition on fourth factual features, retaining the frozen shared first-three correction."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata
from gate36 import ROOT, load_artifact, log_density, observations, posterior, expected_gain, SIGNS
CELLS=('sinusoid_atan','sinusoid_algebraic','quartic_atan','quartic_algebraic')
REGIMES=('strong','weak','null','global_null')

def marginal(artifact):
    # Pxx - Pxy Pyy^{-1} Pyx is the inverse of marginal covariance on X.
    pr=artifact['precision'][2]
    pxx, pxy, pyy=pr[:,:,:5,:5],pr[:,:,:5,5:],pr[:,:,5:,5:]
    prec=pxx-pxy@np.linalg.solve(pyy,pxy.swapaxes(-1,-2))
    return {'means':artifact['means'][2:3,:,:,:5], 'precision':prec[None],
            'log_norm':(.5*np.linalg.slogdet(prec)[1]-np.log(prec.shape[1]))[None]}

def score(art,mar,d):
    x=d['far_x']; y=d['near_y']; dirs=d['dirs']; n=len(x)
    ll=np.zeros((n,len(art['sign'])))
    for j in range(3):
        obs=observations(x[:,j],y[:,j])
        for action,sign in enumerate(SIGNS):
            mask=dirs[:,j]==sign
            if mask.any():
                ll[mask]+=log_density(art,obs[mask],action)
    w=posterior(ll)
    z=w@art['far_sign']
    np.testing.assert_allclose(z,d['z'],atol=1e-11)
    s=expected_gain(art,w,z)
    np.testing.assert_allclose(s,d['score3'],atol=1e-11)
    fourth=observations(x[:,3],np.zeros((n,2)))[:,:5]
    wx=posterior(ll+log_density(mar,fourth,0))
    sx=expected_gain(art,wx,z)
    baseline,applied=d['risk'][2:4]
    g4=d['accept4'].astype(float)
    g3=d['accept3'].astype(float)
    gx=(sx>0).astype(float)
    delta=applied-baseline
    return np.stack([(g4-gx)*delta, (gx-g3)*delta,gx,g4!=gx,
                     wx@art['near_strength'],d['near_strength4'],g4,delta],axis=-1)

def auc(strong,others):
    # Probability(area) that a randomly drawn strong case receives higher score,
    # sharing any ties equally. Order irrelevant for the point estimate.
    pos=strong.ravel(); neg=others.ravel()
    ranks=rankdata(np.r_[pos,neg],method='average')
    return (ranks[:len(pos)].mean()-(len(pos)+1)/2)/len(neg)

def main():
    a=load_artifact(); m=marginal(a)
    panel=np.zeros((4,8,4,2,8,8))
    for c,cell in enumerate(CELLS):
        for b in range(8):
            for t,reg in enumerate(REGIMES):
                for i,side in enumerate(('plus','minus')):
                    if reg=='global_null' and i:
                        panel[c,b,t,i]=panel[c,b,t,0]
                        continue
                    path=ROOT/'results'/cell/f'block_{b:03d}'/f'{reg}_{side}.npz'
                    with np.load(path) as d:
                        panel[c,b,t,i]=score(a,m,d)
    obs=panel.mean((3,4))
    rng=np.random.default_rng(360529)
    ids=(2*rng.integers(4,size=(12000,4,4,2))+np.arange(2)).reshape(-1,4,8)
    rp=obs[np.arange(4)[None,:,None],ids].mean((1,2))
    est=obs.mean((0,1))
    scopes={}
    for name,ix in [('local_equal',[0,1,2]),*[(r,[i]) for i,r in enumerate(REGIMES)]]:
        vals=est[ix,:6].mean(0)
        boot=rp[:,ix,:6].mean(1)
        scopes[name]={'means':vals.tolist(), 'pointwise95':np.quantile(boot,[.025,.975],axis=0).T.tolist()}
    # Conditional randomization across three coupled regimes within a calibration world.
    chosen=panel[:,:,:3,:,:,6]
    costs=panel[:,:,:3,:,:,7]
    randomized=chosen.mean(axis=2,keepdims=True)
    causal=((chosen-randomized)*costs).mean(axis=(2,3,4))
    replicates=causal[np.arange(4)[None,:,None],ids].mean((1,2))
    conditional=dict(delta=float(causal.mean()),pointwise95=np.quantile(replicates,[.025,.975]).tolist(),
                     identical_decision_fraction=float((chosen.min(2)==chosen.max(2)).mean()))
    ranking={}
    rank_intervals={}
    # The two-sample U statistic is a weighted sum over block pairs. Bootstrap the
    # 32 blocks jointly across compared classes, preserving their shared draws.
    flattened=(ids+8*np.arange(4)[None,:,None]).reshape(-1,32)
    counts=np.stack([(flattened==i).sum(axis=1) for i in range(32)],axis=1)/32.
    for k,name in [(4,'first3_plus_fourth_factual'),(5,'first3_plus_full_fourth')]:
        plus=panel[:,:,0,:,:,k].reshape(32,16)
        others=panel[:,:,1:3,:,:,k].reshape(32,32)
        ranking[name]=auc(plus,others)
        cmp=plus[:,None,:,None]-others[None,:,None,:]
        block_matrix=((cmp>0)+.5*(cmp==0)).mean(axis=(2,3))
        np.testing.assert_allclose(block_matrix.mean(),ranking[name],atol=1.e-12)
        estimates=np.einsum('ni,ij,nj->n',counts,block_matrix,counts)
        rank_intervals[name]=np.quantile(estimates,[.025,.975]).tolist()
    out={'measures':['gate4_minus_gate3x_brier','gate3x_minus_gate3_brier','accept3x','changed_gate4_vs_3x','mean_near_strength3x','mean_near_strength4'],
         'scopes':scopes,'auc_local_strong_vs_weak_null':ranking,'auc_pointwise95':rank_intervals,'paired_regime_assignment':conditional}
    (ROOT/'increment36.json').write_text(json.dumps(out,indent=2)+'\n')
    fmt=lambda n:f'{n:+.6f}'
    lines=['# 네 번째 factual feature를 공통으로 제공한 대조','',
           '| 조건 | Gate4 − gate3+x4 Brier [paired 95%] | Gate3+x4 − gate3 Brier [paired 95%] | Gate3+x4 적용률 | Paid velocity를 보고 결정이 바뀐 비율 |',
           '|---|---:|---:|---:|---:|']
    for name,v in scopes.items():
        vals=v['means']; ci=v['pointwise95']
        fields=[name]+[fmt(vals[i])+' ['+', '.join(map(fmt,ci[i]))+']' for i in range(2)]+[f'{100*vals[i]:.1f}%' for i in (2,3)]
        lines.append('| '+' | '.join(fields)+' |')
    lines.extend(['',f'같은 첫 세 접촉과 네 번째 factual feature에서 strong과 weak/null을 구별하는 posterior strength의 AUROC는 {ranking["first3_plus_fourth_factual"]:.3f}, 네 번째 paid velocity까지 주면 {ranking["first3_plus_full_fourth"]:.3f} [paired 95% {rank_intervals["first3_plus_full_fourth"][0]:.3f}, {rank_intervals["first3_plus_full_fourth"][1]:.3f}]였다. 세 local regime의 기존 관측들이 완전히 짝지어 있으므로 paid velocity 이전의 strong 및 weak/null score 분포는 같다. AUROC는 “보정 여부”가 아니라 source posterior strength의 순위를 평가한다. 강도 차이를 일부 읽어도 Brier 개선과 적용 여부의 순위는 달라질 수 있다.','',
                  f'세 local regime에서 같은 물체와 noise를 짝지은 상태로 gate decisions만 regime들 사이에서 무작위 permutation했을 때의 평균 loss와 실제 배정을 비교했다. 각 조건의 원래 deployment action, base probability와 실제 query label을 사용했다. 실제 − 무작위 배정 Brier는 {conditional["delta"]:+.6f} [{conditional["pointwise95"][0]:+.6f}, {conditional["pointwise95"][1]:+.6f}]였다. 세 regime에서 gate decision이 모두 같은 경우는 {100*conditional["identical_decision_fraction"]:.1f}%였다. 이 대조는 환경별 평균 적용률과 네 번째 접촉에서 드러나는 공통 물체 성질을 고정한다. 여기에서 음수이면 강도 차이 때문에 바뀐 결정 자체가 유익했다는 근거가 된다.', '',
                  '각 CI는 조건별 95% paired cluster bootstrap이다. Object–surface와 첫 motion을 층화한 32개 block 안에서 orientation, regime, 환경과 32개 observation noise를 묶었다. Global-null은 4개 cell × 8개 독립 block에서 각각 실행했다. 보조 score는 동결한 source Gaussian을 marginalize해 계산했고 별도의 모델 학습이나 GPU timing은 수행하지 않았다. [수치](increment36.json) · [계산](increment36.py)'])
    (ROOT/'INCREMENT_KO.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
