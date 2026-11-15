"""Factor the one-probe information ceiling into velocity roles at a fixed height."""
from pathlib import Path
from math import erf, sqrt
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'design37.json').read_text())
PAIRS = ('strong_vs_null','weak_vs_null')
SCALES = (1., .3)
SPEEDS = tuple(CFG['speeds'])

def collision(theta, cell, orient, scale, moving, rate):
    obj,surface=cell.split('_')
    _,_,slope,curve,mu,e=theta.T
    height=CFG['height']/CFG['query_height']
    local=slope+curve*height+(-.045*np.sin(height) if obj=='sinusoid' else .04*height**3)
    norm=np.hypot(local,1.)
    vt=(moving+local)/norm
    rt=(rate+local)/norm
    jn=(1+e)*np.maximum((1-moving*local)/norm,0.)
    mod=.21*(2./np.pi)*np.arctan(rt/.40) if surface=='atan' else .16*rt/np.hypot(.30,rt)
    capacity=mu*(1.+orient*scale*mod)*jn
    jt=np.clip(vt,-capacity,capacity)
    return np.stack((1-(jn+jt*local)/norm,moving+(jn*local-jt)/norm),axis=-1),abs(vt)<=capacity

def pair_accuracy(theta,cell,sign,scale,moving,rate):
    mu0=collision(theta,cell,sign,0.,moving,rate)[0]
    mu1=collision(theta,cell,sign,scale,moving,rate)[0]
    dist=np.linalg.norm((mu1-mu0)/CFG['noise_sd'],axis=-1)
    return .5*(1+np.array([erf(v/(2*sqrt(2))) for v in dist]))

def cell_values(cell,b):
    folder=ROOT/'results'/cell/f'block_{b:03d}'
    # orientation mean, then environment mean, preserving block as the sampling unit.
    data=np.zeros((2,2,4))
    for sign,side in ((1,'plus'),(-1,'minus')):
        truth=None
        for regime,scale in (('strong',1),('weak',.3),('null',0)):
            with np.load(folder/f'{regime}_{side}.npz',allow_pickle=False) as raw:
                th=raw['theta4']
                if truth is None: truth=th.copy()
                np.testing.assert_array_equal(th,truth)
                for j,speed in enumerate(SPEEDS):
                    vel,stick=collision(th,cell,sign,scale,speed,speed)
                    np.testing.assert_allclose(vel,raw['matched_clean'][j],rtol=1e-12,atol=1e-12)
                    np.testing.assert_array_equal(stick,raw['matched_stick'][j])
        for pi,s in enumerate(SCALES):
            for i,high in enumerate(SPEEDS[1:]):
                grid=[(SPEEDS[0],SPEEDS[0]),(high,SPEEDS[0]),(SPEEDS[0],high),(high,high)]
                data[pi,i]+=.5*np.array([pair_accuracy(truth,cell,sign,s,m,r).mean() for m,r in grid])
    return data

def decompose(x):
    a,b,c,d=np.moveaxis(x,-1,0)
    return np.stack((.5*((b-a)+(d-c)),.5*((c-a)+(d-b)),d-a,d-b-c+a),axis=-1)

def summarize():
    cells=CFG['cells']
    x=np.array([[cell_values(cell,b) for b in range(CFG['blocks_per_cell'])] for cell in cells])
    decomp=decompose(x)
    mean=decomp.mean(axis=(0,1))
    np.testing.assert_allclose(mean[...,:2].sum(-1),mean[...,2])
    ids=(2*np.random.default_rng(371449).integers(4,size=(12000,4,4,2))+np.arange(2)).reshape(-1,4,8)
    reps=decomp[np.arange(4)[None,:,None],ids].mean(axis=(1,2))[...,:2]
    se=reps.std(axis=0,ddof=1)
    crit=np.quantile(np.max(abs((reps-mean[...,:2])/se),axis=(1,2,3)),.95)
    ci=np.stack((mean[...,:2]-crit*se,mean[...,:2]+crit*se),axis=-1)
    body={'regime_pairs':PAIRS,'speed':SPEEDS[1:],'response_mean_order':['low_low','high_low','low_high','high_high'],
          'response_mean':x.mean(axis=(0,1)).tolist(),'effect_order':['kinematics','modulation','total','interaction'],
          'effects':mean.tolist(),'component_ci95sim':ci.tolist(),'checked_local_archives':4*8*6,
          'bootstrap_iterations':len(ids),'verified_diagonal_speeds':SPEEDS}
    (ROOT/'physics_factor37.json').write_text(json.dumps(body,indent=2,allow_nan=False)+'\n')
    return body

def report(body):
    lines=['## 속도 변화가 sensor separation에 기여하는 두 경로','',
      '같은 calibration 물체가 두 friction law 중 어느 쪽인지 구분한다고 생각하자. Kinematics 속도는 접촉 직전 tangential velocity와 normal impulse를 정하고, modulation 속도는 directional friction coefficient의 velocity 항에 들어간다. 두 입력을 독립적으로 .50 또는 높은 속도로 바꿔 같은 물체의 outgoing velocity를 계산했다. 두 속도를 같게 둔 경우에는 192개 local raw archive의 세 속도에서 clean velocity와 stick label이 모두 일치했다. 교차한 조건은 simulator에만 넣은 반사실이다. [설계](PHYSICS_FACTOR_PLAN_KO.md) · [계산 코드](physics_factor37.py)','',
      '표의 단위는 accuracy percentage point다. 두 후보를 같은 확률로 가정하고 latent geometry와 orientation까지 제공한 privileged Bayes accuracy의 증가분을 표시한다. Kinematics를 먼저 바꿨을 때와 modulation을 먼저 바꿨을 때의 증가분을 평균해 두 성분에 배분했다. 양쪽 합은 총 증가량과 같다. 마지막 열은 두 경로의 interaction이며 총 증가량에 다시 더하지 않는다. 여덟 성분 추정에 simultaneous 95% interval을 적용했다.','',
      '| 구별할 regime | 높은 속도 | 총 증가 (pp) | Kinematics (pp) [95%] | Friction modulation (pp) [95%] | Interaction (pp) |',
      '|---|---:|---:|---:|---:|---:|']
    for j,pair in enumerate(PAIRS):
        for i,v in enumerate(SPEEDS[1:]):
            kin,mod,total,inter=body['effects'][j][i]
            cl=body['component_ci95sim'][j][i]
            def fmt(w,ci):return f'{w*100:+.2f} [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]'
            lines.append(f'| {pair} | {v:.2f} | {total*100:+.2f} | {fmt(kin,cl[0])} | {fmt(mod,cl[1])} | {inter*100:+.2f} |')
    lines+=['','같은 물체의 두 표면을 비교하면 normal impulse Jₙ과 접선 방향 단위벡터가 같아, 두 outgoing velocity mean 사이의 거리는 |Jₜ₁−Jₜ₂|다. 두 충돌이 모두 stick이면 Jₜ가 tangential demand로 같아진다. 두 충돌이 모두 slip이고 접선 방향이 같으면 그 거리는 Jₙ μ |a₁−a₂| |f(vₜ)|가 된다. 여기서 a는 해당 높이에서의 방향 효과 세기, f는 속도에 따른 방향 효과의 modulation이다. 따라서 미끄러지는 물체 수가 더 늘지 않는 구간에서도 Jₙ이나 f가 달라지면 두 velocity 분포를 더 쉽게 또는 어렵게 구별하게 된다.', '', 'Kinematics 성분에는 stick에서 slip으로 넘어간 물체뿐 아니라, 이미 미끄러졌어도 normal impulse 등이 달라진 물체가 포함된다. Modulation 성분 역시 바뀐 friction coefficient가 contact state에 주는 피드백을 포함한다. 두 경로는 이 simulator가 생성한 두 Gaussian sensor likelihood에 관한 계산이다. 독립 물체의 query Brier와 실제 gate의 식별 성능은 본문의 held-out 결과로 판단한다. [수치 원본](physics_factor37.json)']
    (ROOT/'PHYSICS_FACTOR_KO.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report(summarize())
