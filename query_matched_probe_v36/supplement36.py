"""Evaluate a frozen gate using the conventional fourth probe likelihood."""
from pathlib import Path
import hashlib
import json
import numpy as np

from gate36 import BETA, ROOT, SIGNS, expected_gain, load_artifact, log_density, observations, posterior
from stats36 import CELLS, REGIMES, draw, resampling, simultaneous

METHODS = ('far_original','near_original','far_gate3','far_gate4','near_gate3','near_gate4',
           'near_predictions_far_gate4','far_predictions_near_gate4')
CONTRASTS = ('near4_minus_far4','fourth_increment_difference','near4_minus_far4_common_near_decoder',
             'near4_minus_far4_common_far_decoder')
SCOPES = {'local_equal':[0,1,2], 'strong':[0], 'weak':[1], 'null':[2],
          'global_null':[3], 'four_equal':[0,1,2,3]}

def conventional_likelihood(artifact, x, velocity, directions):
    result = np.empty((len(x), len(artifact['sign'])))
    if not np.all(np.isin(directions,SIGNS)):
        raise ValueError('Conventional direction must be negative or positive.')
    for action, sign in enumerate(SIGNS):
        selected = directions==sign
        if selected.any():
            result[selected] = log_density(artifact, observations(x[selected],velocity[selected]),action)
    return result

def choose(artifact,x,far_y,near_y,directions):
    if x.shape!=(8,4,11) or far_y.shape!=(8,4,2) or near_y.shape!=far_y.shape or directions.shape!=(8,4):
        raise ValueError('Unexpected paid-record dimensions')
    np.testing.assert_array_equal(far_y[:,:3], near_y[:,:3])
    ll3 = sum(conventional_likelihood(artifact,x[:,j],far_y[:,j],directions[:,j]) for j in range(3))
    w3 = posterior(ll3)
    z = w3 @ artifact['far_sign']
    fourth_far = conventional_likelihood(artifact,x[:,3],far_y[:,3],directions[:,3])
    fourth_near = log_density(artifact,observations(x[:,3],near_y[:,3]),2)
    scores = np.stack([expected_gain(artifact, weights, z) for weights in
                       (w3, posterior(ll3+fourth_far), posterior(ll3+fourth_near))])
    return z, scores

def risks(raw,predictions,z):
    weight=np.exp(BETA*z[:,None,None])
    adjusted=predictions*weight/(1-predictions+predictions*weight)
    actual=raw[...,None]
    base=((predictions-actual)**2).mean((1,2))
    shift=((adjusted-actual)**2).mean((1,2))-base
    return base,shift

def archive(artifact,path,checksum):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
    with np.load(path,allow_pickle=False) as result:
        z,score=choose(artifact,result['far_x'],result['far_y'],result['near_y'],result['dirs'])
        gate3,far4,near4=score>0
        np.testing.assert_allclose(z,result['z'],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(gate3,result['accept3'])
        np.testing.assert_array_equal(near4,result['accept4'])
        np.testing.assert_allclose(score[[0,2]],np.stack([result['score3'],result['score4']]),atol=1e-9,rtol=0)
        fr,df=risks(result['labels'],result['p_far'],z)
        nr,dn=risks(result['labels'],result['p_near'],z)
        measures=np.stack([fr,nr,fr+gate3*df,fr+far4*df,nr+gate3*dn,nr+near4*dn,
                           nr+far4*dn,fr+near4*df],axis=-1)
        np.testing.assert_allclose(measures[:,[0,1,4,5]].T,result['risk'][[0,2,4,5]],atol=1e-10,rtol=0)
    gates=np.stack([gate3,far4,near4,(near4!=far4)],axis=-1)
    return np.concatenate((measures,gates),axis=-1),far4

def summarize(data):
    ids=resampling(8,seed=360411)
    out={}
    for name,entries in SCOPES.items():
        x=data[:,:,entries].mean(axis=2)
        vals=x.mean((0,1))
        diff=np.stack([x[...,5]-x[...,3],(x[...,5]-x[...,4])-(x[...,3]-x[...,2]),
                       x[...,5]-x[...,6],x[...,7]-x[...,3]],axis=-1)
        mean=diff.mean((0,1))
        band=simultaneous(mean,draw(diff,ids))
        out[name]=dict(means=vals[:8].tolist(),contrast=mean.tolist(),simultaneous_ci95=band.tolist(),
                       application_rate=dict(zip(('prefix3','conventional4','matched4','disagree4v4'),vals[8:].tolist())))
    return out

def write_table(s):
    labels=('세 local 균등','강한 방향성','약한 방향성','query 부근 방향성 없음','전체 표면 방향성 없음','네 regime 균등')
    data=s['scopes']
    lines=['## 같은 gate에 conventional fourth와 query-matched fourth를 각각 제공한 결과','',
           '원래 v35의 conventional reader와 새 near gate의 차이에는 방향 추정 방식의 변화도 들어간다. 여기서는 두 gate가 같은 frozen source model, 처음 세 접촉의 signed proposal, 같은 β를 쓴다. 네 번째 conventional response는 실제 부호에 맞는 action likelihood로 갱신하고, near response는 query 부근 action의 likelihood로 갱신했다. 표에서 on 비율의 단위는 환경이다.','',
           '| 평가 범위 | Gate on, prefix3 | Gate on, conventional4 | Gate on, matched4 | 두 fourth의 결정이 달랐던 비율 | Far gate4 Brier | Near gate4 Brier |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,(name,_) in zip(labels,SCOPES.items()):
        entry=data[name]
        rates=list(entry['application_rate'].values()); avg=entry['means']
        lines.append('| '+' | '.join([label,*[f'{100*r:.1f}%' for r in rates],f'{avg[3]:.5f}',f'{avg[5]:.5f}'])+' |')
    lines+=['','| 평가 범위 | Near4 − far4, 각자의 decoder | 네 번째 측정의 difference-in-differences | Near4 − far4, 같은 near decoder | Near4 − far4, 같은 far decoder |',
            '|---|---:|---:|---:|---:|']
    for label,(name,_) in zip(labels,SCOPES.items()):
        entry=data[name]
        lines.append('| '+' | '.join([label,*[f'{v:+.5f} [{lo:+.5f}, {hi:+.5f}]'
                           for v,(lo,hi) in zip(entry['contrast'],entry['simultaneous_ci95'])]])+' |')
    entry=data['local_equal']
    def interpretation(k):
        v=entry['contrast'][k]; lo,hi=entry['simultaneous_ci95'][k]
        return (f'{v:+.5f} [{lo:+.5f}, {hi:+.5f}]', 'matched가 유리했다' if hi<0 else 'conventional이 유리했다' if lo>0 else '부호를 판정하기 어렵다')
    common,cc=interpretation(2); did,dc=interpretation(1)
    lines+=['',f'같은 near probabilities에 적용했을 때 세 local regime 평균은 {common}로 {cc}. Difference-in-differences는 {did}로 {dc}. 양쪽에서 네 번째 probe를 제공했을 때의 변화량을 각각 구했으므로, 이 지표는 처음 세 번만 읽은 정책에 대한 단순 우위를 넘어 어느 네 번째 동작이 더 유용했는지 평가한다.',
            '','세 local regime은 conventional 접촉 지점의 law를 공유한다. 실제로 같은 orientation·block에서 conventional fourth까지 사용한 gate decision이 세 regime 모두에서 같음을 모든 archive에서 확인했다. 그럼에도 conventional fourth가 첫 세 번의 방향 추정을 보완하거나 global-null을 구별하는 데 쓰일 수 있다. 각자의 decoder를 사용한 Brier에는 deployment action이 달라진 효과도 포함된다. 같은 decoder의 두 열에서는 같은 물체의 같은 sensor realization과 같은 proposal을 사용한다.',
            '','각 범위의 네 contrast에는 object × surface와 시작 부호를 층화한 paired bootstrap 12,000회의 simultaneous 95% interval을 사용했다. 새 conventional gate는 GPU가 저장한 관측과 prediction으로 CPU에서 평가했으며, 이 방법만의 GPU p90은 미측정이다. [보조 분석 계획](SUPPLEMENT_PLAN_KO.md) · [계산 코드](supplement36.py) · [수치](supplement36.json)']
    (ROOT/'SUPPLEMENT_KO.md').write_text('\n'.join(lines)+'\n')

def main():
    artifact=load_artifact()
    panel=np.empty((4,8,4,len(METHODS)+4))
    count=0
    for c,cell in enumerate(CELLS):
        for b in range(8):
            folder=ROOT/'results'/cell/f'block_{b:03d}'
            meta=json.loads((folder/'meta.json').read_text())
            fars={}
            for r,regime in enumerate(REGIMES):
                orientations=('plus',) if regime=='global_null' else ('plus','minus')
                values=[]
                for orientation in orientations:
                    name=f'{regime}_{orientation}.npz'
                    row,far=archive(artifact,folder/name,meta['sha256'][name])
                    if regime!='global_null':
                        if orientation in fars: np.testing.assert_array_equal(far,fars[orientation])
                        fars[orientation]=far
                    values.append(row.mean(0))
                    count+=1
                panel[c,b,r]=np.mean(values,axis=0)
    assert count==224 and np.isfinite(panel).all()
    output=dict(methods=METHODS,contrasts=CONTRASTS,verified_archives=count,scopes=summarize(panel))
    (ROOT/'supplement36.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    write_table(output)
    print(output['scopes']['local_equal'])
if __name__=='__main__':main()
