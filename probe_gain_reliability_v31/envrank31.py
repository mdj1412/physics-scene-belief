"""Rank uncertainty by population setting for the paid-contact-only panel."""
from itertools import product
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
LAWS=('coulomb','split_static','speed_weak')


def ranks(gain):
    # Stage x panel cell x candidate. Higher signed gain is the preferred candidate.
    values=gain.reshape(2,8,16).transpose(0,2,1)
    diff=values[:,:,:,None]-values[:,:,None,:]
    return 1+(diff<0).sum(-1)+.5*((diff==0).sum(-1)-1)


def rank_unit(gain):
    r=ranks(gain)
    left,right=zip(*[(i,j) for i,j in product(range(16),repeat=2)
                     if i//4 != j//4 and i%4 != j%4])
    d=np.abs(r[:,left]-r[:,right])
    return np.stack([gain.mean((2,3)),r.mean(1),d.mean(1),(d>=3).mean(1)],axis=2)


def main():
    out=ROOT/'results_pool'
    all_laws=[]
    alpha_values=None
    for law in LAWS:
        blocks=[]
        for b in range(6):
            key=out/law/f'block_{b:03d}/metrics.npz'
            with np.load(key) as z:
                a=z['alpha']
                ix=np.lexsort((a[:,1],a[:,0]))
                if alpha_values is None:
                    alpha_values=a[ix]
                np.testing.assert_array_equal(a[ix],alpha_values)
                blocks.append(rank_unit(z['gain'][:,ix]))
        all_laws.append(blocks)
    bylaw=np.array(all_laws).mean(1)
    obj=dict(alpha=alpha_values.tolist(),metrics=['signed_gain','mean_rank','mean_absolute_rank_change',
             'prob_rank_change_at_least_3'],pooled=bylaw.mean(0).tolist(),bylaw=bylaw.tolist(),
             explanation='The eight candidates have independent future and deployment seeds. Sixteen panel replicates use matched labels across candidates. Exact average over disjoint row-and-column ordered pairs of those panel replicates, then equal weight within laws and across laws. Ties use midranks. Descriptive.')
    (out/'envrank31.json').write_text(json.dumps(obj,indent=2))
    rows=[]
    for e,(ag,av) in enumerate(alpha_values):
        fields=[]
        for k in range(2):
            g,r,delta,jump=bylaw.mean(0)[k,e]
            fields.extend([f'{r:.2f}',f'{delta:.2f}',f'{100*jump:.1f}%'])
        rows.append('| '+' | '.join([f'{ag:+.2f}',f'{av:+.2f}',*fields])+' |')
    md=['### 어떤 환경의 순위가 다시 측정해도 유지되는가?',
        '기존 물체에서 Y만 사는 조건에서 같은 prefix에 독립적인 future와 batch를 다시 적용했다. 각 반복마다 같은 law 안의 여덟 환경을 실제 gain 순으로 1–8위로 정렬했다. 아래는 모든 법칙·block을 같은 가중치로 평균한 결과다. 각 순위는 cell 하나에서 측정한 realized gain으로 정했다. 예를 들어 2위에서 5위가 되면 절대 이동은 3이다. 같은 cohort에서 양쪽 난수가 모두 다른 반복끼리 비교했고, 동률에는 평균 순위를 부여했다.',
        '| Geometry coupling | Velocity–friction coupling | 2→4 평균 순위 | 평균 절대 이동 | 3순위 이상 이동 | 4→8 평균 순위 | 평균 절대 이동 | 3순위 이상 이동 |\n|---|---|---|---|---|---|---|---|\n'+'\n'.join(rows),
        '두 coupling은 각각 profile mean–curvature spread와 horizontal velocity–friction spread 관계의 simulator 계수다. 각 행에는 같은 계수로 생성한 18개 서로 다른 prefix가 들어 있다. 순위 이동은 매번 같은 물체를 다시 배치한 결과가 아니라, 같은 prefix에서 새 접촉과 새 deployment worlds를 생성한 결과다. 완전히 무작위이고 서로 독립인 1–8위 순위의 평균 절대 이동은 2.625이고, 3순위 이상 움직일 확률은 46.875%다. [환경별 수치와 law별 결과](results_pool/envrank31.json) · [계산 코드](envrank31.py)']
    (ROOT/'ENV_RANK_KO.md').write_text('\n\n'.join(md)+'\n')

if __name__=='__main__':
    main()
