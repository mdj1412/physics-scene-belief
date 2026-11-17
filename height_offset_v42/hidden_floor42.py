"""Exact paired Brier decomposition for the stipulated eight-offset distribution."""
import json
from pathlib import Path
import numpy as np
from selector42 import ROOT,BETA
from selector_audit42 import interval


def block(path):
    with np.load(path) as a:
        labels=a['labels'][a['offsets_mm']!=0]
        mu=labels.mean(0)
        floor=(mu*(1-mu)).mean()
        first=a['p_base'][2]; z=a['z'];cc=a['weights']; methods=a['choice'][2]
        # sign x regime x environment x noise x weight for the deployed selector.
        weight=np.choose(methods,[0.,.325,np.broadcast_to(cc[:,:,:,None],methods.shape)])
        odds=np.exp(BETA*z[:,None,:,None]*weight)[:,:,:,None,:]
        p=first[:,None]
        sel=p*odds/(1-p+p*odds)
        probs=(a['p_base'][0,:,None],p,sel)
        errors=[((q-mu[:,:,:,:,None])**2).mean() for q in probs]
        refs=a['risk'][a['offsets_mm']!=0]
        checks=np.array([refs[:,0,0].mean(),refs[:,2,0].mean(),refs[:,2,3].mean()])
        assert np.max(abs(checks-floor-errors))<1.e-12
        return np.array([floor,((mu>0)&(mu<1)).mean(),*errors])


def main():
    meta=json.loads((ROOT/'results/completion.json').read_text())
    grid=np.array([[block(ROOT/'results'/c/f'catalog_{p:02d}.npz') for c in meta['cells']] for p in range(12)])
    data={}
    for grouping in ('all','base','hollow','bulge','flare_h19','flare_h31','roll_h19','roll_h31'):
        index=[i for i,c in enumerate(meta['cells']) if grouping=='all' or grouping in c]
        s=grid[:,index].mean(1)
        data[grouping]=interval(s)
    result=dict(columns=['known_object_irreducible_Brier','fraction_ambiguous_objects','original_zero_minus_floor','estimated_zero_minus_floor','estimated_selector_minus_floor'],groups=data)
    (ROOT/'hidden_floor42.json').write_text(json.dumps(result,indent=2)+'\n')
    out=['# 관측되지 않는 contact height가 만드는 Brier 하한','',
         'Nonzero offset 여덟 값을 같은 확률로 적용하는 실험 분포에서, 물체와 noise realization을 고정하면 stick 확률 μ는 여덟 event의 평균이다. 모든 offset에 같은 확률 p를 제출할 때 평균 Brier는 (p−μ)²+μ(1−μ)다. 두 번째 항은 잠재 물체와 모든 offset event를 알아도 실제 offset만 숨겨져 있다면 남는 오차다. 첫 번째 항에는 물체 잠재값을 완전히 알 수 없는 데서 생기는 손실도 포함된다. 실제 offset을 알려준 protocol에는 같은 하한을 적용할 수 없다.','',
         '| 조건 | Event가 두 가지 이상인 물체 비율 | μ(1−μ) 평균 [95% CI] | Original 0%의 추가 Brier | Estimated 0%의 추가 Brier | Estimated selector의 추가 Brier |','|---|---:|---:|---:|---:|---:|']
    for group,v in data.items():
        x=v['mean']
        out.append(f"| {group} | {x[1]:.1%} | {x[0]:.5f} [{v['lower'][0]:.5f}, {v['upper'][0]:.5f}] | {x[2]:.5f} | {x[3]:.5f} | {x[4]:.5f} |")
    out+=['', '같은 prototype의 orientation·regime별 event도 따로 계산하고, 모든 비교에서 panel당 평균부터 냈다. 구간은 12개 panel의 개별 paired t 구간이다. 여기서 표본은 여덟 개 offset으로 정의한 synthetic distribution이다. 실제 기계의 placement error 분포에 대한 하한은 별도로 추정해야 한다. [원자료](hidden_floor42.json) · [검산 코드](hidden_floor42.py)']
    (ROOT/'HIDDEN_FLOOR_KO.md').write_text('\n'.join(out)+'\n')


if __name__=='__main__':main()
