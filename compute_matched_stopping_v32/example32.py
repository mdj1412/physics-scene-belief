"""A fixed index rule to illustrate a budget decision independently of its eventual payoff."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
def main():
    path=ROOT/'results_eval/directional/block_000/metrics.npz'
    with np.load(path) as z:
        counts=z['counts']
        differing=np.flatnonzero(counts[:,0]!=counts[:,1])
        e=int(differing[0]) if len(differing) else 0
        vals=[(method,counts[e,i],z['brier'][e,i]) for i,method in enumerate(['1×1','2×2','Fixed 4','Uncertainty-only'])]
        risk=z['oracle_stage_risk'][e]
        score=z['scores'][e]
        coupling=z['alpha'][e]
    text=[f'**하나의 실제 결정.** Directional law 첫 block에서 두 scorer의 최종 배분이 다른 첫 번째 환경(slot {e}, coupling {coupling[0]:+.2f}·{coupling[1]:+.2f})을 보자.' if len(differing) else
          f'**하나의 실제 결정.** Directional law 첫 block에서는 두 scorer의 최종 배분이 같아 사전 규칙대로 slot 0을 보았다.',
          '이 환경에 구매한 접촉 수는 '+', '.join(f'{m} {n}회' for m,n,_ in vals)+'였다. 같은 환경의 128개 물체를 각각 2·4·8회의 접촉 후 예측했을 때 실제 Brier는 '+', '.join(f'{v:.5f}' for v in risk)+'였다.',
          f'첫 2회를 받은 시점에 두 scorer가 예측한 2→4 Brier gain은 각각 {score[0,0]:+.5f}, {score[0,1]:+.5f}였다. 실제로 계속할지는 같은 block의 나머지 일곱 환경과 비교한 순위가 정한다. 이 예시는 전체 평균을 대체하지 않는다. [해당 block의 실제 결과](results_eval/directional/block_000/metrics.npz)']
    (ROOT/'EXAMPLE_KO.md').write_text(' '.join(text)+'\n')
if __name__=='__main__':main()
