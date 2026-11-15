"""Write the paper assessment only after all three independent train pairs are sealed."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
def main():
    main=json.loads((ROOT/'results_eval/full_retrain32.json').read_text())
    laws=json.loads((ROOT/'results_eval/seed_laws32.json').read_text())
    primary=json.loads((ROOT/'results_eval/paired_stats32.json').read_text())
    p=np.asarray(main['pooled'])
    band=np.asarray(main['crossed_simultaneous_ci'])
    if np.all(np.abs(band)<.003):
        takeaway='다섯 비교 모두에서 세 training seeds와 target blocks를 함께 재표집한 구간도 ±0.003 안에 들었다. 이 simulator와 예산에서 그 크기의 차이는 관측과 맞지 않는다.'
    else:
        takeaway='일부 비교에서는 training seed와 target block의 변동을 함께 고려한 구간이 ±0.003을 벗어났다. 세 번의 학습만으로 그 비교의 실용적 차이를 배제하기는 어렵다.'
    individual=np.asarray(laws['by_seed_law'])[:,2,0]
    positive=sum(individual>0)
    h=laws['directional_minus_historical_average']
    bounds=laws['interaction_crossed_ci']
    inference=('0을 포함하므로 directional law에 유독 취약하다고 단정할 수 없다.' if bounds[0]<=0<=bounds[1]
            else '0을 포함하지 않았다. 첫 seed를 본 후 제기한 보조 가설이고 training seed가 세 쌍뿐이므로, 새 법칙과 seed에서 독립적인 검증이 필요하다.')
    full=primary['full']['p90_ms']
    cost=(full[1]/full[2]-1)*100
    out=[f'**현재 선택은 fixed 4다.** 세 번의 독립 학습을 평균한 Brier에서 2×2 − 1×1은 {p[0]:+.5f} [{band[0,0]:+.5f}, {band[0,1]:+.5f}], 2×2 − fixed 4는 {p[2]:+.5f} [{band[2,0]:+.5f}, {band[2,1]:+.5f}]였다. {takeaway} 같은 probe 비용에서 fixed 4는 중간 순위를 다시 계산할 필요가 없어, 실측한 첫 seed의 전체 GPU p90에서 2×2보다 유리했다(2×2/fixed 4: +{cost:.1f}%).',
    f'**새 directional law에서의 재현성.** 개별 seed의 2×2 − 1×1 Brier는 {", ".join(f"{v:+.5f}" for v in individual)}였다. 2×2의 손실이 더 컸던 것은 세 쌍 중 {positive}쌍이었다. Directional에서의 차이에서 이전 두 laws의 평균 차이를 빼면 {h:+.5f} [{bounds[0]:+.5f}, {bounds[1]:+.5f}]였다. 이 탐색적 구간은 {inference} [개별 law와 stage별 검산](results_eval/seed_laws32.json)',
    (ROOT/'DISCUSSION_BASE_KO.md').read_text().strip()]
    (ROOT/'DISCUSSION_KO.md').write_text('\n\n'.join(out)+'\n')

if __name__=='__main__':main()
