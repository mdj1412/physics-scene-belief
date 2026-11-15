"""Write a compact interpretation from prespecified primary and replication estimates."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(fn):return json.loads((ROOT/fn).read_text())
def interval(center,band):
    return f'{center:+.5f} [{band[0]:+.5f}, {band[1]:+.5f}]'
def main():
    results=load('results_eval/paired_stats32.json')
    rep=load('results_eval/full_retrain32.json')
    lp=load('results_train/label_precision.json')
    fits=results['training']
    val=results['brier']['delta'];bounds=results['brier']['simultaneous_ci']
    dur=results['full']['p90_ms']
    enough=all(bounds[i][1]<=-.003 for i in [2,4])
    better=all(bounds[i][1]<0 for i in [2,4])
    excludes=all(bounds[i][0]>-.003 for i in [0,2,4])
    if enough:
        first='2×2 평균 label로 배운 정책은 두 baseline에 대해 사전 설정한 0.003 Brier 개선을 달성했다.'
    elif better:
        first='2×2 평균 label로 배운 정책의 Brier가 두 baseline보다 낮았지만 일부 비교에서 실용적 개선 기준 0.003을 확립하지 못했다.'
    elif excludes:
        first='같은 학습 시간에서 2×2 평균 label로 학습해도 전체 stopping policy의 Brier는 실용적으로 개선되지 않았다. 1×1, fixed 4, uncertainty-only와의 비교 모두에서 0.003 개선은 simultaneous interval 밖이었다.'
    else:
        first='두 baseline을 모두 넘는 2×2 학습의 실용적 이득을 확인하지 못했다. 구간이 허용하는 개선 크기는 비교 대상에 따라 달랐다.'
    first='**'+first+'**'
    mean=results['brier']['mean']
    main_text=f"첫 seed에서 scalar gain을 한 번 측정한 1×1은 {fits['broad']['n_environments']}개 환경, 2개 futures와 2개 batches를 교차한 2×2는 {fits['repeated']['n_environments']}개 환경을 봤다. 실제 label 생성·fit 시간은 {fits['broad']['labels_plus_fit_s']:.2f}초와 {fits['repeated']['labels_plus_fit_s']:.2f}초였다. 두 정책, fixed 4, uncertainty-only의 held-out Brier는 각각 {mean[0]:.5f}, {mean[1]:.5f}, {mean[2]:.5f}, {mean[3]:.5f}였다. 2×2 − 1×1 차이는 {interval(val[0],bounds[0])}였다."
    rd=np.array(rep['delta_seed'])[:,0]
    p=lp['output']
    rs=[p[k]['conditional_noise_rmse_ratio_2x2_over_1x1']['estimate'] for k in ('2_to_4','4_to_8')]
    more=f"같은 prefix에서 2×2 평균으로 label을 만들면 conditional noise RMS는 2→4 stage에서 1×1의 {rs[0]*100:.1f}%, 4→8 stage에서 {rs[1]*100:.1f}%가 됐다. 별도 source data로 처음부터 세 번씩 학습했을 때 repeated − broad Brier는 {', '.join(f'{x:+.5f}' for x in rd)}였고, 평균은 {rd.mean():+.5f}였다."
    duration=f"여덟 환경의 전체 stopping·acquisition·decoding을 모두 실행한 GPU p90은 1×1 {dur[0]:.0f} ms, 2×2 {dur[1]:.0f} ms, fixed 4 {dur[2]:.0f} ms, uncertainty-only {dur[3]:.0f} ms였다. 네 방법 모두 한 block에서 calibration 32회와 deployment probe 104회를 지불했다."
    (ROOT/'FINDINGS_KO.md').write_text('\n\n'.join([first,main_text,more,duration])+'\n')
if __name__=='__main__':main()
