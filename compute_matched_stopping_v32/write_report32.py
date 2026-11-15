"""Assemble v32 measured results, publication figure, evidence review and reproducibility links."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
EVAL=ROOT/'results_eval'
NAMES=['1×1 · 더 많은 이력','2×2 · 반복 평균','Fixed 4','Uncertainty-only']
PAIRS=[(1,0),(0,2),(1,2),(0,3),(1,3)]
LAWS=['Static–kinetic split','Speed weakening','Directional friction']
def j(name):
    return json.loads((ROOT/name).read_text())
def md(name):
    return (ROOT/name).read_text().strip()
def table(t,rows):
    return '\n'.join(['| '+' | '.join(t)+' |','|'+'---|'*len(t)]+['| '+' | '.join(map(str,v))+' |' for v in rows])
def ci(x,p,dec=5):
    return f'{x:+.{dec}f} [{p[0]:+.{dec}f}, {p[1]:+.{dec}f}]'
def save_plot(stats,rep):
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    labels=['2×2 vs 1×1','1×1 vs fixed 4','2×2 vs fixed 4','1×1 vs uncertainty','2×2 vs uncertainty']
    ax=axes[0]
    xs=1000*np.array(stats['brier']['delta']);lo=1000*np.array(stats['brier']['simultaneous_ci'])
    y=np.arange(5)
    ax.axvline(0,c='0.5',ls='--',lw=.8)
    ax.errorbar(xs,y,xerr=np.array([xs-lo[:,0],lo[:,1]-xs]),fmt='o',color='#236e85',capsize=3,label='Seed 1, simultaneous 95%')
    pooled=1000*np.asarray(rep['pooled']);band=1000*np.asarray(rep['crossed_simultaneous_ci'])
    ax.errorbar(pooled,y+.19,xerr=[pooled-band[:,0],band[:,1]-pooled],fmt='D',ms=3.8,
                color='#ac633a',lw=1.2,capsize=2.5,label='3 pairs, crossed 95% (exploratory)')
    ax.set(yticks=y,yticklabels=labels,xlabel='Difference in Brier loss × 1,000',title='Held-out stopping policy');ax.invert_yaxis()
    ax.set_ylim(4.55,-.55)
    ax.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.45,-.34),ncol=2)
    ax=axes[1]; names=['1×1','2×2','Fixed 4','Uncertainty']
    colors=['#447d9b','#cf8d59']
    for i,key in enumerate(('full','warm')):
        y=np.arange(4)+(i-.5)*.32
        q=np.asarray(stats[key]['p90_ms']);lo=np.asarray(stats[key]['p90_ms_pointwise_ci'])
        ax.barh(y,q,height=.3,color=colors[i],label=['Full call','Deployment only'][i])
        ax.errorbar(q,y,xerr=np.maximum(0,np.array([q-lo[:,0],lo[:,1]-q])),fmt='none',ecolor='0.25',lw=1,capsize=2)
    ax.set(yticks=range(4),yticklabels=names,xlabel='GPU-synchronized p90 (ms / 1,024 worlds)',title='Measured on the first training pair')
    ax.invert_yaxis();ax.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.48,-.34),ncol=2)
    fig.savefig(EVAL/'v32.png',dpi=160);plt.close(fig)

def main():
    st=j('results_eval/paired_stats32.json');br=st['brier']
    repeat=j('results_eval/full_retrain32.json')
    charge=j('results_train/model_lock.json')['training']
    manifest=j('results_train/labels_complete.json')
    precision=j('results_train/label_precision.json')['output']
    paths=j('results_eval/paths32.json')
    horizon=j('results_eval/horizon32.json')
    shift=j('results_eval/shift32.json')
    ctrl=j('results_eval/attribution32.json')
    sens=j('results_eval/source_subset_sensitivity32.json')['subsets']
    plans=j('retrain_roots.json')['model_dirs']
    ledgers=[]
    for seed,path in enumerate(plans):
        ln=j(path+'/model_lock.json')
        lc=j(path+'/labels_complete.json')
        for n,arm in enumerate(['broad','repeated']):
            a=ln['training'][arm]
            ledgers.append([seed+1,NAMES[n],a['n_environments'],2*a['n_environments'],a['labeled_cells'],
                            f"{a['labels_s']:.2f}",f"{1000*a['fit_s']:.2f}",f"{a['labels_plus_fit_s']:.2f}",
                            f"{lc['warmup_s'][arm]:.2f}"])
    rows=[]
    for i,label in enumerate(NAMES):
        q=st['full']['p90_ms'][i];p=st['full']['p90_ms_pointwise_ci'][i]
        w=st['warm']['p90_ms'][i];v=st['warm']['p90_ms_pointwise_ci'][i]
        rows.append([label,f"{br['mean'][i]:.5f}",*[f"{br['secondary'][law]['brier'][i]:.5f}" for law in st['laws']],
                     f'{q:.1f} [{p[0]:.1f}, {p[1]:.1f}]',f'{w:.1f} [{v[0]:.1f}, {v[1]:.1f}]'])
    comparisons=[];diff_rows=[]
    for k,(a,b) in enumerate(PAIRS):
        lab=f'{NAMES[a].split(" ·")[0]} − {NAMES[b].split(" ·")[0]}'
        comparisons.append([lab,ci(br['delta'][k],br['simultaneous_ci'][k]),
                            *[ci(br['secondary'][law]['delta'][k],br['secondary'][law]['pointwise_ci'][k]) for law in st['laws']]])
        diff_rows.append([lab,*[f'{v[k]:+.5f}' for v in repeat['delta_seed']],
                          ci(repeat['pooled'][k],repeat['target_only_simultaneous_ci'][k]),
                          ci(repeat['pooled'][k],repeat['crossed_simultaneous_ci'][k])])
    prec=[]
    for who,stage in [('2_to_4','2→4'),('4_to_8','4→8')]:
        d=precision[who]
        val=lambda k:d[k]['estimate']
        lohi=d['conditional_noise_rmse_ratio_2x2_over_1x1']['percentile_95']
        prec.append([stage,f"{np.sqrt(val('single_cell_noise_variance')):.5f}",f"{np.sqrt(val('mean_2x2_noise_variance')):.5f}",
                     f"{val('conditional_noise_rmse_ratio_2x2_over_1x1'):.3f} [{lohi[0]:.3f}, {lohi[1]:.3f}]",
                     f"{val('within_block_diagonal_spearman'):.3f}"])
    precision_seeds=[]
    for i,path in enumerate(plans):
        measures=j(path+'/label_precision.json')['output']
        def ratio(stage):
            m=measures[stage]['conditional_noise_rmse_ratio_2x2_over_1x1']
            l,h=m['percentile_95']
            return f"{m['estimate']:.3f} [{l:.3f}, {h:.3f}]"
        precision_seeds.append([i+1,ratio('2_to_4'),ratio('4_to_8')])
    bypath=[]
    for i,name in enumerate(NAMES):
        bypath.append([name,*[ci(paths['pooled'][k][i],paths['pooled_pointwise95'][k][i]) for k in range(3)]])
    avg=np.array(repeat['by_seed']).mean(0)
    report=['# v32 · 같은 학습 시간에서 접촉 이력의 수와 gain label의 정밀도 비교',
            md('FINDINGS_KO.md'),'## 실험 설계와 비용',md('README_KO.md').split('\n',1)[1].strip(),
            table(['Training seed','학습 방식','독립 환경 수','Stage별 prefix 합계','Gain cells','Label 생성 (s)','Ridge fit (ms)','청구 GPU 시간 (s)','별도 warm-up (s)'],ledgers),
            '한 환경에서 2회·4회 prefix를 각각 하나씩 만든다. 2×2의 gain cell 네 개는 같은 prefix에 대한 한 개 label로 평균한다. 각 cell은 128개 deployment worlds 위에서 계산한 한 개 risk 변화량이다. Cell 수 ×128은 고유 물체 수가 아니다. 표의 시간은 같은 모델 GPU를 단독 사용하며 동기화한 wall time이다. Data generation, CPU posterior 처리, acquisition, Brier 계산, block 저장이 label 시간에 들어간다. GPU kernel time 또는 FLOPs와는 다르다. 초기 공통 detector 구성과 shape별 warm-up 및 audit 시간은 별도다. 두 arm의 전체 wall time은 남은 예산에 새 환경을 하나 더 처리할 수 있는지에 따라 조금 달라진다. [고정한 학습 예산](training_plan.json) · [GPU 학습 코드](train_labels32.py) · [GPU ridge fit](fit32.py)',
            f"첫 seed에서 model fit 전에 쓰는 파일 읽기·hash까지 포함하면 1×1 {charge['broad']['labels_plus_read_plus_fit_s']:.3f}초, 2×2 {charge['repeated']['labels_plus_read_plus_fit_s']:.3f}초다. 공통 posterior detector 초기화 {manifest['shared_setup_s']:.3f}초와 CUDA solver warm-up은 양쪽의 배분 시간 밖이다. 비교는 동일한 540초 allowance와 실제 소비량을 기준으로 하며, 물리 실험에서의 수집 시간은 측정하지 않았다.",
            '## Held-out Brier loss와 실제 GPU p90',
            table(['정책','세 law 평균 Brier','Static–kinetic','Speed weakening','Directional','전체 p90 ms [95%]','Deployment p90 ms [95%]'],rows),
            '첫 training seed로 고정한 모델의 결과다. 한 번의 full call은 8개 환경의 1,024개 물체를 처리한다. 법칙당 6개 독립 timing block ×3회, 정책당 54개 실측으로 p90을 구했다. Batch p90을 1,024로 나누면 단일 물체의 streaming p90이 되지 않는다. Warm p90과 full p90은 각각의 시간 분포에서 계산했다.',
            table(['Brier 차이','전체 [5개 simultaneous 95%]','Static–kinetic [95%]','Speed weakening [95%]','Directional [95%]'],comparisons),
            '각 차이에서 음수는 앞쪽 정책의 loss가 더 낮다는 뜻이다. 한 law 안의 16개 block을 통째로 재표집하는 paired bootstrap을 10,000회 수행했다. 법칙별 결과는 pointwise interval이다. 다섯 primary 대비에는 같은 resample로 구한 최대 표준화 편차를 써 simultaneous interval을 계산했다. 두 이전 law와 새 directional law의 결과를 구분했다. Reference practical margin은 absolute Brier 0.003이다. [사전 통계 계획](STATISTICAL_PLAN_KO.md) · [전체 효과와 p90 ratio 구간](results_eval/paired_stats32.json)',
            '![Held-out Brier loss contrasts and measured GPU p90](results_eval/v32.png)',
            '세 target 법칙에서 같은 latent world의 Coulomb 결과와 실제 stick label이 달라진 물체의 비율은 각각 '+', '.join(f'{100*v:.1f}%' for v in paths['target_label_diff_from_coulomb'])+'였다. 전체 repeated − broad 차이를 그 비율까지 곱한 Brier 기여로 나누면, label이 유지된 집단에서 '+f"{shift['pooled'][0][0]:+.5f}, 바뀐 집단에서 {shift['pooled'][1][0]:+.5f}였다. 이 그룹은 물성과 선택 난도가 달라질 수 있어 law mismatch의 인과 효과로 해석하지 않는다. [대조 설계](SHIFT_PLAN_KO.md) · [law별 분해 결과](results_eval/shift32.json)",
            '## 독립적으로 세 번 학습하면 유지되는가?',
            table(['Brier 차이','Seed 1','Seed 2','Seed 3','평균 [target block simultaneous 95%]','평균 [training·target crossed simultaneous 95%]'],diff_rows),
            f"세 training seed의 절대 Brier 평균은 1×1 {avg[0]:.5f}, 2×2 {avg[1]:.5f}, fixed 4 {avg[2]:.5f}, uncertainty-only {avg[3]:.5f}였다. 같은 target block을 공유한다. Seed 2·3은 독립 source data 생성부터 실제 GPU fit까지 반복하고, 동일한 evaluation archive에서 각 모델이 선택한 4곳과 2곳을 순차적으로 다시 구해 해당 action의 loss를 적용했다. 첫 번째 seed의 GPU 정책과 replay가 같은 counts와 Brier를 냈는지 검산했다. 첫 구간은 세 학습 쌍을 고정하고 target blocks만 바꿨을 때다. 마지막 구간은 training seed pair와 law별 target block을 함께 재표집한 탐색적 구간이다. 학습 반복이 세 쌍이므로 다른 모든 training seeds에 대한 정밀한 보장을 주기 어렵다. [재학습 계획](REPEATED_TRAINING_PLAN_KO.md) · [모델별 계산 결과와 seed 제외 분석](results_eval/full_retrain32.json)",
            '### Training seed와 contact law를 함께 바꾼 경우',
            '\n'.join(md('SEED_LAWS_KO.md').splitlines()[1:]).strip(),
            '## 반복으로 label이 안정돼도 예측과 배분이 좋아지는가?',
            table(['Stage','단일 gain의 조건부 noise RMS','2×2 평균의 조건부 noise RMS','RMS 비율 [95%]','독립 diagonal 간 환경 순위 Spearman'],prec),
            '위 표는 첫 학습 쌍의 진단이다. 아래는 각 training seed에서 새로 생성한 prefix로 같은 추정을 반복한 값이다. 비율이 1보다 작으면 반복 평균이 단일 label보다 정밀하다.',
            table(['Training seed','2→4 RMS 비율 [pointwise 95%]','4→8 RMS 비율 [pointwise 95%]'],precision_seeds),
            '같은 repeated-arm source prefix 안에서 future·batch·interaction variance를 각각 추정해 단일 측정과 교차 평균의 noise를 Brier 단위로 구했다. 두 축을 겹치지 않게 뽑은 대각 cell 쌍의 순위는 법칙 전체의 순위와 구분한다. 이는 source training label 자체의 변화다. [정밀도 추정](results_train/label_precision.json)',
            md('SOURCE_REFERENCE_KO.md').replace('# 학습한','### 학습한',1),
            '### 같은 training prefix에서 averaging만 바꾼 비교',
            table(['관측 집단','평균 label scorer','4개 단일 cell scorer의 Brier 범위','평균 label − 단일 cell 정책 평균 [pointwise 95%]'],
                [[label,f'{v[0]:.5f}',f'{min(v[1:]):.5f}–{max(v[1:]):.5f}',ci(d,band)] for label,v,d,band in
                 zip(['세 law 평균']+LAWS,[ctrl['pooled']]+ctrl['by_law'],[ctrl['repeated_minus_single_mean']]+ctrl['delta_law'],[ctrl['paired_ci']]+ctrl['delta_law_ci'])]),
            f"동일한 repeated-arm prefix에서 네 cell 중 하나만 골라 학습한 네 정책과 네 cell의 평균으로 학습한 정책을 비교했다. Ridge의 score는 네 single-cell score의 평균과 계산 오차 안에서 같다. 그러나 top-half 선택은 비선형이므로 평균 score가 선택한 물체와 네 policy가 선택한 결과의 평균은 달라질 수 있다. 최종 접촉 예산이 달랐던 환경 비율은 {100*ctrl['allocation_difference']:.1f}%였다. 이 보조 비교는 단일-cell 쪽의 절약 시간을 추가 prefix에 쓰지 않는다. [설계](ATTRIBUTION_PLAN_KO.md) · [분석](results_eval/attribution32.json)",
            '### 같은 probe budget을 어디에 썼는가?',
            table(['정책','2회에서 멈춰 생긴 Brier 차이','8회까지 계속해 생긴 Brier 차이','동일 histogram의 무작위 배분 대비 차이'],bypath),
            f"앞의 두 열을 더하면 fixed 4 대비 총 Brier 차이가 된다. 마지막 열은 2·4·8회에 각각 4·2·2곳을 균등 무작위로 배정하는 정확한 기대 Brier와 비교한다. Fixed 4도 총 32회지만 환경별 횟수 분포는 다르다. 두 learned policy가 stage 1에서 고른 네 환경의 평균 겹침은 {100*paths['overlap'][0]:.1f}%, 마지막 두 곳의 겹침은 {100*paths['overlap'][1]:.1f}%였고, 최종 접촉 횟수는 환경의 {100*paths['overlap'][2]:.1f}%에서 달랐다. [경로별 분석과 cutoff tie 수](results_eval/paths32.json)",
            md('EXAMPLE_KO.md'),
            '### 2→4회의 gain으로 첫 네 곳을 고르면 후속 기회를 놓치는가?',
            table(['평가','같은 block 정답으로 단기 순위','같은 block 정답으로 420개 배분 탐색','다른 blocks로 단기 순위','다른 blocks로 전체 배분 결정','Fixed 4'],
                  [[label,*[f'{v:.5f}' for v in arr]] for label,arr in zip(['세 law 평균']+LAWS,[horizon['pooled']]+horizon['by_law'])]),
            '각 행의 단위는 전체 policy Brier다. 4회까지 계속하는 이득이 작아도 8회까지 갈 기회가 중요할 수 있다. 첫 두 열은 target labels를 미리 아는 사후 계산이다. 다음 두 열은 동일 law의 나머지 15개 blocks에서 동일한 여덟 population coupling별로 stage Brier를 추정하고, 떼어둔 block에 그 배분을 적용한다. 이 참조는 law와 coupling을 안다고 가정하고 개별 prefix evidence는 읽지 않는다. 겹치는 학습 folds에 독립 표본 수를 부여하지 않았다. [보조 가설](HORIZON_PLAN_KO.md) · [결과](results_eval/horizon32.json)',
            '부분 training corpus만 쓴 sensitivity에서는 각 arm의 source blocks를 서로소인 절반 또는 삼분의 일로 나눠 동일 ridge policy를 다시 fit했다. 다섯 계산에서 repeated − broad Brier는 '+f"{min(x['repeated_minus_broad'] for x in sens):+.5f}에서 {max(x['repeated_minus_broad'] for x in sens):+.5f}까지였다. 분모가 다른 분할이 서로 겹치므로 다섯 번의 독립 full-budget 실험으로 세지 않았다. [결과와 각 분할의 실제 비용](results_eval/source_subset_sensitivity32.json)",
            '## 논문 기여 가능성과 다음 결정',md('DISCUSSION_KO.md'),md('PHYSICAL_SCOPE_KO.md'),
            md('LITERATURE_KO.md').replace('# v32 문헌 검토','## v32 문헌 검토',1),
            '## 재현 자료',md('PROVENANCE_KO.md')]
    save_plot(st,repeat)
    (ROOT/'REPORT_KO.md').write_text('\n\n'.join(report)+'\n')
if __name__=='__main__':main()
