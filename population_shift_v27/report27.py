"""Rebuild Korean report from verified server statistics and the source reading ledger."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def read(name):return json.loads((root/'results'/(name+'.json')).read_text())
s=read('summary');p=s['pooled'];ex=read('extended');powr=read('identifiable_power');tw=read('twins');pairs=read('twin_actions')
cfg=json.loads((root/'design.json').read_text())
def tab(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
def f(v):return f'{v:.5f}'
def sig(v):return f'{v:+.5f}'
def pct(v):return f'{100*v:.1f}%'
def interval(ab):return '['+', '.join(sig(v) for v in ab)+']'
def delta(val,band):return sig(val)+' '+interval(band)
names=['Estimated-prior K256','Frozen gate','고정 K256','Known-prior K256','Always-query','Neutral K256']
families=s['target_laws']+['source','f_negative','f_positive']
def law(k):
    ac,af=next(e['alpha'] for e in cfg['environments'] if e['family']==k)
    return f'({ac:g}, {af:g})'
parts=['# v27 · 여러 world의 factual evidence로 observation 선택을 어디까지 바꿀 수 있는가',
'''**두 proxy 중 profile–curvature 관계는 추정할 수 있었고, velocity–friction 관계는 이 실험의 factual evidence만으로 식별할 수 없었다.** 8개 unseen shift에서 estimated-prior K256의 held-out Brier는 **0.11083**이었다. 기존 prior를 쓰는 K256은 **0.19296**, 모집단의 두 shift를 실제로 아는 known-prior K256은 **0.09765**였다. 그러나 어떤 calibration worlds도 보지 않고 두 관계를 중립으로 둔 neutral K256도 **0.11468**에 도달했다. Estimated−neutral 차이는 −0.00385, simultaneous 95% interval은 [−0.01163, +0.00392]였다. 관측하지 않고 얻을 수 있는 개선을 먼저 제하면 factual batch 자체가 제공한 decision 이득의 크기는 아직 불확실하다.

같은 known-prior decoder로 평가해도 estimated policy는 source K256보다 −0.02674 낮은 Brier를 기록했다. 이 차이는 observation 선택에서 나온다. 반면 always-query와 동일 decoder에서 비교한 차이는 −0.00274 [−0.00918, +0.00371]였다. GPU p90은 estimated 3.343 ms, source K256 3.313 ms, always-query 0.737 ms였다. Brier 개선과 속도 개선은 다른 결론을 준다.''',
'## 1. 데이터와 배포 조건',
'''각 world에서 한 번의 factual collision으로 noisy boundary profile 9개와 outgoing velocity 2개를 읽는다. 정책은 같은 world에서 query 위치의 정밀 geometry를 한 번 더 볼지(query view), 정해진 다른 충돌의 velocity를 측정할지(probe) 고른다. 그 한 번의 추가 관측으로 다른 위치·입사속도에서 접촉이 stick할 확률을 예측한다. 평가값은 binary stick label과 이 확률의 squared difference인 Brier loss다. 작은 값이 좋다.

V26과 같은 geometry, Coulomb contact law, sensor noise, action pair, M2048 acquisition을 썼다. αc는 profile offset의 중심값이 curvature spread에, αf는 restitution의 중심값이 friction spread에 정렬되는 정도다. 각 관계에서 중심을 반대로 쓰는 확률은 (1−α)/2다. 1은 source 관계, 0은 독립, −1은 완전 반전이다. 학습된 v25 RBF gate와 cutoff를 hash로 확인했다. V26의 여섯 단계 grid에서 직접 시험하지 않았던 8개 조합을 먼저 정했고, (1,1), (0,−1), (0,+1)은 별도 controls로 남겼다. 여기서 unseen은 동일한 simulator family에서의 새로운 coupling strength를 뜻한다.

각 조합에서 독립 cohort 3개를 생성했다. 각 cohort의 calibration 1,024개 worlds와 평가 128개 worlds는 서로 다른 난수 stream으로 iid 추출했다. Hidden cell 수를 강제로 균형화하지 않았다. 각 평가 world는 32회 추가 observation noise를 두 action에 대해 갖는다. Primary는 24개 cohort, 3,072개 평가 worlds이며 controls까지 총 33개 cohort, 4,224개 평가 worlds다. 같은 replicate index의 numerical bank pair를 환경 사이에 공유해 비교를 짝지었다. 추가 observation이나 label을 생성하기 전에 정책이 정한 action과 factual inputs를 별도 hash 파일로 동결했다.

실행 전 사용자 allocation 세 장의 process·memory·utilization을 조회했다. 두 장에 다른 Python 작업이 있었고, node55의 RTX PRO 6000 Blackwell Server Edition 한 장(0 MiB, 0%, Slurm 2744468)에서 모든 GPU 실험을 순차적으로 실행했다. 통계와 분석적 simulation은 할당된 compute-node CPU에서 처리했다. [실험 설계](design.json) · [사전 protocol](PROTOCOL.md) · [GPU worker 종료 기록](gpu_worker/results/worker_exit.json)''',
'## 2. 무엇이 식별되고 무엇이 남는가',
'''Factual collision에서는 incoming tangential velocity가 0이고 모든 후보가 stick한다. Local slope를 s, restitution을 e라 두면 outgoing velocity는 (−e/(1+s²), es/(1+s²))가 된다. Friction coefficient가 식에서 빠진다. 넓은/좁은 friction 집단은 등확률이고 geometry와 독립이다. 두 집단 사이에서 restitution 중심을 바꿔도 e의 marginal과 geometry와의 joint law가 유지된다. 따라서 임의 크기의 iid factual batch에 대해 αf의 likelihood는 평평하다. 방향이 반대인 두 환경에 같은 prior probability를 주면, factual-only classifier의 최선의 평균 방향 판별 정확도도 50%다.

Profile에는 다른 신호가 있다. M을 9점 profile mean, T를 quadratic curvature의 least-squares coefficient로 두면 대칭 좌표와 독립 Gaussian noise 아래 E[M(T²−E[T²])] = 0.00115 × (0.340²−0.022²) × αc / 2다. 이 값은 αc가 변하면 달라진다. 즉 관측 marginal 전체가 보존되는 friction과 달리, 눈에 띄기 어려운 profile 내부의 고차 관계는 남아 있다. [물리 식](../within_calibration_v25/physics25.py) · [moment와 독립 power 검산](diagnostics27.py)''',
'''추정기는 실제 simulator에서 8,192개 후보로 aligned/swapped profile에 대한 두 factual likelihood를 계산한다. Friction spread를 쌍으로 구성해 αf 반전이 같은 candidate signals의 순열이 되게 했다. 각 calibration world에서 두 값을 αc에 따라 혼합하고, 401개 지점에서 log likelihood를 더해 uniform hyperprior에 대한 posterior mean과 구간을 구한다. αf의 posterior는 처음 정한 uniform hyperprior 그대로이며 mean 0을 사용한다. **αf=0은 실제로 shift가 없다는 측정값이 아니다.** 두 독립 coupling의 one-world predictive mixture가 각 축에 선형이므로 이 설정에서 posterior mean을 사용하면 marginal predictive prior와 일치한다. 이후 K256은 이를 finite bank로 근사한다. 이 방법은 contact law와 sensor calibration을 안다고 가정한다. [Estimator 구현](estimator27.py)''',
'## 3. Held-out Brier와 GPU p90',
'''Frozen gate는 source RBF score로 관측을 직접 고르고 낮은 confidence에서는 K1024로 넘긴다. 고정 K256은 source prior (1,1), neutral K256은 (0,0), known-prior K256은 실제 환경의 (αc,αf)를 사용한다. Known-prior의 특권 정보는 집단의 law까지이며 개별 world의 latent나 label은 숨겼다. Always-query는 source decoder로 query view만 읽는다. Native 비교에서는 각 정책이 배포할 decoder를 사용했다. 공통 decoder 열에서는 모든 정책이 고른 action을 같은 known-prior decoder로 다시 채점했다. 따라서 공통 열의 차이는 action 변화에 한정된다.''',
tab(['정책','Native Brier','공통 decoder Brier','Probe 선택률','Warm GPU p50 ms','Warm GPU p90 ms','128개에 setup 배분한 p90 ms'],[[name,f(p['native'][i]),f(p['common'][i]),pct(p['probe_fraction'][i]),f"{p['p50'][i]:.3f}",f"{p['p90'][i]:.3f}",f"{p['amortized_p90'][i]:.3f}"] for i,name in enumerate(names)]),
tab(['Estimated에서 뺀 비교 대상¹','Native Δ [simultaneous 95%]','공통 decoder Δ [simultaneous 95%]','Estimated와 다른 선택','Warm p90 ratio [pointwise 95%]'],[[names[i],delta(p['deltas'][i-1],p['sim95'][i-1]),delta(p['deltas'][i+4],p['sim95'][i+4]),pct(p['choices_differ_from_est'][i]),f"{p['p90'][0]/p['p90'][i]:.3f} [{p['time_ratio_ci95'][i][0]:.3f}, {p['time_ratio_ci95'][i][1]:.3f}]"] for i in range(1,6)]),
'''¹ Δ = estimated loss − 비교 대상 loss다. 8개 target laws에 같은 가중치를 주고, 3개 replicate/bank blocks를 함께 재추출한 뒤 해당 block·law 안의 test worlds를 재추출했다. 방법 간 짝은 유지했다. 10,000회 bootstrap으로 native/common의 총 10개 평균 차이에 대해 studentized scale의 최대 절댓값을 써서 simultaneous bands를 구했다. Timing은 2,000회 paired world/block bootstrap의 pointwise interval이다. 세 bank의 반복으로 모집단 전체에 대한 오차를 정밀하게 측정하기는 어렵다. 각 block을 한 번씩 제외해도 native estimated−source K256은 −0.08628에서 −0.07837, estimated−neutral은 −0.00643에서 −0.00091이었다. 절댓값이 작은 후자에서 simultaneous interval은 0을 포함한다.''',
'''Source K256에서 estimated로 바꾸면 action이 29.6% 달라진다. Neutral K256은 primary의 모든 world에서 query view를 골랐다. 그래서 common decoder를 쓰면 neutral과 always-query의 Brier가 정확히 같다. Estimated는 10.2%에서 probe를 사용했다. 이 변경에 고유한 평균 이득은 통계적으로 확정되지 않았다. Known-prior가 고른 action과도 12.7%에서 달랐다. Known-prior K256 역시 근사 계산이므로 Bayes lower bound로 취급할 수 없다.''',
f"Timing은 batch-1 raw factual input부터 실제로 고른 observation의 전송과 최종 확률 계산까지 CUDA synchronization을 포함했다. 환경별 무작위 16 worlds × 2 repeats × 24 cohorts로 정책당 768회다. 여섯 정책의 실행 순서는 섞었다. Estimated의 평균 calibration fit은 {p['fit_ms']:.2f} ms, 환경별 bank 생성까지 합하면 {p['overhead_ms'][0]:.2f} ms다. 마지막 열은 각 방법의 환경별 setup을 128개 평가 world에 균등 배분한 계산값이다. Cold static estimator library 생성 {s['init_ms']:.1f} ms는 별도 1회 비용이다. GPU p90에는 실물 sensing, 1,024개의 calibration world 수집, 물체 reset 시간이 들어 있지 않다. Always-query도 동일 engine 생성 비용으로 계상해 setup 추정이 전용 구현보다 클 수 있다. [원시 GPU 실행 및 timing](runner27.py) · [통계 요약](results/summary.json)",
'## 4. 어느 환경에서 이득이 사라지는가',
tab(['(αc, αf)','용도','평균 α̂c','Estimated','Frozen gate','고정 K256','Known K256','Always-query','Neutral K256'],[[law(k),'target' if k in s['target_laws'] else 'control',f"{s['by_family'][k]['alpha_mean'][0]:+.3f}"]+[f(v) for v in s['by_family'][k]['native']] for k in families]),
'''두 관계가 아직 원래 방향에 남은 (+0.35,+0.65)에서 estimated의 Brier 0.13236은 frozen gate 0.12554와 always-query 0.11788보다 높았다. Source control에서도 frozen gate 0.09016에 비해 estimated 0.10511이었다. 변화가 없는 환경에서도 αf를 중립화하는 대가가 나타난다. 환경별 수치는 독립 calibration batch 세 개의 평균이며, 약한 shift를 모두 검출할 수 있다는 증거로 해석하기 어렵다.''',
'## 5. Batch에서 실제로 알아낸 정보',
tab(['Calibration worlds','Profile MAE','Profile 부호 정확도','부호 posterior probability >95%인 cohort','평균 posterior KL (nats)'],[[n,f"{v['mae']:.3f}",pct(v['sign_accuracy']),str(v['sign_confident_n'])+'/24',f"{v['mean_kl']:.3f}"] for n,v in s['prefixes'].items()]),
'''서로 같은 batch의 prefixes를 비교했다. 1,024개에서 profile 부호는 20/24회 맞았고, continuous alternative 대 source (αc=1)의 Bayes factor가 10을 넘은 target cohort는 14/24개였다. 후자는 profile 방향 판별과 다른 지표다. 95% equal-tail posterior interval의 평균 폭은 1.266이었다. αc=1은 연속 uniform prior의 경계값이므로 중앙 equal-tail interval로는 이 endpoint를 포함할 수 없다. 24개 전체를 한데 묶어 명목 coverage를 주장할 수 없다. αf의 posterior information gain은 모든 batch 크기에서 정확히 0이다.''',
'''별도 CPU power 실험에서는 αc=−0.8, 0, +0.8마다 새 iid batch 96개를 만들었다. (M,T)의 projected likelihood에 대한 bounded MLE를 사용했다. 이는 GPU 배포 estimator와 독립적인 식별력 검산이다. 아래 power는 αc=0에 대한 nominal 5% two-sided score test가 귀무가설을 기각한 비율이다. 0행은 false positive rate다.''',
tab(['αc','N','MLE 평균','SD','부호 정확도','αc=0 기각률'],[[r['alpha_c'],r['n'],f"{r['projected_mle']['mean']:+.3f}",f"{r['projected_mle']['sd']:.3f}",'—' if r['alpha_c']==0 else pct(r['projected_mle']['sign_accuracy']),pct(r['projected_mle']['reject_zero'])] for r in powr['rows'] if r['n'] in (1024,4096,16384)]),
'αc의 부호가 맞을 가능성과 0에서 유의하게 떨어졌다고 말할 수 있는 확률은 다르다. 예컨대 +0.8, N=1,024에서 각각 93.8%와 39.6%였다. [전체 표본수·moment·quadrature 검산](results/identifiable_power.json)',
f"Bayes factor >10을 profile-shift 경보로 해석한 탐색적 대조에서, 실제 αc가 1인 source control은 {s['by_family']['source']['strong_source_bf']}/3회, friction만 반전한 target (1,−0.8)은 {s['by_family']['f_rev']['strong_source_bf']}/3회 경보를 냈다. αc가 source와 다른 나머지 target cohort에서는 {p['strong_source_bf']-s['by_family']['f_rev']['strong_source_bf']}/21회였다. 표본이 작고 Bayes factor는 policy를 켜고 끄는 threshold로 사용하지 않았다. 추정 posterior는 경보 여부와 무관하게 K256의 prior에 반영됐다.",
'## 6. Friction이 보이지 않는 것이 decision에도 영향을 주는가',
'''반대 αf를 가진 두 환경의 world를 일대일로 짝지었다. Friction spread bit와 coupling flip을 함께 반전하면 같은 restitution과 geometry 및 factual noise를 유지하면서 서로 다른 friction을 만들 수 있다. 네 쌍의 환경에 16,384개씩 생성했을 때 모든 factual input의 최대 차이는 4.45×10⁻¹⁶ 이하였다. 같은 두 world의 query view는 동일했지만 probe response와 query stick label은 달라질 수 있었다. 이는 한 가지 명시적 coupling에서 관찰한 label 변경률이며 임의로 뽑은 두 world 사이 오차율을 뜻하지 않는다.''',
tab(['고정 αc','서로 반대인 αf의 절댓값','최대 factual 차이','평균 noiseless probe L2 차이','Query label 변경'],[[z['alpha_c'],z['opposite_alpha_f'],f"{z['max_factual_error']:.2e}",f"{z['mean_probe_l2']:.4f}",pct(z['query_label_flip'])] for z in tw]),
'이어 실제 test의 동일한 x와 동일 numerical seeds에서 두 privileged K256에 서로 반대의 αf를 제공했다. 아래는 각 방향·replicate를 같이 평균한 탐색적 수치다. 오른쪽 열은 실제 환경의 known decoder로 양쪽 action을 채점한 값이다.',
tab(['(αc, 반대 방향의 αf)','평가 worlds','두 known-prior 선택 불일치','반대 αf의 action을 사용한 Brier 증가'],[[f'({ac:g}, ±{af:g})',sum(j['n'] for j in group),pct(mean(j['action_disagreement'] for j in group)),sig(mean(j['opposite_minus_actual_common_loss'] for j in group))] for ac,af in ((-.65,.85),(.35,.65),(-.2,.35),(0.,1.)) for group in [[j for j in pairs if j['alpha'][0]==ac and abs(j['alpha'][1])==af]]]),
f"총 {sum(j['n'] for j in pairs):,}개 test worlds에서 같은 factual input에 대해 known-prior 선택이 {pct(mean(j['action_disagreement'] for j in pairs))} 달랐다. 실제 환경 대신 반대 방향으로 관측을 고른 뒤 실제 환경 decoder로 채점하면 평균 Brier가 {sig(mean(j['opposite_minus_actual_common_loss'] for j in pairs))} 변했다. 이는 이 finite K256에서의 측정값이고, 최적 decision regret에 대한 보편적 lower bound는 아니다. [쌍을 이룬 물리 worlds](results/twins.json) · [동일 input의 action 검증](results/twin_actions.json)",
'## 7. Calibration worlds를 16배로 늘린 보조 실험',
'''(−0.65, ±0.85), (+0.35, ±0.65)의 네 환경만 사용했다. 12개 cohort의 test worlds 1,536개와 bank를 그대로 두고 독립 calibration 16,384개를 새로 생성했다. 따라서 이 수치는 주 실험 8개 환경 평균과 직접 비교할 수 없다. αc만 정확히 알고 αf를 0으로 둔 privileged control도 평가했다.''',
tab(['정책','Native Brier','같은 known decoder Brier'],[[n,f(ex['loss'][i]),f(ex['loss'][i+5])] for i,n in enumerate(['Estimated (1,024)','Estimated (16,384)','Known profile only','Neutral','Known both'])]),
tab(['탐색적 대비','Native Δ [동시 95%]','공통 decoder Δ [동시 95%]'],[[n,delta(ex['deltas'][j],ex['sim95'][j]),delta(ex['deltas'][j+5],ex['sim95'][j+5])] for j,n in enumerate(['16,384 − 1,024','16,384 − neutral','Known profile − neutral','16,384 − known profile','Known profile − known both'])]),
f"Profile MAE는 {ex['maes'][0]:.3f}에서 {ex['maes'][1]:.3f}로 줄었다. 큰 batch fit에는 평균 {ex['fit_ms']:.1f} ms가 들었다. 같은 test에서 native Brier도 0.11322에서 0.10991로 내려갔으나 paired interval은 0을 포함했다. Profile을 정확히 알려 줘도 αf까지 아는 comparator보다 native loss가 +0.00933 높았다. 알려진 family 안에서 한 축을 더 잘 식별하는 일과 전체 decision loss를 낮추는 일을 별도로 검증해야 한다. [보조 실험 코드](context27.py) · [같은 test의 비교 결과](results/extended.json)",
'![Native and common-decoder risk, GPU latency and context adaptation](results/v27.png)',
'## 8. 논문 기여에 대한 판단',
'''**현재 자료로 주장할 수 있는 기여는 어떤 population shift가 관측 선택에 필요하면서도 factual evidence에서 보이지 않는지 실측한 것이다.** Profile은 고차 관계를 통해 약하게 보인다. Friction은 대칭성 때문에 완전히 가려진다. 실제로 동일한 factual input을 받는 known-prior K256 두 개에서 action이 달라졌고, 반대 환경의 action에는 양의 평균 비용이 있었다. Source-only 정책과 비교해 selection 자체가 개선된다는 점도 공통 decoder 대조에서 확인했다.

**새 적응 방법이 neutral policy보다 낫다는 주장은 보류한다.** Neutral 대비 Brier interval이 0을 포함하고, 원 배포 환경에서는 적응에 비용이 있었다. 또한 오랜 관측이나 다른 instance를 사용해 다음 design을 바꾸는 방법과 decision-aware experimental design은 이미 연구됐다. 이 결과는 하나의 analytic simulator, 한 sensor calibration, 두 actions, 알려진 parametric shift family에 한정된다. [Huang et al.](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) · [Zhao et al.](https://arxiv.org/html/2611.11512v1)

**다음으로 판별력이 높은 실험은 과거 world 하나에 유료 calibration probe를 허용하는 것이다.** 그 관측이 현재 world에는 주어지지 않도록 나눈 뒤, probe 0·1·4·16회에서 αf posterior와 새로운 world의 Brier를 측정한다. 같은 sensing budget의 random calibration, neutral K256, known-prior K256을 포함하고, unseen contact law에서 반복한다. 추가 측정으로 αf의 방향과 neutral 대비 selection 이득을 함께 회복하는지, probe 시간을 상쇄할 만큼 환경이 오래 유지되는지를 별도 평가해야 한다. 이것이 입증되면 식별 한계 진단을 넘어 구체적인 observation policy 기여로 발전할 여지가 있다.''',
(root/'LITERATURE_KO.md').read_text(),
'## 재현 자료',
'''서버 프로젝트는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/population_shift_v27`이다. GPU guard를 이용해 runner27.py, context27.py, twin_actions27.py를 각각 별도 worker directory에서 같은 GPU에 순차 실행했다. CPU에서 twins27.py, analyze27.py, diagnostics27.py, extended_stats27.py, report27.py를 실행했다. 각 cohort에 calibration/test factual inputs, 동결된 선택의 SHA-256, 두 potential observations에 대한 예측값·Brier, timing samples, numerical seeds를 보존했다. [사전 설정](design.json) · [paired bootstrap](analyze27.py) · [동일 input 검산](twin_actions27.py) · [large-context 통계](extended_stats27.py) · [primary 수치](results/summary.json) · [추가 GPU 종료 기록](context_worker/results/worker_exit.json) · [동일 input GPU 종료 기록](twin_worker/results/worker_exit.json)''']
fig,axs=plt.subplots(1,3,figsize=(13.8,4.6),layout='constrained')
labels=['Estimated','Frozen gate','Source K256','Known K256','Always-query','Neutral K256']
y=np.arange(6)
a=axs[0];a.barh(y-.15,p['native'],.3,label='Native',color='#275d85');a.barh(y+.15,p['common'],.3,label='Common decoder',color='#81b5a2');a.set(yticks=y,yticklabels=labels,xlabel='Held-out Brier loss',title='Eight unseen shift laws');a.invert_yaxis();a.legend(fontsize=8)
a=axs[1];a.barh(y,p['p90'],color='#69819a');a.set(yticks=y,yticklabels=labels,xlabel='Milliseconds',title='Warm end-to-end GPU p90');a.invert_yaxis()
a=axs[2]; pts=np.arange(5)
a.scatter(pts,ex['loss'][:5],label='Native',color='#275d85');a.scatter(pts,ex['loss'][5:],label='Common decoder',color='#529581',marker='x');a.set(xticks=pts,xticklabels=['1,024','16,384','Known c','Neutral','Known both'],ylabel='Held-out Brier loss',title='Same four laws, larger context');a.tick_params(axis='x',rotation=30);a.legend(fontsize=8)
fig.savefig(root/'results/v27.png',dpi=155)
(root/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')
