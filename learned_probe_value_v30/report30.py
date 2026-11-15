"""Build the Korean empirical report directly from verified v30 aggregates."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
def read(p):return json.loads((P/p).read_text())
s=read('results/analysis30.json')
t=read('results/timing_stats.json')
m=read('results/model_lock.json')
noise=read('source_target_noise.json')
ref=read('results/source_oof_policy.json')
parts30=read('results/allocation_parts.json')
half=read('results/half_replay.json')
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'---|'*len(head)]+['| '+' | '.join(map(str,row))+' |' for row in rows])
def f(v):return f'{v:.5f}'
def sg(v):return f'{v:+.5f}'
def ci(ab):return '['+', '.join(sg(z) for z in ab)+']'
def pct(v):return f'{100*v:.2f}%'
def fmt(x,d=3):return f'{x:.{d}f}'
def metric(delta, bounds):return f'{sg(delta)} {ci(bounds)}'
laws=['Static–kinetic split','Speed weakening']
names=['Learned gain','Fixed 4','Uncertainty-only']
rows=[]
for d,i,label in [(0,0,'Batch · source decoder'),(0,1,'Batch · law-aware decoder'),(1,0,'Streaming · source decoder'),(1,1,'Streaming · law-aware decoder')]:
    point=s['pooled_delta'][d][i]; ab=s['primary']['sim95'] if d==0 and i==0 else s['pooled_pointwise95'][d][i]
    rows.append([label,*map(f,s['risk_pooled'][d][i]),metric(point[0],ab[0]),metric(point[1],ab[1])])
laws_table=[]
for j,label in enumerate(laws):
    for d,kind in enumerate(['Source','Law-aware']):
        delta=s['law_delta'][j][0][d];bounds=s['law_pointwise95'][j][0][d]
        laws_table.append([label,kind,*map(f,s['risk_by_law'][j][0][d]),metric(delta[0],bounds[0]),metric(delta[1],bounds[1])])
cost_rows=[]
for k,lbl in [('stream_warm','Streaming 1개 · warm'),('batch_warm','Batch 128개 · warm'),('stream_amortized','Streaming 1개 · setup/128 합산'),('batch_with_setup','Batch 128개 · 전체 setup 합산')]:
    e=t['endpoints'][k]
    ratios=[f"{x:.3f} [{a:.3f}, {b:.3f}]" for x,(a,b) in zip(e['learned_over_baseline'],e['ratio_pointwise_ci95'])]
    cost_rows.append([lbl,*[fmt(v) for v in e['p90_ms']],*ratios,e['n_per_method']])
score_rows=[]
for y,label in enumerate(['2→4 · 후보 전체','4→8 · 첫 단계 생존 환경']):
    p=s['stages']['pooled'][y];b=s['stages']['pooled95'][y]
    score_rows.append([label,f(p[0]),f(p[1]),f(p[2]),f(p[3]),f(p[4]),f"{p[5]:.3f} [{b[5][0]:.3f}, {b[5][1]:.3f}]",metric(p[6],b[6])])
parts=['# v30 · 추가 calibration probe의 Brier gain을 학습하면 새 contact law에도 전이되는가',
(P/'FINDINGS_KO.md').read_text().strip(),
'''## 1. 서베이가 바꾼 실험 질문

네 연구자의 최근 약 10년 대표작을 읽고, 네 개의 citation 경로를 두 단계씩 추적했다. Yale Song의 연구에서 *도움이 필요해 보이는 상태*와 *실제로 도움이 되는 행동*을 구분하는 질문을 가져왔다. Seon Joo Kim의 연구에서는 training과 streaming에서 들어오는 정보의 차이, Seungryong Kim의 연구에서는 internal score가 실제 대응 관계를 보존하는지 독립적으로 확인하는 방법, Lingjie Liu의 연구에서는 한 scene을 설명하는 모델이 새로운 힘에서도 맞는지 검증하는 방법을 가져왔다. 이 네 의견은 논문을 읽고 구성한 가상 지도 의견이다. 연구 질문·방법·표본·결과·한계 비교표, 상충하는 결과의 해석, 10개 후속 질문과 출처는 별도 서베이에 담았다. [네 연구자 서베이 및 방법론 비교](SURVEY_KO.md)

가장 직접적인 선례는 LAL의 error-reduction regression이다. Action-BED는 acquisition과 decision의 co-adaptation까지 교차 평가했고, PhysiCalWorld는 같은 scene에서 과거 interaction을 재사용했다. 이 연구가 차별화될 수 있는 지점은 **같은 표면 분포의 과거 물체에 접촉해서, 나중에 만난 다른 물체에서 어떤 sensor를 쓸지 더 잘 선택하는가**다. 이를 budget, decoder, hidden contact law를 통제해서 묻는다. [LAL](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) · [Action-BED, Appendix G](https://arxiv.org/html/2606.23662v1) · [PhysiCalWorld, §§3–4](https://arxiv.org/html/2610.30125v1)

## 2. 학습 target, 예산과 평가 분할

한 world는 geometry·friction·restitution이 숨은 충돌 장면이다. 새 world에서는 정밀 boundary profile을 읽는 query view 또는 별도 충돌의 속도를 읽는 probe 한 개를 골라 다른 충돌의 stick probability를 예측한다. Calibration probe는 같은 물성 분포의 다른 과거 world에 가한다. 최종 label은 simulator에서 계산한 실제 binary stick event이며 Brier loss는 예측 확률과 event의 차이 제곱이다. 각 action에 대한 32개 sensor noise draw를 공유했다. 실물 접촉에서 얻은 label은 없다.

**무엇을 학습했나.** V29의 Coulomb law에서 생성한 24개 independent blocks를 v30의 development data로 재정의했다. 과거 v29에선 평가에 쓰인 데이터이며 v30의 신규 평가와 분리했다. 환경 192곳 각각에서 probe가 2→4회 또는 4→8회로 늘었을 때 같은 128개 미래 worlds의 선택을 다시 계산했다. 한 환경마다 가장 큰 risk gap을 가진 13곳에 test probe를 배정했다. 같은 decoder로 평가한 이전 prefix의 Brier에서 이후 prefix의 Brier를 뺀 **signed gain**을 회귀했다. 관측을 더 받고 성능이 떨어지면 음의 target 그대로 남겼다. 이 target은 실제로 실현된 하나의 calibration future와 하나의 미래 batch에서 측정한 값이다. Scorer의 예측 대상은 그 조건부 평균이다.

같은 world의 binary label을 y, 추가 calibration 전후의 선택한 관측으로부터 얻은 확률을 p_old, p_new라 하면 관측 한 쌍의 Brier 감소량은 (p_old−p_new)(p_old+p_new−2y)이다. Action이 같으면 common decoder의 예측도 같아 이 값은 0이다. Action이 달라져도 새 예측이 실제 y에서 멀어지면 음수다. V29의 flip score에 없던 부호가 여기에 들어간다.

각 stage에서 별도 ridge regression을 학습했다. 입력 15개는 paid prefix의 coupling posterior 평균·표준편차·상호작용과 측정한 probe 속도 요약, factual/probe 속도 공분산이다. 현재까지 구입한 2개 또는 4개의 응답만 입력한다. 각 fold의 feature 표준화와 ridge penalty 선택은 훈련 blocks 안에서 처리했다. 외부 6-fold nested grouped validation으로 source 성능을 추정하고, 전체 24개 source blocks에서 정해진 penalty로 최종 계수를 동결했다. Test contact law, 환경 이름, 현재까지 구입하지 않은 probe, 미래 stick label은 score 입력에 들어가지 않는다. [구현과 features](score30.py) · [동결된 계수와 OOF predictions](results/model_lock.json)

**무엇을 같게 했나.** 한 allocation block에 독립 환경 8곳을 두고, 환경별로 factual worlds 1,024개를 먼저 본다. Learned와 uncertainty-only는 처음 2회씩 접촉한 뒤 네 곳에 2회, 그중 두 곳에 4회를 더 쓴다. 후자는 friction coupling posterior의 표준편차로 순위를 정한다. Fixed 4는 여덟 곳에 4회씩 쓴다. 세 방법 모두 calibration 32회와 quota test probe 104회, 합계 136회다. 중단한 환경도 평가에서 제외하지 않는다. 두 test law에 16개 independent blocks씩, 총 256개 환경·32,768개 held-out worlds를 평가했다. 각 block마다 새로운 수치 bank pair를 사용했고 source와 숫자 seeds가 겹치지 않는지 검산했다. 128개 세계가 동시에 모이는 batch quota가 primary다. 자유 선택 streaming에서는 calibration 32회만 같고 실제 test probe 수는 별도로 센다. [설정](design.json) · [사전 분석 계획](ANALYSIS_PLAN.md)

**어떤 decoder로 채점했나.** Primary common decoder는 각 환경의 실제 population coupling을 입력받지만 새 contact law의 ID와 개별 물체 물성은 모른다. 모든 정책에 동일한 source Coulomb likelihood·source coupling-conditional label model을 적용하고 실제 target-law label로 채점한다. 보조 평가에서는 관측 선택을 그대로 고정한 채 해당 law를 아는 matched decoder로 다시 채점한다. 두 decoder 모두 population 정보를 받는 privileged comparator다. Law-aware는 selector나 scorer를 재학습시키지 않는다. Allocation을 먼저 hash로 동결하고, test factual을 받은 뒤 observation 선택도 동결한 다음, 가능한 관측과 label을 evaluation에만 공개했다. [GPU runner](runner30.py) · [contact law와 decoder](contact30.py)''',
'## 3. Contact law가 바뀐 것을 무엇으로 확인했나',
'''Source Coulomb model의 friction coefficient를 μ, 접선 속도를 vₜ라 한다. 첫 law에서는 stick threshold에 1.20μ, 미끄러질 때의 impulse에 0.80μ를 썼다. 두 번째에서는 q=exp(−|vₜ|/0.45)로 두고 각각 μ(1+0.22q), μ(0.64+0.25q)를 썼다. 비교할 때 geometry, restitution, population panel, noise 모델을 유지했다. 같은 물체의 factual 충돌은 stick하므로 두 law에서 같은 관측을 준다. 충분한 접선 속도를 가한 probe는 slip하고, query 위치의 다른 충돌은 stick/slip 경계가 이동한다. 아래는 같은 숨은 물체에서 law만 source로 되돌린 paired 진단이다. Δ velocity는 noise를 더하기 전 2차원 차이의 L2 norm 평균이다.''',
table(['Target law','Target stick','같은 물체의 source stick','Stick label 변경','Factual Δ','Probe Δ'],[[law,*[pct(v) for v in row[:3]],*[f(v) for v in row[3:]]] for law,row in zip(laws,s['physical']['law'])]),
(P/'SHIFT_AUDIT_KO.md').read_text().strip(),
'## 4. Held-out Brier · 관측 선택 효과',
table(['운영 조건·decoder','Learned','Fixed 4','Uncertainty-only','Learned − Fixed 4 [95%]','Learned − Uncertainty [95%]'],rows),
'''차이가 음수이면 learned에 유리하다. 두 law의 평균에 동일한 가중치를 주었다. 서로 다른 seeds로 생성한 두 law 각각에서 paired block 전체를 10,000회 재추출했다. 첫 행은 두 대비에 대한 simultaneous 95% interval, 나머지는 pointwise 95% interval이다. Noise 32회를 독립적인 labels로 세지 않았다. 두 law는 고정한 두 사례이며 새로운 contact law 전체로 interval을 일반화할 수 없다.''',
table(['Law · batch quota','Decoder','Learned','Fixed 4','Uncertainty-only','Learned − Fixed 4 [95%]','Learned − Uncertainty [95%]'],laws_table),
f"각 stage의 예산 histogram을 같은 확률로 무작위 배정했을 때의 정확한 기대 Brier는 {f(s['random_pooled'][0][0])}였다. Learned와 이 기대값의 차이는 {metric(s['vs_random']['mean'],s['vs_random']['interval'])}다. 자유 선택 streaming에서 learned, fixed 4, uncertainty가 실제로 probe를 고른 비율은 각각 {', '.join(pct(z) for z in s['probe']['pooled'])}였다. Batch에서 learned가 고른 관측이 fixed 4, uncertainty의 선택과 다른 비율은 {', '.join(pct(z) for z in s['switches']['pooled'][0])}였다. [전체 risk·비교 수치](results/analysis30.json)",
'아래 값은 고정 4회에서 일찍 멈춘 환경의 손익과, 8회까지 계속한 환경의 손익을 여덟 환경 전체에서 평균한 기여분이다. 두 행을 합치면 primary 차이가 정확히 복원된다. 어떤 환경이 일찍 중단되는지 정책마다 달라지는 점을 포함한다.',
table(['환경별 재배분','Learned − Fixed 4의 기여 [pointwise 95%]','Learned − Uncertainty의 기여 [pointwise 95%]'],[[k]+[metric(v,b) for v,b in zip(parts30['pooled_mean'][j],parts30['ci95'][j])] for j,k in enumerate(['2회에서 중단','8회까지 관측'])]),
'[재배분 기여분 검산](results/allocation_parts.json)',
'같은 world에서 query view만 또는 probe만 받았다고 가정한 두 potential predictions의 Brier도 따로 계산했다. 아래 표에서는 관측 종류를 고정했으므로 선택 정책의 영향 없이 decoder mismatch가 어느 측정값 해석에 집중되는지 볼 수 있다. 이 두 행동은 동일 quota로 경쟁하는 정책이 아니므로 primary 순위에 넣지 않았다.',
table(['Target law','Source decoder · query','Source decoder · probe','Matched decoder · query','Matched decoder · probe'],[[name]+[f(v) for decoder in row for v in decoder] for name,row in zip(laws,parts30['potential_action_brier'])]),
'## 5. 직접 학습한 gain은 얼마나 예측 가능한가?',
table(['Source · stage','환경 수','Ridge penalty','OOF gain correlation','OOF MSE','Train-fold mean MSE','Split-half gain rank correlation [95%]'],[[k,rec['n'],rec['chosen'],fmt(rec['oof_corr']),f"{rec['oof_mse']:.7f}",f"{rec['oof_baseline_mse']:.7f}",f"{no['split_half_spearman']:.3f} [{no['interval'][0]:.3f}, {no['interval'][1]:.3f}]"] for k,rec,no in zip(['2→4','4→8'],m['training_diagnostics'],noise.values())]),
f"예측값을 실제 배분에 적용한 비교도 source에서 확인했다. 각 과거 block에 대해 그 block을 통째로 제외하고 학습된 두 stage의 OOF predictions로 2/4/8회 배분을 재구성했다. Source batch Brier는 learned {f(ref['risk'][0])}, fixed 4 {f(ref['risk'][1])}, uncertainty-only {f(ref['risk'][2])}였다. 따라서 unseen law에서 차이가 나더라도 그 원인을 전부 contact-law shift에 귀속할 수 없다. Fold 간 학습 표본이 겹치고 별도의 source replication이 없어 이 비교는 기술적 참고값으로 제시한다. [OOF 정책 재생](results/source_oof_policy.json)",
'''Source target을 진단하려고 이미 동결된 128개 test actions를 유지한 채, 두 64-world 부분표본에서 gain을 따로 계산했다. 위 상관은 두 부분표본이 환경의 순서를 얼마나 일관되게 매기는지 나타낸다. 두 부분표본이 전체 batch의 13개 quota 결정과 하나의 calibration future를 공유하므로, 이는 새 calibration futures 간 신뢰도의 추정치도, 엄밀한 성능 상한도 아니다. [Split-half 원수치](source_target_noise.json) · [진단 코드](target_noise30.py)''',
f"128개의 world와 calibration future를 고정한 뒤 32회 sensor noise만 다시 측정했을 때의 환경별 gain 표준오차(RMS)는 2→4에서 {f(noise['2to4']['conditional_sensor_se_rms'])}, 4→8에서 {f(noise['4to8']['conditional_sensor_se_rms'])}이다. 이 진단은 기존 source data에서 계산했다. 같은 label을 공유하는 noise 반복을 32배 많은 독립 물체로 취급하면 표기되는 정밀도가 과장된다.",
table(['Unseen law · stage','예측 gain','실제 gain','예측 MAE','항상 0 MAE','Source mean MAE','환경 순위 일치율 [95%]','계속−중단 실제 gain [95%]'],score_rows),
'''Gain은 동일한 stage에서 추가 probe를 가상 구입한 환경의 Brier 감소량이다. 환경 둘의 실제 gain 순서를 맞히면 1, 동률은 0.5로 계산했다. 둘째 stage는 learned가 첫 stage에 남겨둔 네 환경을 평가한다. 같은 stage에서 큰 예상값을 고르는 정책은 모든 점수가 음수여도 정해진 예산을 사용한다. 이 예산 계약으로는 자유로운 조기 종료나 접촉 가격을 별도로 최적화했다고 볼 수 없다.''',
f"**사후에 가장 좋아 보이는 환경을 골라도 이득이 재현되는가?** 2·4·8회라는 같은 배정 분포를 지키는 420개 배분 모두를 평가했다. 128개 미래 물체의 정답을 전부 열고 가장 작은 Brier를 선택한 낙관값은 {f(half['pooled'][1])}이었다. 서로 다른 64개 물체 중 한쪽의 정답만 보고 배분을 정한 뒤 반대쪽에서 채점하고 역할을 바꾸자 {f(half['pooled'][2])}였다. Fixed 4와의 차이는 {metric(half['pooled'][3],half['intervals'][3])}다. 동일 factual batch의 quota와 calibration future를 공유하는 탐색적 진단이며, 배포 정책이나 독립적인 성능 상한이 아니다. [배분 재평가 원수치](results/half_replay.json)",
'## 6. Compute와 probe 비용',
table(['GPU p90 (ms)','Learned','Fixed 4','Uncertainty-only','Learned/Fixed [95%]','Learned/Uncertainty [95%]','정책당 시료 수'],cost_rows),
f"GPU는 {t['gpu']}였다. 각 law의 첫 8개 block에서 사전 seed로 지정한 환경 하나씩, 총 16곳에서 측정했다. Warm-up 뒤 방법 순서를 섞고 host 입력부터 선택·decoder·host 반환까지 CUDA synchronization을 포함했다. Timing 때의 action과 stick probability를 저장한 예측과 검산했다. Batch는 128건 전체의 wall time이고 streaming은 1건의 응답시간이다. setup은 경로별 posterior fit, scored allocation, selector bank 생성이다. Streaming의 setup은 128개 world로 상각했으며 batch에는 전액 가산했다. 95% interval은 같은 환경 내 paired timings을 보존한 law-stratified block bootstrap이다. Source 파일 로딩·validation·fit는 {t['offline_seconds_source_fit_and_loading']:.2f}초, 공통 estimator 초기화는 {t['shared_detector_static_ms']:.2f} ms였다. 과거 v29 label 생성, 공통 decoder 생성, 센서·물체 reset은 timing에서 제외했다. 따라서 이 표는 실물 end-to-end wall time을 뜻하지 않는다. [Timing 기록](results/timing_stats.json)",
'![학습한 probe value와 새 물리 법칙에서의 관측 선택](results/v30.png)',
'## 7. 논문으로 주장할 수 있는 범위와 다음 결정',(P/'ASSESSMENT_KO.md').read_text().strip(),
'''## 재현 자료

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/learned_probe_value_v30`이다. Source는 같은 프로젝트의 `adaptive_stopping_v29/results`를 사용한다. [GPU 실행 entry point](run_gpu.sh) · [사전 CPU 검산](preflight.json) · [GPU 완료 기록](results/completion.json) · [GPU guard 종료](gpu_worker/results/worker_exit.json) · [통계 재계산 코드](analyze30.py) · [GPU timing 분석](timing_stats30.py) · [9개 항목을 반영한 동반 서베이](SURVEY_KO.md)''']
fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
p=np.asarray(s['primary']['contrasts']);ab=np.asarray(s['primary']['sim95'])
axes[0].errorbar(p,[1,0],xerr=np.array([p-ab[:,0],ab[:,1]-p]),fmt='o',capsize=4,color='#286479')
axes[0].axvline(0,color='gray',linestyle='--');axes[0].set(yticks=[0,1],yticklabels=['vs uncertainty','vs fixed 4'],ylim=(-.55,1.55),title='Batch quota, source-law decoder',xlabel='Learned minus baseline Brier (95% simultaneous CI)')
p=np.asarray(s['stages']['pooled'])[:,5];ab=np.asarray(s['stages']['pooled95'])[:,5,:]
axes[1].errorbar(p,[1,0],xerr=np.array([p-ab[:,0],ab[:,1]-p]),fmt='o',capsize=4,color='#935719')
axes[1].axvline(.5,color='gray',linestyle='--');axes[1].set(yticks=[0,1],yticklabels=['4 to 8','2 to 4'],ylim=(-.55,1.55),title='Ranking true marginal gain by environment',xlabel='Pairwise concordance (95% pointwise CI)')
axes[0].ticklabel_format(axis='x',style='plain',useOffset=False)
fig.savefig(P/'results/v30.png',dpi=155)
(P/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')
