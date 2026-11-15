"""Korean report and reproducible scientific figure from the completed v29 measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P/'results'
def read(k):return json.loads((R/(k+'.json')).read_text())
s=read('summary');diag=read('diagnostics');audit=read('score_audit');lat=read('quota_latency29')
supp=read('quota29/summary');charged=read('cost_with_setup')
cfg=json.loads((P/'design.json').read_text()); first=read('block_000/metrics')
def t(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'---|'*len(h)]+['| '+' | '.join(map(str,r))+' |' for r in rows])
def f(x):return f'{x:.5f}'
def sg(x):return f'{x:+.5f}'
def iv(x):return '['+', '.join(sg(z) for z in x)+']'
def withCI(x,ab):return sg(x)+' '+iv(ab)
def pc(x):return f'{100*x:.1f}%'
def law(n):
 x=next(e['alpha'] for e in cfg['environments'] if e['family']==n);return '('+', '.join(f'{v:g}' for v in x)+')'
names=['Adaptive stopping','Fixed 4','Random 2/4/8','Uncertainty stopping','Known-prior K256']
nblock=len(cfg['bank_pairs']);B=lat['metrics']['batch'];W=lat['metrics']['stream']
def tp(c):return f'{c:.3f}'
def tCI(v):return '['+', '.join(tp(i) for i in v)+']'
def ell(x):return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
parts=['# v29 · 추가 calibration probe를 어느 환경에서 계속할 것인가',
(P/'FINDINGS_KO.md').read_text(),
'## 1. 같은 예산을 어떻게 보장했나',
'''v28에서 16회의 calibration probe는 friction shift의 방향을 맞히고 최종 stick probability를 개선했다. 그러나 관측을 고르는 정책만 바꾸고 decoder를 고정하면 neutral 대비 Brier 이득을 입증하지 못했다. V29에서는 “다음 probe가 미래 world에서 query view와 probe의 순위를 바꿀 가능성이 큰 환경에 한정된 접촉을 몰아주면 fixed 4보다 좋은 선택을 하는가?”를 검증했다. 정답은 별도 충돌의 stick 여부이고, 평가는 선택한 관측을 한 개 받은 뒤의 Brier loss다. [v28 보고서](../calibration_probe_v28/REPORT_KO.md)

한 world는 곡률·마찰·반발계수가 숨겨진 하나의 충돌 장면이다. Query view는 다음 충돌 위치 부근의 정밀 boundary profile을 관측하고, probe는 별도의 입사속도로 미끄러짐을 일으킨 뒤 2차원 outgoing velocity를 측정한다. 두 관측 중 하나를 받아 다른 충돌이 stick할 확률을 예측한다. Brier loss는 이 확률과 실제 binary label의 차이를 제곱한 값이다. Calibration probe는 같은 물성 분포를 공유하는 과거 world에서 구입하고, test probe는 지금 예측하려는 world에서 구입한다. 환경의 αc는 profile offset과 curvature spread, αf는 velocity proxy와 friction spread 사이의 결합 강도와 방향을 나타낸다. [물리 모델](../within_calibration_v25/physics25.py) · [shift 생성](../proxy_shift_v26/physics26.py)

각 독립 block에서 8개 환경을 동시에 다뤘다. 환경마다 서로 독립인 1,024개 calibration worlds에서 factual boundary profile과 outgoing velocity를 받았다. 서로 다른 물체에 가하는 추가 calibration probe를 무작위 순서로 한 개씩 공개했다. 모든 학습 가능 정책은 한 block에 정확히 **32회**를 쓴다. Fixed 4는 8곳에 4회씩 배정한다. Adaptive는 8곳에 2회씩 사용하고 점수가 높은 4곳을 남겨 각각 2회 추가한다. 네 곳을 다시 평가해 그중 두 곳에 4회씩 더 쓴다. 최종 배분은 2회인 곳 4개, 4회인 곳 2개, 8회인 곳 2개다. 중단한 환경도 똑같은 가중치로 최종 평가에 남는다. 한 환경을 중단하고 예산을 절약하는 단일 환경 정책과는 운영 조건이 다르다.

같은 2/4/8 배분을 무작위로 섞은 정책, friction coupling αf의 posterior 표준편차가 큰 곳을 남기는 정책을 추가했다. Known-prior K256은 환경의 실제 αc와 αf만 아는 privileged comparator이며 calibration을 하지 않는다. 모든 방법이 개별 평가 world의 물성·정답과 선택하지 않은 관측을 볼 수 없도록 했다. 먼저 예산과 단계별 의사결정을 SHA-256으로 고정하고, 그 뒤에만 가능한 모든 prefix(2·4·8회)의 가상 정책을 평가용으로 구성했다. Test action을 다시 hash로 동결한 다음 label과 potential observation을 열었다. [사전 protocol](PROTOCOL.md) · [실험 설정·seeds](design.json) · [GPU runner](runner29.py)

V28의 8개 shift law를 재사용하고 **24개 새 독립 blocks, 192개 환경 cohort, 24,576개 held-out worlds**를 생성했다. 환경마다 128개 새 test worlds와 각 action의 observation noise 32회 반복을 사용했다. V29는 알려진 물리 family 안의 같은 shift panel에 대한 새 표본 검증이다. 동일한 contact law와 sensor calibration을 사용했다. 정책에는 숨은 환경 이름도 주지 않았다. 경쟁하는 환경은 매 block에서 독립적으로 순서를 섞었다.''',
t(['운영 조건','Calibration probes / block','평가 관측과 test probes','해석'],[
['A · Streaming','32','1,024 worlds 각각에서 query view 또는 probe를 자유 선택','Calibration 예산 일치. 실제 test probe 수 별도 보고'],
['B · Batch quota','32','128개 factual inputs가 모이면 환경당 정확히 13회 probe, 나머지 115회 query','총 136 probes/block. 두 정책 모두 동일한 batch에 접근']]),
'''조건 B에서는 각 환경 안에서 모든 test world의 risk(query)−risk(probe)를 정렬해 가장 높은 13개에 probe를 배정했다. 양의 gap이 13개보다 적어도 13회에 맞춘다. 한 방법에만 미래 factual inputs를 허용하는 비교를 피했다. 이 조건은 128개 모두의 factual evidence가 도착한 후 처리해야 하므로, 단일 world가 즉시 도착하는 streaming latency와 구별해야 한다.''',
'## 2. 다음 probe의 가치를 어떻게 계산했나',
'''각 환경의 posterior는 v28의 paired likelihood를 사용한다. Factual만 관측한 world에는 p(x|αc), 이미 calibration probe를 산 world에는 p(x,y|αc,αf)를 한 번 적용한다. 추가 probe는 query 위치의 geometry와 contact law에 대한 현재 믿음을 바꾸고, 그로 인해 미래 world에서 선호하는 observation이 달라질 수 있다.

평가 데이터와 별도로 만든 **96개 reference factual worlds**에 대해 5×9개 coupling grid에서 K256으로 두 action의 Brier risk 차이 g=R(query)−R(probe)를 계산했다. 환경을 계속할 때 2회 또는 4회 뒤 가능한 미래를 예측하기 위해 **256개의 독립 simulator probe sequences**를 고정했다. 각 sequence의 측정값에 posterior predictive weight를 주고, 추가 관측 뒤의 coupling 평균에서 grid risk를 bilinear interpolation했다. 각 미래의 점수는 현재와 미래의 g 부호가 바뀐 reference world에서 |g_after|를 더한 값이다. 환경별 factual 분포와 미래 sequence 확률로 가중 평균해 큰 곳에 예산을 줬다. 따라서 단순히 많이 뒤집힐 곳보다는, 뒤집힐 때 틀린 선택의 예상 손실이 큰 곳이 높은 점수를 받는다. 두 stage에서는 각각 앞으로 살 2회와 4회 전체를 예측한다.

계산에는 근사가 있다. 실제 probe 결과를 받은 후의 estimator는 같은 calibration world에서 나온 x,y의 joint likelihood와 두 coupling의 posterior covariance를 보존한다. 미리 점수를 낼 때는 미래의 개별 factual x를 적분한 probe velocity likelihood를 쓰고, risk lookup은 두 축의 평균으로 근사한다. 점수는 K256 내부의 예상 손실로 정의됐고 held-out Brier 감소 자체를 보장하지 않는다. 추가로 산 probe가 두 방법의 실제 우열을 바꾸는지는 §5에서 평가했다. [점수 구현](policy29.py) · [posterior와 미래 가중치 직접 적분 검산](check_policy29.py)''',
'## 3. Held-out Brier와 실제 사용량',
t(['배분 정책','Streaming common Brier','Batch quota common Brier','Streaming native Brier','Batch quota native Brier','Streaming test probe 선택률','Streaming 총 probes / world¹'],[[n]+[f(s['risk'][k][i]) for k in range(4)]+[pc(s['probe'][i]),f"{s['total_stream'][i]:.4f}"] for i,n in enumerate(names)]),
'''¹ 총 probe 수는 calibration 횟수를 배포 world 수로 나누고 test probe rate를 더한 값이다. Query view 취득, factual 관측, 물체 reset은 probe 횟수로 세지 않았다. Native는 각 정책이 자기 posterior로 decoding한 결과다. 두 co-primary 비교는 동일한 known-prior decoder에서 같은 world·action에 같은 확률 예측과 Brier를 할당한다. 따라서 common 열에서 나타나는 차이는 observation 선택에서만 나올 수 있다. Known-prior K256도 finite bank를 사용하므로 정확한 Bayes lower bound는 아니다.''',
t(['사전 지정 대비 · Adaptive − Fixed 4','Δ common Brier','Simultaneous 95% interval','유리한 block 수','한 block씩 제외한 평균 범위','Fixed 4와 다른 action'],[[name,sg(s['common_delta'][i]),iv(s['sim95'][i]),f"{s['win_blocks'][i]}/24",iv([min(row[i] for row in s['LOO']),max(row[i] for row in s['LOO'])]),pc(s['switches'][i])] for i,name in enumerate(['Streaming','Batch quota'])]),
f"음의 차이는 adaptive에 유리하다. 두 대비에 대해 24개 독립 block 전체를 짝지어 10,000회 재추출하고, 표준오차로 나눈 최대 절댓값으로 simultaneous 95% interval을 만들었다. 경쟁하는 환경들은 예산을 공유하고, batch quota에서는 한 world의 순위가 같은 환경의 다른 world 선택에도 영향을 준다. 따라서 환경과 world를 제각각 독립 표본으로 세지 않았다. Streaming에서 block과 그 안의 world를 함께 재추출한 별도 sensitivity interval은 {iv(s['stream_hier_ci95'])}다. 사전 기준은 upper bound가 0 미만이면 우월, −0.003 미만이면 연구용 실질 개선이다. Interval이 0을 포함하면 동등성을 주장할 수 없다. [검산된 전체 수치](results/summary.json) · [통계 코드](analysis29.py)",
'## 4. 어디에 배정했으며 무엇을 얻었나',
t(['실제 (αc,αf) · 평가용','평균 배정 횟수','2 / 4 / 8회 배정 block 수','Streaming adaptive / fixed','Batch quota adaptive / fixed'],[[law(key),f"{v['budget']:.2f}",' / '.join(map(str,v['histogram'])),' / '.join(f(x) for x in v['common'][:2]),' / '.join(f(x) for x in v['quota_common'][:2])] for key,v in s['bylaw'].items()]),
'''Fixed 4와의 차이는 네 곳을 일찍 중단하며 생긴 변화와 두 곳에 더 많은 probe를 주며 생긴 변화로 정확히 나눌 수 있다. 두 곳은 예산이 4회로 같아 기여가 0이다. 아래 값은 8개 환경 전체에 대한 가중 기여다. 음수이면 손실이 줄었다.''',
t(['조건','조기 중단 (2−4) 기여 [95%]','계속 관측 (8−4) 기여 [95%]','균등 무작위 배분 기대 Brier','Adaptive − 무작위 기대 [95%]','Hindsight 배분 Brier'],[[name,withCI(s['audit'][j][2],s['allocation_parts_ci95'][j][2]),withCI(s['audit'][j][4],s['allocation_parts_ci95'][j][4]),f(s['audit'][j][0]),withCI(s['vs_expected_random'][j],s['vs_expected_random_ci95'][j]),f(s['audit'][j][1])] for j,name in enumerate(['Streaming','Batch quota'])]),
'''무작위 기대값은 각 환경에서 2·4·8회로 끝날 확률 1/2·1/4·1/4를 실제 측정한 세 potential losses에 곱했다. 이와 별도로 §3의 random 정책은 예산 순열을 한 번 실제 추출한 결과다. Hindsight는 가능한 420개 배분 중 같은 held-out outcomes에서 가장 낮은 loss를 골랐다. 재배분에 여지가 얼마나 있었는지 보는 낙관적 진단이며 배포 가능한 성능은 아니다. 각 표의 사후 interval은 별도 표기가 없으면 pointwise block bootstrap이다.''',
'## 5. 점수가 높은 환경에서 실제로 더 좋은 선택을 했나',
t(['Stage','예상 flip 비율','실제 flip 비율','Flip 순서 일치율 [95%]','Brier gain 순서 일치율 [95%]','계속−중단 실제 common Brier 이득 [95%]'],[[n,pc(v['forecast_flip']['mean']),pc(v['observed_flip']['mean']),ell(v['concordance_flip']),ell(v['concordance_common_gain']),withCI(v['continued_minus_stopped_common_gain']['mean'],v['continued_minus_stopped_common_gain']['ci95'])] for n,v in zip(['2→4, 전체 8곳','4→8, 남은 4곳'],audit['stages'])]),
'''같은 prefix에서 후보 환경 모두에 동일한 추가 예산을 가상 적용해 각 held-out world에서 실제 action이 바뀐 비율과 common-decoder Brier 개선을 측정했다. 순서 일치율은 같은 block 안에서 환경 둘을 비교해 순서가 맞으면 1, 동률이면 0.5다. 무작위 기준은 0.5다. 둘째 stage의 결과는 첫 stage를 통과한 환경에 조건부다. 점수가 flip을 예측해도 flip에 따른 이득의 크기와 부호까지 예측해야 재배분으로 평균 Brier를 낮출 수 있다. [Score audit](results/score_audit.json) · [세부 rank correlation과 구간](results/diagnostics.json)''',
 '### Batch quota의 경계에 맞춘 보조 실험',
 '주 정책은 probe가 query를 앞서는 지점을 예측한다. Batch quota로 13개를 선발할 때는 13등과 14등의 순위가 바뀌어야 예산 배정에 영향을 준다. 이를 직접 평가하는 보조 정책을 같은 24개 block에서 추가로 실행했다. 각 가상 미래의 factual 분포에 대해 예산 13/128에 해당하는 상위 reference worlds를 다시 찾고, 교체할 world 쌍의 예측 Brier 차이를 가중해 환경의 우선순위를 정했다. 경계에 걸린 마지막 reference world에는 fractional weight를 주어 모든 가상 시나리오에서 정확히 같은 예산을 사용했다. 정책 변경은 탐색적이다. 별도 tuning 없이 24개 block의 보조 배분을 모두 고정하고 test actions까지 동결한 후 평가했다. [보조 정책 소스](quota29.py) · [결과와 hash](results/quota29/summary.json)',
 t(['조건','Quota-aware adaptive','Fixed 4','주 adaptive','보조 − Fixed 4 [simultaneous 95%]','보조 − 주 adaptive [simultaneous 95%]'],[[label]+[f(z) for z in supp['risk'][i]]+[withCI(d,c) for d,c in zip(supp['deltas_vs_fixed_original'][i],supp['sim95'][i])] for i,label in enumerate(['Batch quota','Streaming'])]),
 f"계속할 환경의 기준을 바꾸자 원래 정책과 최종 calibration 횟수가 달라진 환경은 {pc(supp['mean_changes']['budget_diff_from_primary'])}, batch quota에서 다른 관측을 고른 worlds는 {pc(supp['mean_changes']['quota_action_diff_from_primary'])}였다. 환경당 보조 scoring 시간은 평균 {supp['score_ms_per_env']:.2f} ms였다. 이 보조 표의 4개 contrast에만 별도로 simultaneous interval을 계산했다. 동일한 held-out panel을 재사용했으므로 독립적인 confirmatory replication은 남아 있다.",
 '## 6. GPU p90, setup, 접촉 비용',
t(['Common-decoder 실측²','Adaptive p90','Fixed 4 p90','p90 ratio [pointwise 95%]','정책당 시료 수'],[[n,tp(v['p90_ms'][0])+' ms',tp(v['p90_ms'][1])+' ms',tp(v['adaptive_over_fixed'])+' '+tCI(v['ratio_ci95']),v['n_per_method']] for n,v in [('Streaming, 1 world',W),('Batch quota, 128 worlds 전체',B)]]),
'''² 같은 known decoder를 실제로 실행했다. 미리 정한 첫 12개 block에서 seed로 정한 환경 두 곳씩, 총 24개 환경을 사용했다. Streaming은 10개 worlds × 2회, batch는 전체 128개 factual inputs의 risk 계산·상위 13개 정렬·선택한 observation의 decoding을 3회 실행했다. Warm-up 후 방법 순서를 무작위화하고 CUDA synchronization과 host 전송을 포함했다. 모든 action과 첫 noise draw의 예측 확률을 저장 결과와 검산했다. Batch p90은 128개 전체의 wall time이다. 이를 128로 나눈 throughput은 한 world의 응답 지연과 다르다. Interval은 3,000회 paired bootstrap에서 산출했다. [GPU timing 원수치](results/quota_latency29.json) · [측정 코드](quota_latency29.py)''',
t(['환경당 calibration 단계 (ms)','Adaptive','Fixed 4','무작위','Uncertainty'],[[label]+[f"{v:.2f}" for v in s['setup_means'][k][:4]] for k,label in [('fit','Posterior fit'),('score','중단 점수 계산·정렬'),('engines','최종 K256 bank 생성')]]),
f"96개 reference worlds의 risk table 및 미래 probe library를 처음 구성하는 데 {first['setup_ms']['scorer_static']/1000:.2f}초가 걸렸다. 이는 환경마다 되풀이할 필요가 없는 1회 비용이다. Posterior estimator의 공통 초기화는 {first['setup_ms']['detector_static']:.1f} ms였다. 위 setup은 실제 배포 경로가 사용하는 stage 계산과 bank 생성에 실측 시간을 배분한 값이다. Prefix가 겹치는 실험 정책들은 같은 계산을 재사용하고 경로별 비용을 별도 산정했다. 실제 장비의 충돌·reset·관측 대기 시간 및 접촉의 금전적 단가는 측정하지 않았다. 두 정책이 같은 수의 calibration probes를 구매해도 adaptive는 응답을 받고 다음 2→4→8 단계를 선택해야 한다. 물리 probes의 단계별 대기 시간과 병렬성 때문에 실물 wall time이 같다고 단정할 수 없다.",
t(['환경별 setup을 포함한 비용','Adaptive p90','Fixed 4 p90','Ratio [95%]'],[[label,tp(charged[key]['p90_ms'][0])+' ms',tp(charged[key]['p90_ms'][1])+' ms',tp(charged[key]['ratio'])+' '+tCI(charged[key]['ratio_ci95'])] for key,label in [('stream','Streaming · 128개 결정에 균등 배분'),('batch','Batch quota · 환경당 128개 전체')]]),
 '각 timing 대상 환경의 posterior fit, stopping score, 최종 selector bank 생성 시간을 합산했다. Streaming은 환경당 128개 결정으로 나눈 상각값, batch는 setup 전액을 더한 값이다. 실제 매번 발생하는 단일 요청 대기시간과 상각 cost는 구별해야 한다. 두 방법에 공통인 decoder 초기화, 환경 전체에 공유하는 static table 생성, 실제 센서 대기시간은 이 표에서 제외했다. [상각 계산과 구간](results/cost_with_setup.json)',
 '![Common-decoder Brier and calibration allocation](results/v29.png)',
'## 7. 논문 기여 가능성과 다음 검증',
(P/'ASSESSMENT_KO.md').read_text(),
(P/'LITERATURE_KO.md').read_text(),
'## 재현 자료',
'''서버 작업 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_stopping_v29`이다. 세 allocation 중 다른 작업이 없던 node55의 RTX PRO 6000 Blackwell Server Edition 한 장에서 GPU guard 아래 순차 실행했다. 각 block에서 allocation과 선택의 hash, paired outcomes, 2·4·8회 각각의 risk/action, GPU timing을 보존했다. 수치 분석은 할당된 compute node의 CPU에서 수행했다. [사전 protocol](PROTOCOL.md) · [실험 설정](design.json) · [대표 block의 예산 동결](results/block_000/allocation_lock.json) · [주 실험 종료 기록](gpu_worker/results/worker_exit.json) · [timing 종료 기록](timing_worker/results/worker_exit.json) · [보조 실험 종료 기록](quota_aware_worker/results/worker_exit.json)''']
fig,axes=plt.subplots(1,3,figsize=(14,4.55),layout='constrained')
axis=axes[0]
for i,(label,color) in enumerate([('Streaming','#21638b'),('Batch quota','#cd783b')]):
    pt=s['common_delta'][i];lo,hi=s['sim95'][i]
    axis.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',capsize=5,color=color)
axis.axvline(0,linestyle='--',color='grey');axis.set(yticks=[0,1],yticklabels=['Streaming','Batch quota'],xlabel='Adaptive minus fixed Brier',title='Common decoder, 95% intervals',ylim=(-.7,1.7))
axis=axes[1];ls=list(s['bylaw']);yy=np.arange(len(ls))
axis.barh(yy,[s['bylaw'][k]['budget'] for k in ls],color='#5e9c89');axis.axvline(4,ls='--',color='black');axis.set(yticks=yy,yticklabels=[law(k) for k in ls],xlabel='Mean calibration probes',title='Spend by shift law',xlim=(0,8))
axis=axes[2];xx=np.arange(2)
for j,(n,k) in enumerate([('Expected flip','forecast_flip'),('Observed flip','observed_flip')]):
    axis.bar(xx+(j-.5)*.32,[v[k]['mean'] for v in audit['stages']],width=.32,label=n,color=['#658eac','#e1a060'][j])
axis.set(xticks=xx,xticklabels=['2 to 4','4 to 8'],title='Action switches after probing',ylabel='Fraction of held-out worlds');axis.legend(fontsize=8)
fig.savefig(R/'v29.png',dpi=155)
(P/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')
