"""Generate the v28 report from final on-server measurements and analysis records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def read(file):return json.loads((root/'results'/(file+'.json')).read_text())
cfg=json.loads((root/'design.json').read_text());s=read('summary');p=s['pooled'];pp=read('power28');cs=read('controls_summary');mg=read('marginal_summary');dec=read('decomposition')
def tab(h,rows):
    return '\n'.join(['| '+' | '.join(h)+' |','|'+'---|'*len(h)]+['| '+' | '.join(map(str,r))+' |' for r in rows])
def f(x):return f'{x:.5f}'
def sg(x):return f'{x:+.5f}'
def pc(x):return f'{100*x:.1f}%'
def iv(z):return '['+', '.join(sg(t) for t in z)+']'
def ivm(v,z):return sg(v)+' '+iv(z)
bs=[0,1,4,16]
names=['0회','1회','4회','16회','Neutral K256','Known-prior K256']
def who(key):return next(e['alpha'] for e in cfg['environments'] if e['family']==key)
def lab(k):return '('+', '.join(f'{z:g}' for z in who(k))+')'
def con(i,j,common=False):
    k=s['pairs'].index([i,j])+(11 if common else 0)
    return ivm(s['deltas'][k],s['bands'][k])
def first(i,common):
    good=[bs[k] for k in range(1,4) if s['bands'][s['pairs'].index([k,i])+(11 if common else 0)][1]<0]
    return str(good[0])+'회' if good else '검출 없음'
def bothfirst(common):
    return next((str(bs[k])+'회' for k in range(1,4) if all(s['bands'][s['pairs'].index([k,j])+(11 if common else 0)][1]<0 for j in (0,4))),'검출 없음')
def pctstr(i):
    return 100*p['signacc'][i]
def power_for(a,b):
    return [row for row in pp['rows'] if abs(row['alpha_f'])==a and row['n_probes']==b]
def pw(a,b,name):
    rows=power_for(a,b);return float(np.mean([r['methods'][name]['sign_accuracy'] for r in rows]))
def oracle(a,b):return power_for(a,b)[0]['latent_bit_oracle_accuracy']
parts=['# v28 · Calibration probe가 velocity–friction shift와 다음 관측 선택에 주는 정보',
(root/'FINDINGS_KO.md').read_text(),
'## 1. 무엇을 검증했나',
'''v27에서는 모든 factual collision이 stick했다. 경계의 local slope가 s, restitution이 e일 때 outgoing velocity는 (−e/(1+s²), es/(1+s²))다. Friction coefficient는 관측식에서 빠지고, 두 friction spread 집단에서 restitution 중심을 서로 바꿔도 population의 factual evidence 분포는 같다. 그 때문에 factual worlds를 늘리는 것만으로는 velocity–friction 관계의 방향을 구별할 수 없었다. V28은 calibration 단계에서 다른 입사속도의 충돌을 일으켜 slip을 만들고, 그 반응이 **다른 미래 world**의 관측 선택에도 유용한지 묻는다. [v27 결과](../population_shift_v27/REPORT_KO.md) · [contact model](../within_calibration_v25/physics25.py)

αc는 profile offset과 curvature spread 사이의 정렬, αf는 restitution과 friction spread 사이의 정렬이다. α=1은 원래의 관계이고, 0은 독립, −1은 완전 반전이다. 넓고 좁은 두 집단의 조건부 중심을 반대 집단 값으로 바꿀 확률이 (1−α)/2다. 이번 실험은 αf의 부호와 크기가 다른 여덟 target laws에서 수행했다. 네 쌍은 αc를 고정하고 αf의 부호만 반전한다. 추가 controls는 (0,0)과 (1,1)이다. 같은 simulator와 sensor calibration을 사용했고 모든 physical worlds·calibration contexts·numerical bank seeds를 새로 뽑았다.

각 환경의 독립 batch 8개마다 factual worlds 1,024개를 먼저 관측했다. 미리 무작위로 뽑은 서로 다른 16개 world를 한 줄로 정렬하고 앞에서부터 0·1·4·16개까지만 probe response를 공개했다. 관측 하나는 두 성분의 noisy outgoing velocity다. 같은 물체를 여러 번 측정하지 않았다. 평가 worlds는 batch당 별도 128개다. 정책은 각 평가 world에서 정밀 query view와 probe 중 하나를 고르고, 그 관측을 사용해 다른 충돌의 stick probability를 예측한다. Brier는 이 확률과 binary outcome의 squared difference다. Target은 64개 독립 calibration cohort, 평가 8,192 worlds이며, controls를 포함하면 80개 cohort와 10,240개 평가 worlds다. 각 action의 observation noise를 32번 반복하고 모든 정책에서 같은 potential observations를 공유했다. [사전 protocol](PROTOCOL.md) · [실험 설정](design.json)''',
'## 2. 적은 수의 probe를 어떻게 posterior에 반영했나',
'''Factual-only world에는 p(x | αc)를, probe를 본 calibration world에는 같은 물체의 joint likelihood p(x,y_probe | αc,αf)를 한 번씩 적용했다. 두 가지 geometry 정렬 × 두 가지 friction 정렬로 얻은 4개 component의 likelihood를 사용했다. 8,192개의 별도 simulator 후보로 적분하고, 균일한 독립 hyperprior에서 401 × 201개 grid 지점으로 두 환경 계수의 joint posterior를 구했다. Probe가 0회이면 v27의 factual estimator를 그대로 호출한다. Friction 방향의 posterior probability는 이때 정확히 0.5다.

Calibration evidence로 두 계수가 종속될 수 있으므로 미래 world의 각 component 확률은 posterior에서 (1±αc)(1±αf)/4를 평균해 구했다. K256은 그 네 확률로 만든 inner/outer banks에서 query view와 probe의 risk를 비교한다. 모든 정책에서 같은 acquisition budget과 numerical seeds를 사용했다. Neutral K256은 (0,0)을, known-prior K256은 실제 환경의 두 α를 입력받는다. Known-prior도 개별 world의 숨은 상태나 참 label을 볼 수 없으며 finite-bank comparator다. 두 baseline은 사전 calibration probe를 쓰지 않는다. 각 cohort의 평가 action을 먼저 저장하고 hash를 계산한 뒤, 평가 label과 추가 관측을 생성했다. [Estimator](estimator28.py) · [GPU runner](runner28.py)''',
'## 3. Direction은 몇 번부터 보이는가',
tab(['Calibration probes','방향 정확도','동시 95% 범위¹','정답 방향에 준 평균 posterior probability','95% 이상 확신한 비율','고확신 오답 수 / 64','Friction MAE'],[[b,pc(p['signacc'][i]),'동률, 결정 보류' if i==0 else '–'.join(pc(max(0,min(1,t))) for t in s['signbands'][i-1]),pc(p['signprob'][i]),pc(p['confidence'][i]),p['wrong_conf_count'][i],f"{p['friction_mae'][i]:.3f}"] for i,b in enumerate(bs)]),
'''¹ 0회에서는 방향을 선택하지 않아 동률에 0.5점을 부여했다. 나머지는 posterior probability가 0.5를 넘은 쪽으로 판정했다. 8개 numerical bank/replicate blocks를 10,000회 재추출하고, 사전 지정한 1·4·16회의 정확도에 대해 simultaneous interval을 구했다. 64개 cohort마다 direction 예측은 하나다. 평가 world 수나 observation noise 반복 수를 direction 검정의 표본수로 세지 않았다.''',
'독립 power 실험에서는 αc를 알려 주고 4개 ±αf 쌍과 null에 각 256개씩 새 batch를 생성했다. 총 2,304개 batch이며 주 실험의 Brier 집계에 넣지 않았다. 각 칸은 **paired factual+probe / probe velocity만 / 잠재 정렬 bit를 완벽하게 읽는 oracle**의 방향 정확도다. 각 부호에 같은 가중치를 줬다. Oracle은 부호 후보의 크기를 안다고 가정한 기대 상한이다. 유한 Monte Carlo 관측값은 이론적 기댓값 주위에서 변동한다.',
tab(['αf 절댓값','0회','1회','4회','16회'],[[f'{a:.2f}']+[' / '.join(pc(v) for v in (pw(a,b,'factual_probe'),pw(a,b,'probe_only'),oracle(a,b))) for b in bs] for a in (.35,.65,.85,.92)]),
'실제 αf=0일 때 95%보다 높은 확률로 방향을 선언한 비율도 측정했다.',
tab(['Null에서 발생한 방향 경보','0회','1회','4회','16회'],[['주 실험 (8 batches)']+[pc(v) for v in s['laws']['null']['confidence']],['독립 검산 (256 batches)']+[pc(power_for(0,b)[0]['methods']['factual_probe']['confident_fraction']) for b in bs]]),
'구체적인 비교 대상은 1,024개 factual evidence만 있는 환경, noisy paired probe가 추가된 환경, probe 그 자체만 보는 환경이다. Probe에는 normal과 tangential response가 모두 들어 있어 단독으로도 정보를 가질 수 있다. [각 law별 Wilson interval 및 sign power](results/power28.json)',
'## 4. Held-out Brier와 observation selection',
tab(['정책','Native Brier ↓','공통 known decoder Brier ↓','Test에서 probe 선택','0회와 다른 선택','Known과 다른 선택'],[[n,f(p['native'][i]),f(p['common'][i]),pc(p['probe'][i]),pc(p['switch0'][i]),pc(p['diff_known'][i])] for i,n in enumerate(names)]),
'''Native는 각 방법이 자기 prior로 선택하고 자기 decoder로 예측한 값이다. 공통 decoder는 동일한 known-prior decoder가 각 방법의 선택을 다시 채점한 값이다. 두 정책이 같은 action을 고른 world에서는 공통 decoder loss도 동일하므로, 이 열의 차이는 관측 선택의 효과로 해석할 수 있다. 아래 Δ는 왼쪽 방법의 Brier에서 오른쪽 방법의 Brier를 뺀 값이다. 음수일수록 왼쪽이 유리하다.''',
tab(['대비','Native Δ [simultaneous 95%]','공통 decoder Δ [simultaneous 95%]'],[[names[i]+' − '+names[j],con(i,j),con(i,j,True)] for i,j in s['pairs']]),
f"사전 기준에 따라 0회와 neutral 양쪽보다 upper bound가 0 아래로 간 최초 budget은 native에서 **{bothfirst(False)}**, 공통 decoder에서 **{bothfirst(True)}**다. Grid 사이를 보간하지 않았다. 유의성은 실질적 차이와 구분한다. 기존 연구용 margin 0.003만큼의 실질 개선은 해당 대비의 upper bound가 −0.003보다 작은지를 기준으로 읽을 수 있다.",
'''동일 replicate block에 속하는 여덟 환경을 함께 재추출하고, 그 안의 test worlds를 방법 간 짝을 유지해 재추출했다. 10,000회 draw, cohort별 4,096개 bootstrap world-mean을 재사용했다. 사전 지정한 22개 native/common 평균 차이에서 표준오차로 나눈 최대 절댓값을 기준으로 구간을 만들었다. 수치는 여덟 target laws에 같은 가중치를 준 평균이며 새로운 shift family의 모집단에 대해 무작위 표집한 구간은 아니다. [원 통계](results/summary.json) · [통계 코드](analyze28.py)''',
 '### 선택과 decoder가 각각 설명하는 부분',
'각 world에서 두 정책이 고른 action을 두 정책의 decoder에 각각 넣었다. Action과 decoder를 바꾸는 두 순서에 대해 변화량을 평균한 Shapley decomposition이다. 성분의 합은 native Brier 차이와 정확히 같다. 다음 구간은 12개 탐색적 성분에 대한 동시 95% 범위다.',
tab(['대비','Action 성분','Decoder 성분'],[[names[i]+' − '+names[j],ivm(dec['deltas'][k],dec['sim95'][k]),ivm(dec['deltas'][k+1],dec['sim95'][k+1])] for j,off in ((0,0),(4,6)) for i in (1,2,3) for k in [off+2*(i-1)]]),
'16회−neutral의 native 차이 −0.00542 중 action 성분은 −0.00084, decoder 성분은 −0.00459였다. Decoder 성분의 구간은 [−0.00650, −0.00267]이었다. 두 정책이 같은 결정을 내려도 calibration이 예측 확률을 개선할 수 있음을 보여준다. 8개 replicate 중 하나씩 제외한 16회−0회 native 차이도 −0.00645에서 −0.00493이었다. [분해 수치](results/decomposition.json) · [계산 코드](decomposition28.py)',
'## 5. 어느 환경에서 결정이 달라졌나',
tab(['(αc, αf)','용도','0회','1회','4회','16회','Neutral','Known','공통 decoder 16회−0회'],[[lab(k),'target' if k in s['target'] else 'control']+[f(v) for v in s['laws'][k]['native']]+[sg(s['laws'][k]['common'][3]-s['laws'][k]['common'][0])] for k in s['laws']]),
'''보조 실험에서는 첫 세 replicate의 24개 target cohort에서 동일한 16회 calibration과 3,072개 test worlds를 다시 비교했다. Friction-only는 0회에서 얻은 profile marginal을 고정하고 16회의 friction marginal만 적용한다. Deranged pairing은 각 world에 다른 world의 probe response를 매칭해 원래의 paired likelihood를 적용한다. 후자는 marginal을 정확히 모델링한 probe-only estimator와 목적이 다르다.''',
tab(['동일 부분표본','Native Brier','공통 decoder Brier','방향 정확도'],[[n,f(cs['native'][i]),f(cs['common'][i]),pc(cs['sign'][i]) if i<4 else '—'] for i,n in enumerate(['0회','16회','Friction-only','Deranged pairing','Neutral K256','Known-prior K256'])]),
tab(['부분표본 대비','Native Δ [탐색적 동시 95%]','공통 decoder Δ [탐색적 동시 95%]'],[[n,ivm(cs['delta'][i],cs['sim95'][i]),ivm(cs['delta'][i+4],cs['sim95'][i+4])] for i,n in enumerate(['16회−0회','16회−friction-only','16회−deranged','Friction-only−0회'])]),
'부분표본에는 독립 blocks가 3개뿐이다. 작은 차이와 null result의 정밀도가 낮다. [대조군 구현](controls28.py) · [대조 실험 수치](results/controls_summary.json)',
 '### 같은 비용의 더 단순한 estimator',
f"16회 paired estimator와, probe를 받지 않은 1,008개 factual worlds에서 αc를 추정하고 나머지 16개 world의 probe velocity만으로 αf를 추정하는 marginal estimator를 같은 24개 cohort에서 비교했다. 후자는 두 marginal posterior를 곱한다. Paired와 marginal의 방향 정확도는 각각 {pc(cs['sign'][1])}, {pc(mg['sign'])}였다. Native Brier는 {f(mg['means'][0])} 대 {f(mg['means'][1])}, 차이 {ivm(mg['paired_minus_marginal'][0],mg['sim95'][0])}였다. 공통 decoder의 차이는 {ivm(mg['paired_minus_marginal'][1],mg['sim95'][1])}였다. 두 방법의 action은 {pc(mg['changed_actions'])}에서 달랐다. 이 결과와 probe-only power를 함께 보면 더 복잡한 paired estimator의 필요성을 별도로 입증해야 한다. 이 대조의 GPU p90은 측정하지 않았다. [Marginal baseline](marginal28.py) · [수치](results/marginal_summary.json)",
'## 6. GPU latency와 물리적 probe 비용',
tab(['정책','Warm GPU p90 (ms)','Neutral 대비 ratio [95%]','Calibration fit (ms)','Fit + bank setup (ms)','128 worlds에 setup 배분한 p90 (ms)','Calibration probes / 128 worlds','총 probe 횟수 / test world²'],[[name,f"{p['p90'][i]:.3f}",f"{p['p90'][i]/p['p90'][4]:.3f} [{s['timing_ratio95'][i][0]:.3f}, {s['timing_ratio95'][i][1]:.3f}]",f"{p['fit'][i]:.1f}",f"{p['setup'][i]:.1f}",f"{p['amortized'][i]:.3f}",bs[i] if i<4 else 0,f"{p['total'][i]:.3f}"] for i,name in enumerate(names)]),
f"² 총 probe 비용은 환경당 calibration B회를 H=128개 배포 worlds에 배분한 B/H에, test 시 probe 선택률을 더했다. Query view를 선택하면 추가 접촉은 0건이다. 모든 정책이 world마다 추가 observation 하나를 읽는다. Warm p90은 host input, acquisition, 선택한 observation의 GPU 전송 및 최종 prediction을 포함한다. 각 정책에 12 worlds × 2 repeats × 64 target cohorts = 1,536개의 batch-1 synchronized timings을 사용했다. 실행 순서는 world 안에서 무작위였다. Ratio는 1,500회 paired bootstrap의 pointwise 95% interval이다. Zero-probe와 positive-budget에서 posterior 적분 코드가 다르고 setup은 고정 순서로 측정했으므로, fit 시간의 미세한 순위를 probe당 계산비로 해석하기는 어렵다. 초기 simulator library 구성 {s['detector_static']:.1f} ms는 1회 비용이다. Fit·bank를 배분한 p90은 실측 warm sample에 cohort setup/H를 더한 계산값이다. 실제 장비의 접촉·이동·reset 시간은 측정하지 않았다.",
'''한 번의 접촉에 Brier 단위 λ를 매긴다면 예상 비용은 native Brier + λ × (B/H + test probe rate)다. 다음은 H=128에서 관측된 평균으로 계산한 손익분기값이다. 양의 추가 접촉과 양의 Brier 개선이 함께 있을 때 λ가 표의 손익분기값보다 작아야 유리하다. 음수이거나 접촉을 줄이는 경우는 해당 열의 부호와 원래 식으로 판단해야 한다. 가격에 대해 실제 사용자 선호나 유의성을 검정한 값은 없다.''',
tab(['비교 기준','Budget','Native Brier 개선','추가 calibration 접촉 / world','총 추가 접촉 / world','Calibration만 가격화한 λ*','모든 probe를 가격화한 λ*'],[[ref,b,sg(row['risk_gain']),f"{row['overhead']:.4f}",f"{row['all_increment']:+.4f}",f"{row['break_even_cal']:.4f}",f"{row['break_even_all']:.4f}" if row['break_even_all'] is not None else '—'] for ref in ('probe0','neutral') for b,row in zip(bs[1:],s['costs'][ref])]),
'![Direction identification, loss, and physical interactions](results/v28.png)',
'## 7. 논문 기여 가능성과 다음 검증',
(root/'ASSESSMENT_KO.md').read_text(),
(root/'LITERATURE_KO.md').read_text(),
'## 재현 자료',
'''서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/calibration_probe_v28`이다. 할당된 GPU 세 장의 process·메모리·사용률을 확인했고 두 장에는 다른 작업이 있어 node55의 RTX PRO 6000 Blackwell Server Edition 한 장만 사용했다. Primary, independent power, mechanism controls를 같은 GPU에서 순차 실행했다. 각 guard는 외부 프로세스 출현 여부와 worker return code를 기록했다. CPU 통계는 할당된 compute node에서 실행했다. [Primary 종료](gpu_worker/results/worker_exit.json) · [Power 종료](power_worker/results/worker_exit.json) · [Controls 종료](controls_worker/results/worker_exit.json) · [Power 코드](power28.py) · [대조군 통계](controls_stats28.py) · [대표 cohort의 동결·원자료 hashes](results/strong_neg_r0/metrics.json)''']
fig,axs=plt.subplots(1,3,figsize=(13.5,4.35),layout='constrained')
colors=['#33799d','#ca7042','#48806d','#77558c']
for a,col in zip((.35,.65,.85,.92),colors):
    axs[0].plot(range(4),[pw(a,b,'factual_probe') for b in bs],'-o',color=col,label=f'{a:g}')
axs[0].set(xticks=range(4),xticklabels=bs,title='Direction on independent cohorts',xlabel='Paid calibration probes',ylabel='Sign accuracy',ylim=(.45,1.04));axs[0].legend(title='Friction shift magnitude',fontsize=8,title_fontsize=8)
axs[1].plot(range(4),p['native'][:4],'-o',label='Native');axs[1].plot(range(4),p['common'][:4],'s--',label='Common decoder');axs[1].axhline(p['native'][4],ls=':',color='#777',label='Neutral native');axs[1].axhline(p['native'][5],ls='-.',color='#975',label='Known native')
axs[1].set(xticks=range(4),xticklabels=bs,title='Held-out decision loss',xlabel='Paid calibration probes',ylabel='Brier loss');axs[1].legend(fontsize=8)
axs[2].bar(range(6),p['probe'],label='Held-out probe');axs[2].bar(range(6),np.array(p['total'])-p['probe'],bottom=p['probe'],label='Calibration / 128',color='#d99553')
axs[2].set(xticks=range(6),xticklabels=['0','1','4','16','Neutral','Known'],xlabel='Policy',ylabel='Probes per deployment world',title='Cost across 128 worlds');axs[2].legend(fontsize=8)
fig.savefig(root/'results/v28.png',dpi=160);(root/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')
