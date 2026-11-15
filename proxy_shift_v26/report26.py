"""Korean Markdown report and scientific diagnostic figure built from the archived measurements."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
root=Path(__file__).resolve().parent
r=root/'results'
def load(f):return json.loads((r/(f+'.json')).read_text())
def tbl(h,rr):
    return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,x))+' |' for x in rr])
def f(x):return f'{x:.5f}'
def sg(x):return f'{x:+.5f}'
def pct(x):return f'{100*x:.1f}%'
def valpha(x):return f'{x:g}'
def condition(p):return '('+', '.join(valpha(x) for x in p)+')'
def iv(b,key='sim95'):
    return f"{sg(b['mean'])} [{sg(b[key][0])}, {sg(b[key][1])}]"
def ratio(v):return f"{v['p90_ratio']:.3f} [{v['p90_ratio_ci95'][0]:.3f}, {v['p90_ratio_ci95'][1]:.3f}]"
def at(i):return '검출 없음' if i is None else condition(conds[i])
def first_for(x,kind,margin='0.003'):
    return s['paths'][x]['thresholds'][margin][kind]['first']
s=load('summary');m=load('matched/summary');extra_time=load('matched/onset_latency');design=load('design');rows=s['rows'];mr=m['rows']
conds=design['conditions'];levels=design['levels']
index={tuple(c):i for i,c in enumerate(conds)}
axes=['profile','velocity','diagonal'];axisname={'profile':'Profile 관계만','velocity':'Velocity 관계만','diagonal':'두 관계 동시'}
def both(x,y):return f'{x:.3f} / {y:.3f}'
landmarks=[index[(a,b)] for a,b in [(1,1),(0,1),(-1,1),(1,0),(1,-1),(0,0),(-1,-1)]]
def status(bo):return '검출' if bo else '—'

parts=['# v26 · 두 factual proxy의 약화와 반전에서 동결된 v25 gate의 실패 경계',
(root/'FINDINGS_KO.md').read_text(),
'## 1. 무엇을 고정하고 무엇을 바꿨나',
"""V25 factual gate의 96개 RBF centers, normalization, regression 계수, confidence cutoff(0.0478034), decision rule을 원본 hash와 대조해 그대로 불러왔다. 추가 학습이나 threshold tuning은 없었다. Gate는 raw factual evidence로 Brier(probe)−Brier(query view)를 예측한다. 절댓값이 cutoff 이상이면 action을 직접 선택하고, 작으면 K1024/M2048 posterior acquisition을 사용한다. 고정 K256은 같은 likelihood와 M2048, label-stratified support를 사용하며 각 world에서 Q/P를 다시 비교한다. 어떤 방법이든 선택이 같으면 같은 16,384-hypothesis inner decoder로 최종 stick probability를 계산하므로 Brier도 같다.

각 world의 curvature spread와 friction spread는 독립적인 두 binary factors다. V25에서 넓은 curvature의 물체는 평균 boundary offset이 +0.00115, 좁은 물체는 −0.00115였다. 넓은 friction의 물체는 restitution이 평균 0.795, 좁은 물체는 0.605였다. Factual 충돌이 stick하므로 restitution이 큰 쪽의 outgoing horizontal velocity는 더 음수다. Gate에 보이는 profile mean과 velocity는 각각의 spread에 대한 불완전한 단서이며, cell이나 latent parameter를 입력으로 쓰지 않는다.

두 관계의 세기를 αc, αf로 표시했다. 각 관계에서 world별로 확률 (1−α)/2만큼 **반대 폭 집단의 조건부 중심값**을 사용하고 원래의 개별 jitter는 유지한다. α가 1이면 v25 관계를 유지한다. 0.75는 불일치 확률 12.5%, 0.5는 25%, 0은 50%, −0.5는 75%, −1은 100%다. α=0에서 조건부 중심과 해당 폭은 모집단에서 독립이다. 관측 feature에는 geometry·noise도 들어가므로 측정한 Pearson correlation이 α나 정확한 0일 필요는 없다. 두 flip에는 서로 독립인 난수를 쓰고 severity 간에는 동일 난수를 재사용했다. 이 방식은 모집단의 두 중심 혼합과 그 jitter 분포를 유지하고 coupling만 바꾼다.

Offset 변화는 profile 전체를 평행 이동하므로 slope와 참 stick label을 보존한다. Restitution 변화는 factual velocity, probe response 및 일부 query label을 물리적으로 바꾼다. 두 경우 모두 새 물리를 반영해 실제 sensor observations와 label을 다시 계산했다. 실제 장비를 구동하지 않은 synthetic intervention이다.""",
"""Query 위치·속도, Coulomb law, 후보 actions Q/P, factual/추가 sensor calibration 모두 v25와 같다. 1,024개 새 worlds(각 spread cell 256개), 새 독립 bank pairs 5쌍, world·action당 32개 noise repeats를 사용했다. 변화량을 격리하기 위해 36개 조건에서 world, latent jitter와 표준화된 noise를 공유했다. 한 world에서 실제 가능한 변경은 네 가지(원본/velocity 관계만 뒤집힌 물리/offset 관계만 뒤집힌 물리/둘 다)여서 source-prior policy의 같은 입력은 정확히 한 번 계산해 재사용했다. GPU latency는 각 조건에서 별도로 실행한 실측값이다.""",
'## 2. ‘먼저 실패’의 사전 정의',
"""Primary에서는 gate와 K256, 그 공통 decoder 모두 v25의 원래 물리 prior를 유지했다. Δ(s)=L_gate(s)−L_K256(s)라 하고, intact 조건 (1,1)을 0으로 쓴다. 실용적으로 의미 있는 **상대 실패**는 Δ(s)의 simultaneous 95% lower bound가 +0.003을 넘고, Δ(s)−Δ(0)의 lower bound도 0을 넘는 최초 tested condition이다. +0.003은 v24부터 사용한 연구용 absolute Brier margin이다. 실제 robot 사용자의 허용 비용으로 측정한 값은 없다.

**Gate 단독 선행 악화**는 위 두 조건에 더해 L_gate(s)−L_gate(0)의 lower bound가 +0.003을 넘고, L_K256(s)−L_K256(0)의 upper bound가 +0.003 미만인 경우다. 이것은 두 방법이 동시에 크게 나빠지지만 한 방법이 조금 더 나쁘다는 결과와 구분된다. 물리 시뮬레이터의 시간이 흐르면서 고장이 발생하는 지점이 아니라, 사전 선택한 severity grid에서 통계적 조건을 처음 만족하는 지점이다.

4개 physical cells 안에서 world를, 별도로 bank를 resample한 10,000회 crossed bootstrap을 썼다. 모든 방법·조건에 같은 draws를 적용했다. 36개 조건 × 4개 Brier contrasts에 대해 bootstrap 표준편차로 표준화한 최대 절댓값의 95 percentile을 사용했다. 아래 Brier bands는 이 144개 값에 대한 simultaneous bands다. 동일 관측의 noise repeats나 bank를 별도 독립 world로 세지 않았다. Timing은 12 worlds/cell × 3 repeats × 5 banks로 각 조건·방법당 720개다. 순서를 무작위화하고 batch-1 host→GPU factual 전송부터 선택한 observation의 posterior update까지 CUDA synchronization을 포함했다. Sensor를 이동시키거나 probe를 물체에 가하는 시간과 static bank setup은 제외했다. P90 intervals는 world·bank·repeat를 짝지어 재추출한 3,000회 pointwise bootstrap이다. Timing에는 Brier의 +0.003 기준을 적용하지 않았다.""",
'## 3. 실패가 처음 관측된 조건',
tbl(['원본에서 severity를 키운 경로','Δ 하한 >0 및 상대 악화','상대 실패 (+0.003)','그 뒤 모두 상대 실패','Gate 단독 선행 악화','Gate 자체 악화 하한 >+.003','K256 자체 악화 하한 >+.003'],
[[axisname[a],at(first_for(a,'relative','0.0')),at(first_for(a,'relative')),
 at(s['paths'][a]['thresholds']['0.003']['relative']['first_maintained']),at(first_for(a,'absolute'))]+
 [at(next((i for i in s['paths'][a]['indices'] if rows[i]['bands'][k]['sim95'][0]>.003),None)) for k in ('gate_change','k256_change')] for a in axes]),
tbl(['상대 실패 민감도','Margin 0','Margin +0.001','Margin +0.003','Margin +0.005'],[[axisname[a]]+[at(first_for(a,'relative',v)) for v in ('0.0','0.001','0.003','0.005')] for a in axes]),
'36개 조합에서 각 좌표를 따로 바꾸므로 모든 조건을 단일 순서로 정렬할 수는 없다. 위 표는 두 단독 경로 및 같은 세기로 함께 바꾼 대각선 경로를 따른다. ‘검출 없음’은 이 표본과 기준으로 입증하지 못했다는 뜻이다.',
tbl(['(αc, αf)','Gate Brier','K256 Brier','Δ [simultaneous 95%]','원본 대비 Δ 변화 [simultaneous 95%]','직접 선택률','Gate / K256 p90 ms','p90 ratio [pointwise 95%]'],
[[condition(conds[i]),f(rows[i]['loss']['gate']),f(rows[i]['loss']['k256']),iv(rows[i]['bands']['gap']),iv(rows[i]['bands']['excess_gap']),pct(rows[i]['facts']['high_confidence_rate']),
 both(rows[i]['timing']['gate']['p90_ms'],rows[i]['timing']['k256']['p90_ms']),ratio(rows[i])] for i in landmarks]),
'## 4. Proxy가 실제로 바뀌었는지와 선택 오류의 경로',
tbl(['(αc, αf)','r(profile mean, curvature spread)','r(vx, friction spread)','원본 대비 label 변경','예측 margin과 실현 margin의 r','Gate / K256 probe 비율','Gate와 K256이 달랐던 고유 worlds','갈린 선택에서 평균 Δ'],
[[condition(conds[i]),f"{rows[i]['facts']['corr_profile_curvature_spread']:.3f}",f"{rows[i]['facts']['corr_vx_friction_spread']:.3f}",pct(rows[i]['facts']['label_change']),
 f"{rows[i]['facts']['margin_score_corr']:.3f}",pct(rows[i]['facts']['gated_probe_rate'])+' / '+pct(rows[i]['facts']['k256_probe_rate']),rows[i]['changed_choice_worlds'],
 sg(rows[i]['gap_on_difference']) if rows[i]['gap_on_difference'] is not None else '—'] for i in landmarks]),
"""실현 margin은 같은 실제 world에서 probe의 Brier에서 query view의 Brier를 뺀 값이다. 이 값은 추가 관측 이전에는 알 수 없다. 두 policy의 Δ는 선택이 갈린 비율과 그 경우의 평균 Brier 차이의 곱이다. 따라서 confidence가 유지된다는 사실만으로 gate의 판단이 유효하다고 평가할 수 없다.""",
tbl(['(αc, αf)','직접 branch의 잘못된 부호¹','Fallback branch의 gate−K256','Pure cue replay Brier','Replay의 원본 gate 대비 변화 [pointwise 95%]','Replay에서 바뀐 action'],
[[condition(conds[i]),pct(rows[i]['direct_branch']['direct_wrong_fraction']),sg(rows[i]['fallback_branch']['gate_minus_k256']),f(rows[i]['pure_cue']['gate_loss']),
sg(rows[i]['pure_cue']['change'])+' ['+', '.join(sg(z) for z in rows[i]['pure_cue']['change_ci95'])+']',pct(rows[i]['pure_cue']['choice_flip_rate_vs_origin'])] for i in landmarks]),
"""¹ Bank·world별 반복 noise 평균으로 정의한 realized winner와 direct branch의 예측을 대조했다. **Pure cue replay**는 gate의 score에만 shifted input을 주고, fallback·decoder·실제 observation과 query label은 모두 원본 world의 것을 사용한다. 다른 성능 저하 경로를 차단한 인공 진단이므로, coherent physical shift에서 이 수치만큼 gate가 더 나빠졌다고 해석할 수 없다.""",
(root/'DECOMPOSITION_KO.md').read_text(),
'## 5. 같은 shifted prior를 두 방법에 제공하면',
"""두 번째 실험에서는 frozen gate의 계수·cutoff를 유지하면서, gate fallback·K256·공통 decoder의 numerical banks가 해당 (αc, αf)를 안다고 가정했다. 개별 world의 실제 flip은 제공하지 않았다. 조건별 prior를 만드는 16,384개의 후보와 독립 flip draws로 likelihood·stick labels를 다시 계산했다. 첫 세 bank, 각 cell에서 사전 정한 64개씩 총 256개 worlds를 사용했다. 비교하는 source-prior 값도 정확히 같은 부분표본이다. 더 작은 secondary 표본의 경계를 full-cohort primary와 혼합하지 않았다. Matched condition별 모든 선택을 고정한 뒤 그 실험의 evaluator가 private outcomes를 열었다.""",
tbl(['경로','Matched 상대 실패 첫 조건','Matched gate 단독 선행 악화','동일 부분표본 source-prior 상대 실패','부분표본 source-prior 단독 선행 악화'],
[[axisname[a]]+[at(m['crossing'][a][mode][k]['first']) for mode in ('flag','stale_flag') for k in ('relative','absolute')] for a in axes]),
tbl(['(αc, αf)','Matched gate / K256 Brier','Source gate / K256 Brier (같은 worlds)','Matched Δ [simultaneous 95%]','Matched Δ−source Δ [simultaneous 95%]','Matched gate / K256 p90 ms'],
[[condition(conds[i]),' / '.join(f(z) for z in mr[i]['matched_loss']),' / '.join(f(z) for z in mr[i]['stale_loss']),iv(mr[i]['matched_bands']['gap']),iv(mr[i]['matched_minus_stale_gap']),
 both(*mr[i]['timing']['p90']) if 'timing' in mr[i] else '미측정'] for i in landmarks]),
'사전 지정한 timing 다섯 조건은 (1,1), (−1,1), (1,−1), (−1,−1), (0,0)이다. 각 방법에 대해 조건별 3 banks × 16 worlds × 2 repeats로 측정했다. Online 시간에 matched prior의 offline 구성 비용은 포함하지 않았다.',
f"대각선에서 처음 relative failure가 검출된 (−0.5, −0.5)는 사전 timing 대상에 없었다. 이후 같은 GPU에서 고정한 정책과 선택을 그대로 사용하고 48 worlds × 3 repeats × 3 banks를 추가 측정했다. Gate / K256 p90은 {both(*extra_time['p90_ms'])} ms였다. Ratio {extra_time['ratio']:.3f}의 pointwise 95% 범위는 [{extra_time['pointwise_ci95'][0]:.3f}, {extra_time['pointwise_ci95'][1]:.3f}]였다. 이 보완 측정은 Brier 판정을 바꾸지 않는다. [추가 timing](results/matched/onset_latency.json)", 
'36개 조건 전체에서 matched relative failure가 검출된 곳은 10개였다. 다른 검출점보다 두 축 모두에서 더 심하게 바꿀 필요가 없었던 경계 조합은 (0.5, −1), (−0.5, 0), (−1, 1)이었다. 일부 인접 조건에서는 구간이 다시 넓어져 판정이 달라지므로 연속적인 안전 임계값은 아직 추정할 수 없다.',
tbl(['αc ↓ / αf →']+[valpha(z) for z in levels],[[valpha(ac)]+[sg(mr[index[(ac,af)]]['matched_bands']['gap']['mean'])+(' **∗**' if mr[index[(ac,af)]]['flag']['relative'] else '') for af in levels] for ac in levels]),
'각 값은 matched-prior gate−K256 Brier다. ∗는 +0.003 relative failure를 표시한다. 직접 비교에서는 동시 반전 75%에서 matched Δ가 source Δ보다 커진 양의 simultaneous interval이 0을 포함했다. Profile 완전 반전과 두 축 완전 반전에서는 그 차이의 lower bound가 각각 +0.02327과 +0.04836이었다.',
'## 6. 전체 36개 조건',
tbl(['αc','αf','Gate Brier','K256 Brier','Gate 자체 악화','K256 자체 악화','Δ [simultaneous 95%]','Gate / K256 p90 ms','+.003 상대 실패','Gate 단독 선행 악화'],
[[valpha(c[0]),valpha(c[1]),f(v['loss']['gate']),f(v['loss']['k256']),sg(v['bands']['gate_change']['mean']),sg(v['bands']['k256_change']['mean']),iv(v['bands']['gap']),
 both(v['timing']['gate']['p90_ms'],v['timing']['k256']['p90_ms']),status(v['flags']['0.003']['relative']),status(v['flags']['0.003']['absolute'])] for v,c in zip(rows,conds)]),
'![Gate robustness under two proxy shifts](results/v26.png)',
'두 heatmap은 0을 중심으로 같은 대칭 색 범위를 사용한다. 점은 matched relative failure 검출을 표시한다. 오른쪽 timing 그래프의 값은 point estimate이며 uncertainty는 앞 표에 제시했다.',
'## 7. 논문 기여 가능성과 적용 범위',(root/'ASSESSMENT_KO.md').read_text(),
(root/'LITERATURE_KO.md').read_text(),
'## 재현 자료',
"""서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/proxy_shift_v26`이다. [사전 protocol](PROTOCOL.md), [물리적 shift 구현](physics26.py), [GPU 실행](run26.py), [primary 통계](stats26.py), [matched-prior 실행](matched26.py), [matched-prior 통계](matched_stats26.py), [원본 및 선택 경로 검산](counterfactual26.py), [실행 장치](results/device.json), [GPU 종료 기록](results/worker_exit.json), [추가 p90 측정 코드](onset_timing26.py), [36개 조건 원수치](results/summary.json), [matched 결과](results/matched/summary.json), [동결된 v25 gate](../within_calibration_v25/results/selected.json)를 보존했다. GPU worker는 한 장의 RTX PRO 6000 Blackwell Server Edition만 순차적으로 사용했다. 이전 v18–v25 모듈과 상대 경로를 유지한 새 서버 output directory에서 빈 Slurm GPU를 확인하고 v18 guard로 run26.py를 실행한다. 별도 worker log directory에서 matched26.py를 실행한 후 compute-node CPU로 stats26.py, counterfactual26.py, matched_stats26.py를 실행한다. 원본 선택·parameter hashes와 실제 timing, per-world losses는 results에 보존했다."""]
fig,axesfig=plt.subplots(1,3,figsize=(13.3,4.25),layout='constrained')
a=np.array([[v['bands']['gap']['mean'] for v in rows],[v['matched_bands']['gap']['mean'] for v in mr]])
limit=float(np.max(np.abs(a)))
norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
for ax,arr,title in zip(axesfig[:2],a,['Frozen source prior (1,024 worlds)','Known shifted prior (256 worlds)']):
    im=ax.imshow(arr.reshape(6,6),cmap='RdBu_r',norm=norm,aspect='auto')
    ax.set(xticks=range(6),yticks=range(6),xticklabels=[valpha(i) for i in levels],yticklabels=[valpha(i) for i in levels],title=title,xlabel='Velocity link strength',ylabel='Profile link strength')
for i,row in enumerate(mr):
    if row['flag']['relative']:
        axesfig[1].plot(i%6,i//6,'o',markersize=4,markerfacecolor='white',markeredgecolor='black')
fig.colorbar(im,ax=axesfig[:2],shrink=.83,label='Gate minus K256 Brier')
ax=axesfig[2]
for key,style in zip(axes,['o-','s--','^:']):
    rr=[rows[i] for i in s['paths'][key]['indices']]
    ax.plot(range(len(levels)),[v['p90_ratio'] for v in rr],style,label=key)
ax.axhline(1,color='.5',ls=':');ax.set(xticks=range(6),xticklabels=[valpha(i) for i in levels],xlabel='Remaining link strength',ylabel='Gate / K256 GPU p90',title='Matched hardware, source prior');ax.legend(fontsize=8)
fig.savefig(r/'v26.png',dpi=165)
(root/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')
