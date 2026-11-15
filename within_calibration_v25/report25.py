"""Numerical tables in the Korean report are read from the completed workers' records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
r=root/'results'
def read(name):return json.loads((r/(name+'.json')).read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def n(x):return f'{x:.6f}'
def sg(x):return f'{x:+.6f}' if x is not None else '해당 없음'
def pc(x):return f'{100*x:.1f}%'
def ci(v):return f"{v['mean']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]"
def verdict(flag):return '충족' if flag else '미충족'
s=read('summary');design=s['design'];cfg=read('selected');dev=read('development')
methods=s['methods'];cost=s['mean_loss'];time=s['timing'];compar=s['comparisons']
labels=dict(gate='V25 factual gate',k256='고정 K256',query='Always-query',probe='Always-probe',direct='직접 선택만',
            k1024='고정 K1024',tail_gate='Support-only gate',k128='고정 K128',global_='Fixed global mixture',
            k256_three='고정 K256 · 세 action',v24_frozen='V24 frozen gate')
labels['global']=labels.pop('global_')
cells=['좁음 / 좁음','좁음 / 넓음','넓음 / 좁음','넓음 / 넓음']
stable=design['stable_unique_worlds_by_margin']['0.02']
parts=['# v25 · 하나의 sensor calibration에서 query view와 probe의 선택 가치',
(root/'FINDINGS_KO.md').read_text(), '## 1. 무엇을 독립적으로 조작했는가',
"""센서는 하나로 고정하고 **불확실한 물리량**을 바꿨다. Quadratic boundary의 curvature spread와 Coulomb friction coefficient의 spread를 독립적인 두 요인으로 두어 네 셀을 만들었다. Curvature의 크기 중심은 0.022 또는 0.340이고, 부호는 무작위다. Friction coefficient는 0.251에 ±0.011 또는 ±0.138을 더한다. 두 부호와 두 연속 jitter는 서로 독립이고, 같은 셀에서도 물체의 정확한 물리 상태는 다르다. Factual sensor noise σ는 boundary profile 0.0016, 충돌 후 velocity 0.095다. Query view, factual view, probe의 σ는 각각 0.00030, 0.00120, 0.027이며 모든 split과 world에 공통이다.

**왜 다른 관측이 필요한가?** Factual 충돌은 y=0에서 incoming velocity (1,0)으로 일어나 항상 stick한다. 이때 관측된 velocity에는 friction coefficient의 실제 크기가 드러나지 않는다. Probe는 같은 곳에서 incoming velocity (1,−1.10)로 충돌해 항상 slide하고, 충돌 후 속도는 friction에 반응한다. Query view는 y=0.025 주변의 boundary profile을 측정한다. 어느 쪽을 고르든 예측할 대상은 y=0.025에 incoming velocity (1,0.425)를 가했을 때의 stick 여부로 같다.

Query 위치의 slope를 s_q, restitution을 e라고 하면 stick을 가르는 friction threshold는 **µcrit = |s_q + 0.425| / [(1+e)(1−0.425s_q)]**다. 넓은 curvature·좁은 friction 셀에서 음의 curvature면 µcrit≤0.078, 양이면 µcrit≥0.483이다. 이 셀의 실제 µ는 0.2325–0.2695 안에 있으므로 curvature의 부호가 label을 결정한다. 좁은 curvature·넓은 friction 셀에서는 µcrit가 0.183–0.299 안에 있고, low µ≤0.1205, high µ≥0.3815여서 friction의 high/low가 label을 결정한다. 이 식은 **무엇이 label을 결정하는지** 보여준다. Noise가 있는 observation 중 어느 쪽이 더 낮은 Brier를 주는지는 별도의 paired test로 판정했다.

**정책에 보이는 단서.** Boundary offset은 넓은/좁은 curvature에 따라 평균이 +0.00115/−0.00115이고 독립적인 ±0.0017 jitter가 있다. Restitution은 넓은/좁은 friction에 따라 평균이 0.795/0.605이고 ±0.09 jitter와 velocity measurement noise가 더해진다. 따라서 factual profile과 velocity에는 spread를 짐작할 불완전한 단서가 있다. 어느 부호가 실현됐는지, 실제 셀 번호나 정답은 제공하지 않는다. 이 두 proxy와 spread의 관계는 generator에 의도적으로 넣은 가정이다.""",
'## 2. Factual gate와 비교 정책',
"""모든 방법은 **선택한 observation을 동일한 16,384-hypothesis inner-bank decoder로 업데이트**해 stick probability를 낸다. Brier는 (p−label)²이고 작을수록 좋다. 달라지는 것은 observation 선택이다. Primary action set은 query view와 probe 두 가지다. 두 numerical selectors는 기존 세 action용 acquisition kernel을 실행하고 Q/P 두 risk만 비교한다. 세 방법 모두 같은 물리 연산을 할 필요가 없으므로 고정 action이나 직접 선택이 가능하면 outer posterior와 acquisition을 건너뛴다. 별도의 unrestricted K256 대조에서 factual view도 허용했다.

V24의 factual gate는 support 크기만 정했다. V25 primary는 새로운 **action-value gate**다. 훈련 때 두 observation 모두를 독립적으로 시뮬레이션해 stick Brier를 얻고, 두 training bank pairs에서 평균한 advantage L(probe)−L(query)를 regression target으로 쓴다. 배포 때는 factual profile의 mean, slope, quadratic curvature, residual RMS, 충돌 직후 velocity의 두 성분만 읽는다. Advantage가 양수이면 query, 음수이면 probe로 직접 선택할 수 있다. 예측 절댓값이 작으면 K1024/M2048 posterior acquisition으로 돌아간다. 직접 선택은 새 observation의 결과나 test labels를 사용하지 않는다.

이 변경을 따로 평가하기 위해 ① 새 factual confidence로 support만 K128 또는 K1024로 고르는 control, ② v24-B 계수와 cutoff를 그대로 가져와 현재 calibration의 public metadata만 넣은 frozen gate, ③ 직접 선택만 사용하는 policy, ④ 항상 K1024 및 항상 K128인 policy를 실행했다. Static K256/M2048은 v24와 같은 label-stratified tail preservation이다. 따라서 고정된 것은 K이며, observation은 world에 따라 바꿀 수 있다.""",
f"Training 3,072, validation 1,024, held-out 1,280개 worlds를 분리했다. 각 split은 네 셀의 수가 같고 seeds는 각각 2511001, 2512001, 2513001이다. Pair별 16,384개 inner와 별도의 16,384개 external hypotheses를 사용했다. 2개 model families × 3 ridge penalties × 4 confidence thresholds의 24개 후보 중 validation Brier가 가장 작은 {cfg['family']} 모델을 선택했다. Penalty {cfg['penalty']}, cutoff {cfg['cut']:.5f}, validation 직접 선택 비율 {pc(cfg['confident'])}였다. RBF 모델은 quadratic features와 training set에서 고정 추출한 96개 Gaussian centers를 결합한다. Training target 생성, validation, 모델 선택까지 서버 wall time은 {dev['seconds']:.1f}초였다. 이는 offline 비용이다.",
f"설정을 고정한 validation에서 gate/K256/always-query의 Brier는 {n(cfg['validation_loss'])} / {n(dev['static_brier'])} / {n(dev['query_brier'])}였다. Test는 새로운 bank pairs 5쌍에 같은 독립 worlds와 noise를 적용했다. 모든 policy choices의 SHA-256을 먼저 고정하고, 그 뒤에 evaluator가 held-out labels와 potential observations를 열었다.",
'## 3. 같은 calibration에서 실제로 최적 observation이 바뀌었는가',
"""동일 world에서 Q/P 각각에 32회 observation noise를 적용했다. 16회씩 나눠 절반에서 더 낮은 Brier를 주는 action을 정하고, 다른 절반에서 그 선택의 loss를 측정한 다음 방향을 바꿔 평균했다. 두 절반 모두에서 같은 action이 0.02 넘게 유리하면 bank 평균 기준으로 그 world를 stable winner로 셌다. 사전 기준은 Q와 P의 stable worlds가 각각 10% 이상이고, 방향을 미리 정한 두 셀의 평균 Brier advantage에 대한 one-sided 95% bound가 각각 0.03을 넘는 것이다. 이는 고정된 hidden world에 여러 관측을 반복할 수 있는 evaluator의 판단이다. Factual evidence만 있는 policy가 그 winner를 미리 안다는 의미는 아니다.""",
table(['Curvature / friction spread','Q Brier','P Brier','P−Q Brier','Stable Q / P worlds (각 320)','K256 Q / P 선택률','V25 gate Q / P 선택률'],
      [[cells[i],n(row['actionwise_brier_q_f_p'][0]),n(row['actionwise_brier_q_f_p'][2]),sg(row['actionwise_brier_q_f_p'][2]-row['actionwise_brier_q_f_p'][0]),
        f"{stable['by_cell'][i]['query']} / {stable['by_cell'][i]['probe']}",
        ' / '.join(pc(row['action_counts_q_f_p']['k256'][k]/row['bank_worlds']) for k in (0,2)),
        ' / '.join(pc(row['action_counts_q_f_p']['gate'][k]/row['bank_worlds']) for k in (0,2))] for i,row in enumerate(s['cell_rows'].values())]),
f"사전 물리 검증은 **{verdict(design['design_pass'])}**이다. 넓은 curvature·좁은 friction의 P−Q lower bound는 {sg(design['physical_cell_contrasts']['2']['lower95'])}, 좁은 curvature·넓은 friction의 upper bound는 {sg(design['physical_cell_contrasts']['1']['upper95'])}였다. Stable Q {stable['query_worlds']}개, P {stable['probe_worlds']}개, 나머지 {stable['ambiguous_worlds']}개였다. 개별 bank–world에서 noise halves의 winner가 일치한 비율은 {pc(design['bank_world_split_agreement'])}다.",
table(['안정성에 요구한 Brier gap','두 half 모두 Q 우세','두 half 모두 P 우세','미분류 worlds'],[[k,v['query_worlds'],v['probe_worlds'],v['ambiguous_worlds']] for k,v in design['stable_unique_worlds_by_margin'].items()]),
f"Cross-fitted Q/P winner의 Brier는 {n(design['crossfit_oracle_qp_brier']['mean'])}, always-query 대비 실현 이득은 {ci(design['always_query_minus_qp_oracle'])}이다. 나머지 noise half에서 다시 평가한 winner별 이득은 Q 우세 group에서 {ci(design['heldout_gain_for_q_winners'])}, P 우세 group에서 {ci(design['heldout_gain_for_p_winners'])}였다. 셀 효과는 10,000회 within-cell world 및 bank bootstrap으로 계산했다.",
'## 4. Held-out Brier와 GPU p90',
table(['방법','Brier ↓','Q / factual view / P 횟수','GPU p50 / p90 ms','Gate−방법 Brier [paired 95% CI]'],
      [[labels[m],n(cost[m]),' / '.join(map(str,s['action_counts_q_f_p'][m])),f"{time[m]['p50_ms']:.3f} / {time[m]['p90_ms']:.3f}" if m in time else '미측정',
        '기준' if m=='gate' else ci(compar[m])] for m in methods]),
table(['Primary 비교','Gate−baseline Brier','One-sided 95% upper','다른 action 비율 / 고유 worlds','다른 선택에서 Δ Brier','GPU p90 ratio [paired 95% CI]'],
      [[labels[m],sg(compar[m]['mean']),sg(compar[m]['upper95']),f"{pc(compar[m]['different_choice_rate'])} / {compar[m]['different_choice_worlds']}",sg(compar[m]['difference_when_choice_changes']),
      f"{s['timing_ratios'][m]['ratio']:.3f} [{s['timing_ratios'][m]['ci95'][0]:.3f}, {s['timing_ratios'][m]['ci95'][1]:.3f}]"] for m in ('k256','query')]),
f"Gate가 두 baseline보다 낮은 Brier를 준다는 사전 공동 기준은 **{verdict(s['primary']['superior_to_both'])}**이다. 두 upper bounds가 모두 0 미만이어야 한다. GPU p90은 별도로 보고하며, 항상 같은 observation을 쓰는 경우의 저렴한 실행 비용도 포함했다. 같은 action을 고르면 decoder와 결과가 같으므로 전체 loss 차이는 ‘선택이 다른 비율 × 다른 선택에서의 평균 loss 차이’다.",
"""각 policy에 대해 CUDA synchronization을 포함한 float64 batch-1 wall latency를 측정했다. 각 셀 20 worlds × 4 interleaved repeats × 5 banks로 방법당 1,600회다. Raw factual host input, 필요한 bank fit과 acquisition, 선택된 observation의 host→GPU transfer, posterior decoding을 포함한다. Sensor의 실제 노출·이동·접촉 시간과 정적 bank 구축은 제외했다. GPU는 allocation 2744468, node55의 RTX PRO 6000 Blackwell Server Edition 한 장이다. P90 intervals는 bank·cell별 world·repeat block의 pairing을 유지한 5,000회 bootstrap에서 구했다. Loss와 elapsed time에 사용한 물리적 world 수는 각각 1,280개와 80개다. Noise와 bank는 독립적인 물체로 세지 않았다.""",
table(['Bank pair','Gate−K256','Gate−always-query'],[[i+1,sg(a),sg(b)] for i,(a,b) in enumerate(zip(compar['k256']['per_bank_delta'],compar['query']['per_bank_delta']))]),
'## 5. 어떤 구성에 이득을 귀속할 수 있는가',
table(['Support-only policy','Policy−K256 [95% CI]','Policy−always-query [95% CI]'],[[labels[m],ci(v['k256']),ci(v['query'])] for m,v in s['support_only_comparisons'].items()]),
f"V25의 direct branch는 held-out의 {pc(s['confident_rate'])}에서 실행됐다. V24 frozen gate의 high-budget 비율은 {pc(s['transported_high_rate'])}였다. 같은 direct 실행률이지만 world와 독립인 coin으로 직접 선택과 K1024를 섞은 blind routing의 기대 Brier는 {n(s['blind_routing']['loss'])}이고 실제 gate−blind 차이는 {ci(s['blind_routing']['gate_minus_blind'])}다. Blind routing은 계산한 기대값이며 p90을 별도로 측정하지 않았다.",
table(['분기','Worlds','Gate Brier','K256 Brier','항상 직접 선택 Brier','항상 K1024 Brier','Gate−K256'],[[k,v['worlds'],n(v['mean_loss']['gate']),n(v['mean_loss']['k256']),n(v['mean_loss']['direct']),n(v['mean_loss']['k1024']),sg(v['gate_minus_k256'])] for k,v in s['branches'].items()]),
"두 branch의 worlds는 gate의 confidence로 선별돼 있다. 집단 간 평균 Brier를 비교해 gate의 인과효과로 해석할 수 없다. 같은 world에 서로 다른 정책을 강제로 적용한 paired comparison이 branch 선택의 근거다.",
(root/'ABLATION_KO.md').read_text(),
'![Within-calibration observation value and paired held-out Brier](results/v25.png)',
'## 6. 논문 기여 가능성과 적용 범위',(root/'ASSESSMENT_KO.md').read_text(),
(root/'LITERATURE_KO.md').read_text(),
'## 재현 자료',
"""서버의 프로젝트 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/within_calibration_v25`이다. [사전 protocol](PROTOCOL.md), [simulator와 sensor calibration](physics25.py), [factual gate](gate25.py), [실행 코드](run25.py), [primary 통계](stats25.py), [proxy ablation](diagnostics.py), [검증 결과](results/summary.json), [실측 선택 고정 기록](results/choices_freeze.json)에 구현과 수치를 보존했다. 이전 v18–v24 sibling dependencies를 유지해야 한다. 새 Slurm output directory에서 v18 guard로 run25.py를 실행한 뒤, 같은 directory의 compute-node CPU에서 stats25.py, diagnostics.py를 실행한다. results/eval_*.npz는 양쪽 intervention과 세 번째 view의 realized losses, interleaved timing blocks를 담고 있다."""]

fig,axes=plt.subplots(1,2,figsize=(10.4,4.25),layout='constrained')
inds=np.arange(4);v=np.array([z['actionwise_brier_q_f_p'] for z in s['cell_rows'].values()])
axes[0].bar(inds-.18,v[:,0],width=.36,label='Query view',color='#246975')
axes[0].bar(inds+.18,v[:,2],width=.36,label='Probe',color='#b97744')
axes[0].set(xticks=inds,xticklabels=['N / N','N / W','W / N','W / W'],xlabel='Curvature spread / friction spread (N narrow, W wide)',ylabel='Held-out Brier',title='One sensor calibration')
axes[0].legend()
for i,m in enumerate(('k256','query','v24_frozen','direct')):
    d=compar[m];lo,hi=d['ci95'];axes[1].plot([lo,hi],[i,i],color='#246975',lw=2);axes[1].plot(d['mean'],i,'o',color='#246975')
axes[1].axvline(0,ls=':',color='.4');axes[1].set(yticks=range(4),yticklabels=['K256','Always-query','V24 frozen','Direct only'],title='V25 gate minus comparator',xlabel='Paired Brier difference (95% CI)',ylim=(-.6,3.6))
fig.savefig(r/'v25.png',dpi=155)
(root/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')
