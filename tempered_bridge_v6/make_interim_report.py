"""Report completed evidence without treating paused refinement as convergence."""
import json,datetime,collections
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
read=lambda p:json.loads((root/p).read_text())
s=read('results/analysis/summary.json');failure=read('results/analysis/failure_audit.json')
one=read('results/analysis/one_step_main.json');of=read('results/one_step/failure.json')
pooled=read('results/analysis/same_evidence_main.json')['main'];partial=read('results/analysis/partial_refinement.json')
resource=read('results/resource_state.json');runtime=read('results/runtime.json')
pause=datetime.datetime.fromtimestamp(resource['paused_at'],datetime.timezone(datetime.timedelta(hours=9))).isoformat()
L=[]
def add(x=''):L.append(x)
def pct(x):return f'{100*x:.1f}%'
def ci(c):return f"{c['mean']:+.7f} [{c['low']:+.7f}, {c['high']:+.7f}]"
def table(head,rows):
    add('| '+' | '.join(head)+' |');add('| '+' | '.join(['---']*len(head))+' |')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()

add('# v6 중간 실험 보고서 — Tempered SMC bridge와 nested full SMC\n')
add(f'**실행 상태: {pause}에 GPU inference를 중단했다.** 처음에 비어 있던 두 GPU에서 실행했으나, 이후 네 할당 GPU 모두 외부 `doom2-vagen` training process를 사용하게 됐다. 사용자의 “사용 중인 GPU에서는 inference를 멈춘다”는 조건에 따라 우리 step만 중단했다. 기존 allocation과 외부 training은 변경하지 않았다.\n')
add('**Tempered bridge 구현, 72-world matched comparison, ESS failure audit 및 one-step control은 완료했다. Adaptive refinement는 미완료다. 따라서 “acquisition ranking이 충분히 안정화된 뒤에도 observation selection의 이득이 유지된다”는 최종 검증은 아직 완료되지 않았다.** 완료 결과와 남은 검증을 아래에서 구분한다.\n')

add('## 1. 현재 답할 수 있는 질문\n')
add('1. **기존 ESS collapse의 큰 posterior error를 줄일 수 있는가?** 확인한 failure cases에서는 줄었다. 다만 가장 큰 회복은 factual posterior에서 proposal을 시작하는 방향 전환만으로도 나타났다. 전체 개선을 tempering만의 효과라고 주장하면 안 된다.\n2. **Tempered bridge가 nested full SMC와 비슷한 acquisition을 만드는가?** Main64의 active queries에서 top-action agreement는 high 96.8%, pixel_only 96.0%였다.\n3. **Inner inference가 맞으면 ranking도 안정적인가?** 그렇지 않았다. Outer sampling을 바꾸거나 동일한 evidence의 independent calculations를 비교하면 상당한 action variation이 남았다.\n4. **강한 fixed rule보다 selection이 좋은가?** High에서는 확인하지 못했다. Pixel_only에서는 개선 방향이 있었으나 interval이 0에 매우 가까웠고, 아직 outer refinement가 끝나지 않았다.\n')

add('## 2. 구현과 target distribution\n')
add(r'''Factual likelihood를 $L_f(\theta)$, hypothetical observation likelihood를 $L_a(\theta;y)$라고 쓰면 다음 두 sequence를 비교했다.

$$\pi^{bridge}_{\beta}(\theta)\propto p(\theta)L_f(\theta)L_a(\theta;y)^{\beta},\qquad\beta:0\rightarrow1.$$

$$\pi^{full}_{\beta}(\theta)\propto p(\theta)[L_f(\theta)L_a(\theta;y)]^{\beta},\qquad\beta:0\rightarrow1.$$

Bridge는 factual posterior bank에서 시작하며 **factual likelihood의 exponent를 항상 1로 유지**한다. Full SMC는 hypothetical observation마다 prior에서 다시 시작한다. 최종 target은 같고 경로가 다르다.

Bridge의 incremental weight는 $\exp[(\beta_{t+1}-\beta_t)\log L_a]$다. Adaptive bisection으로 pre-resampling ESS가 0.75N 이상이 되도록 temperature를 정하고, systematic resampling 뒤 covariance/local/global proposal을 번갈아 쓰는 MH rejuvenation을 수행했다. Bounds 밖 proposal은 reject한다. Population마다 temperature, covariance, resampling, factual bank가 독립적이다.

Resampling 후 uniform weights의 ESS=N을 성공 지표로 삼지 않았다. One-step ESS, pre-resampling stage ESS, acceptance, ancestry, predictive means, contact probabilities 및 independent-run 차이를 저장했다. Weight ESS는 MCMC correlation을 포함한 실제 iid-equivalent sample size가 아니다.

이것은 established SMC methodology의 구현이다. Tempering 자체를 새로운 algorithm으로 주장하지 않는다. 배경은 [Del Moral, Doucet & Jasra, Sequential Monte Carlo samplers (2006)](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)에 해당한다.
''')

add('## 3. 같은 조건에서 무엇을 비교했는가\n')
table(['항목','설정'],[
    ['Worlds','v5의 동일한 72 worlds: corner/gate/deflection 각각 24'],
    ['Split','Family마다 앞 8 validation, 뒤 16 test: 총 24/48'],
    ['Queries/actions','World당 6 counterfactual queries; view/left/right/center'],
    ['Latent scene','g,e,μ,d,b의 5D continuous joint posterior'],
    ['Factual bank','v5 continuous SMC 4096 particles; 두 method가 공유'],
    ['Main64','64 hypothetical observations ×1024 particles ×12 MH moves/stage ×2 independent inner runs'],
    ['View quality','high 및 pixel_only boundary representation; dynamic probes는 공유 계산'],
    ['Sensor','Factual은 vx만, probes는 vx/vy; 50 ms finite-frame velocity, dynamic σ=0.025 m/s'],
    ['Target','0.8 s의 instantaneous counterfactual velocity'],
    ['Outcome evaluation','선택한 실제 observation에 대한 v5 full SMC posterior score 재사용'],
    ['Uncertainty','48 test worlds 단위 paired bootstrap 10000회,95% interval']])
add('Uniform prior bounds는 `g∈[-8,8] mm`, `e∈[0.15,0.85]`, `μ∈[0,0.6]`, `d∈[22,34] mm`, `b∈[0,6] mm`다. Radius 20 mm sphere와 static rounded rectangles에 대한 2D continuous collision detection, Coulomb impulse, spin을 사용하며 gravity는 없다. Probe shifts는 −22/+22/+9 mm, query shifts는 −30/−17/−7/+4/+15/+28 mm다.\n')
add('v5의 image-derived geometry representation과 frozen calibration은 유지했다. High는 1.5 mm pre-render position/extent jitter를 포함하고 pixel_only는 이를 제거한 high-resolution condition이다. 둘 다 low-dimensional calibrated feature likelihood이며 unrestricted raw-pixel likelihood가 아니다.\n')
add('Methods와 actions는 같은 outer latent draws 및 Gaussian noise를 사용한다. Policy 입력에는 factual observation, family, query, frozen calibration만 사용한다. Hidden scene parameters, query truth, unselected actual observation은 evaluation에만 사용한다. Family rule과 더 강한 family×query rule은 validation 24 worlds에서 고정했다.\n')
add('이번에는 view뿐 아니라 세 probes의 hypothetical posterior도 nested SMC로 계산했다. 따라서 v5와의 차이는 view estimator 교정만의 효과가 아니다. Mixed-estimator ablation을 아래에 따로 보고한다.\n')

add('## 4. ESS collapse: 방향 전환과 tempering/rejuvenation을 분리해서 보기\n')
table(['World/hypothesis','기존 image-conditioned ESS','기존 RMSE','Factual-bank SNIS RMSE','Tempered bridge RMSE','New full SMC RMSE'],[
    [f"{r['case']}/{r['hypothesis']}",f"{r['old_ess']:.2f}",f"{r['old_rmse']:.6f}",f"{next(a for a in of if a['case']==r['case'] and a['hypothesis']==r['hypothesis'])['rmse']:.6f}",f"{r['bridge_rmse']:.6f}",f"{r['full_rmse']:.6f}"] for r in failure])
add('RMSE 단위는 m/s이며, reference는 v5에서 별도 실행한 full SMC 4096 particles ×18 moves 두 runs의 평균이다. Hidden truth와의 오차가 아니라 posterior inference 간 불일치를 측정한다. 기존 importance estimator는 2048 samples, 새 SNIS 및 failure-audit SMC는 4096 particles이므로 ESS 절대값을 같은 denominator로 비교하지 않는다. Failure cases는 낮은 ESS 때문에 미리 선택한 diagnostic subset이다.\n')
add('가장 큰 failure인 66/110은 기존 RMSE 0.267773에서 factual-bank one-step SNIS만으로 0.001538까지 줄었다. Tempered bridge는 0.000964였다. 따라서 이 큰 회복의 대부분을 “tempering의 새로운 효과”로 설명할 수 없다. 반면 68/110에서는 one-step 0.020969를 bridge 0.003792로 줄여 continuous tempering/rejuvenation의 추가 개선이 나타났다. 68/68처럼 residual disagreement가 남는 case도 있다.\n')
add(f"![Failure recovery]({root}/results/analysis/failure_recovery.png)\n")
table(['Condition','Method','One-step ESS median/min','Minimum stage ESS','Mean stages','Mean acceptance'],[
    [q,m,f"{d['one_step_ess_median']:.1f}/{d['one_step_ess_min']:.2f}",f"{d['stage_ess_min']:.1f}",f"{d['stages_mean']:.2f}",pct(d['acceptance_mean'])]
    for q in s for m,d in s[q]['cost'].items()])
add('이 표는 Main64의 N=1024 기준이다. Bridge의 one-step ESS는 factual-bank subsample을 한 번 reweight했을 때, full의 one-step ESS는 prior에 joint likelihood를 한 번 적용했을 때다. 높은 stage ESS만으로 mixing이나 missed mode의 복구가 증명되는 것은 아니다.\n')

add('## 5. Inner posterior의 일치와 acquisition ranking의 안정성은 다르다\n')
table(['Condition','Bridge/full predictive RMSE median/max','Bridge repeat RMSE median','Full repeat RMSE median','Any-contact MAE','Active top-action agreement'],[
    [q,f"{d['predictive_rmse_median']:.6f}/{d['predictive_rmse_max']:.6f}",f"{d['bridge_rep_rmse_median']:.6f}",f"{d['full_rep_rmse_median']:.6f}",f"{d['contact_mae']:.6f}",pct(d['agreement_active'])]
    for q in s for d in [s[q]['diagnostics_summary']['test']]])
add('각 condition에서 active queries는 249/288이다. Active는 full EVSI maximum >1e-4로 정의한다. Contact 지표는 any-contact이며 contact sequence 전체 distribution의 일치를 검증한 것은 아니다.\n')
add(r'''Acquisition은 $U(a,q)=E_y\|m_y(q)-m_0(q)\|^2$다. Reported MSE는 velocity 두 components의 평균이므로 utility는 expected MSE reduction의 두 배 scale이다. Independent inner runs 두 개의 cross-product $(\hat m_y^{(1)}-m_0)^T(\hat m_y^{(2)}-m_0)$로 finite-inner variance의 leading positive contribution을 줄였다. 실제 finite SMC bias와 shared factual-bank error는 남으며, 완전한 unbiased estimator라고 주장하지 않는다. 음수 estimates는 clipping하지 않았다.
''')
table(['Condition','Method','Half-sample top-action agreement','Practical confidence','Exact-winner decisive'],[
    [q,m,pct(d['half_agreement_active']),pct(d['practical_confidence_active']),pct(d['decisive_active'])]
    for q in s for m in ['bridge','full'] for d in [s[q]['diagnostics_summary']['test'][m]]])
add('Practical confidence는 outer bootstrap 1000회에서 winner regret의 upper 95%가 `max(1e-4, 0.05 × maximum EVSI)` 이하인지로 판단한다. Exact-winner decisive는 winner와 모든 alternatives의 gap이 bootstrap lower 2.5%에서도 양수인지다. Near-tie에서는 action name이 바뀌어도 utility loss가 작을 수 있다. 이 기준의 tolerance는 명시적인 numerical diagnostic이며 모든 task에서 무시 가능한 risk라는 뜻은 아니다.\n')
add('Inner bridge/full agreement가 약 96–97%여도 half-sample agreement는 약 59–62%였다. 따라서 같은 outer draws를 공유한 두 estimators의 agreement만 보고 acquisition이 수렴했다고 판단하면 안 된다.\n')

add('## 6. 기본 budget에서 observation selection의 이득\n')
methods=['bridge','full','value_rule','type_rule','always_view','random','bridge_EIG','bridge_probe_only','v5_bank','v5_conditional','no_observation']
for q,d in s.items():
    add(f'### {q}\n')
    table(['Policy','Velocity MSE [(m/s)²]','Marginal CRPS','90% coverage'],[[m,f"{d['summary'][m]['mse']:.7f}",f"{d['summary'][m]['crps']:.7f}",pct(d['summary'][m]['coverage90'])] for m in methods])
    table(['Paired difference','Mean [95% interval]'],[[f'Bridge − {b}',ci(d['paired_ci']['bridge minus '+b]['mse'])] for b in ['value_rule','type_rule','always_view','bridge_EIG','bridge_probe_only','v5_conditional','full']])
add('High의 strong family×query rule 대비 차이는 −0.0003890 [−0.0034947,+0.0033052]로 개선을 확인하지 못했다. Pixel_only는 −0.0022051 [−0.0045389,−0.0000012]로 음수였지만 upper bound가 0에 매우 가깝다. 아직 outer refinement가 끝나지 않았고 multiple-comparison correction을 거친 confirmatory result도 아니므로 강한 성능 개선 주장으로 사용하면 안 된다.\n')
add(f"![Main64 policy comparison]({root}/results/analysis/policy_comparison.png)\n")
add('Figure error bars는 policy별 interval이다. 비교에는 표의 paired difference interval을 사용한다. 288 queries를 independent samples로 취급하지 않고 48 worlds를 bootstrap 단위로 사용했다. 각 world/action에는 하나의 realized observation만 있어, expected utility가 거의 같은 action도 실제 noise realization에서 다른 error를 낼 수 있다.\n')

add('## 7. 더 저렴한 control과 numerical-invariance 검사\n')
table(['Condition','One-step/full predictive RMSE median/max','One-step/full active rank agreement','Matched64 SNIS MSE','512-outer SNIS MSE','Bridge − matched64 SNIS [95% interval]'],[
    [q,f"{d['prediction_rmse_median']:.6f}/{d['prediction_rmse_max']:.6f}",pct(d['agreement_active']),f"{d['matched_mse']:.7f}",f"{d['max512_mse']:.7f}",ci(d['bridge_minus_snis'])] for q,d in one.items()])
snistime=sum(read(str(p.relative_to(root)))['seconds'] for p in (root/'results/one_step').glob('case_*.json'))
add(f'One-step factual-bank SNIS는 4096 particles,512 outer observations의 72 worlds 전체를 합해 약 {snistime:.1f} s의 per-case timer 합을 사용했다. GPU context startup 및 failure-control 추가 계산을 포함한 전체 wall-clock time과는 다르다. Main SMC와 비교해 훨씬 저렴하지만 finite bank는 continuous posterior와 다르다. Hypothetical generator와 posterior bank를 공유하므로 self-inclusion에 따른 optimistic effect도 가능하다. 특히 pixel_only에서는 full-reference rank agreement가 72.7%로 떨어졌지만 realized MSE에서는 bridge 대비 유의한 열세가 확인되지 않았다.\n')
table(['Condition','v5 conditional MSE','v6 view + v5 probes','v5 view + v6 probes','All v6 bridge'],[[q]+[f"{s[q]['summary'][m]['mse']:.7f}" for m in ['v5_conditional','v6_view_v5_probes','v5_view_v6_probes','bridge']] for q in s])
add('Utility columns만 교체한 mixed-estimator diagnostic이다. v5/v6의 MC draws와 budgets가 다르므로 완전한 causal decomposition은 아니다. 전체 변화가 view correction 하나 때문이라고 단정할 수 없음을 보여준다.\n')
add('### 동일한 factual evidence의 일관성\n')
add('Unique factual x-velocity는 corner 19개, gate 10개, deflection 18개다. 같은 family와 같은 factual observation이면 target posterior도 같아야 한다. 그 경우 action variation은 instance-specific information이 아니라 numerical variation일 수 있다.\n')
table(['Condition','Same-evidence active pair disagreements','Unpooled MSE','Pooled MSE','Pooled − unpooled [95% interval]'],[[q,f"{d['same_evidence_disagreement']}/{d['same_evidence_active_pairs']}",f"{d['unpooled_mse']:.7f}",f"{d['pooled_mse']:.7f}",ci(d['paired_ci'])] for q,d in pooled.items()])
add('이 posthoc control은 동일 inference problem의 independent Monte Carlo calculations를 pool하고 pooled current mean으로 utility를 다시 계산한다. Actual outcomes는 pooling에 사용하지 않았다. 서로 다른 worlds의 likelihood를 곱하거나 같은 물체를 반복 측정했다고 가정한 것이 아니다. Primary policy와 stopping rule에는 사용하지 않았다.\n')

add('## 8. Adaptive refinement의 완료 범위와 남은 부분\n')
add('Main outcome를 보기 전에 practical confidence 기준에 따라 world별 outer budget을 64→128→256→512로 늘리는 규칙을 고정했다. Bridge/full method disagreement의 full-utility regret도 같은 tolerance 아래여야 한다. Maximum budget에서도 남으면 unresolved로 보고하고 outcome 평균에서 제외하지 않는 계획이다. Adaptive stopping 뒤 bootstrap은 strict sequential confidence guarantee가 아니다.\n')
add(f"128 outer 대상 57 worlds 중 **{partial['completed']} worlds만 완전히 완료**됐다. 구성은 {partial['families']}이며 deflection refinement는 아직 완료되지 않았다. Completed case IDs: `{partial['cases']}`.\n")
table(['Condition','이 20 worlds의 budget','Bridge/full agreement','Bridge practical confidence','Full practical confidence'],[
    [q,'64 → before' if stage=='before' else '128 → after',pct(d['method_agreement']),pct(d['bridge_practical_confidence']),pct(d['full_practical_confidence'])]
    for q,st in partial['summary'].items() for stage,d in st.items()])
add('위 표는 selected 20 worlds의 numerical diagnostic이다. 전체 test population의 stabilized-policy result가 아니다. 이 subset에서 confidence가 개선됐다는 사실만으로 전체 48-world selection 이득이 유지된다고 결론 내리지 않는다.\n')
table(['작업','상태'],[['Main64 matched comparison','72/72 완료;1440 batched runs,92160 conditional posterior populations'],['기존 ESS failure audit','8 hypothetical views 완료; bridge/full 각각4096 particles,18 moves,2 runs'],['One-step SNIS control','72/72 완료;512 outer,4096 bank particles'],['Gaussian/API validation','통과'],['Outer128 refinement','20/57 worlds 완료; 그 외 완료된 population files 보존'],['Outer256/512 refinement','아직 미실행'],['Increased-inner-budget audit','case1/25/49/66/68의128 outer,2048 particles,18 moves,2 runs — 아직 미실행'],['최종 stabilized-policy evaluation','아직 미완료']])

add('## 9. 비용, validation, 실행 기록\n')
table(['Condition','Bridge SMC seconds','Full SMC seconds','Full/bridge time ratio','Full/bridge rollout ratio'],[[q,f"{d['cost']['bridge']['seconds']:.1f}",f"{d['cost']['full']['seconds']:.1f}",f"{d['cost']['full']['seconds']/d['cost']['bridge']['seconds']:.2f}×",f"{d['cost']['full']['rollouts']/d['cost']['bridge']['rollouts']:.2f}×"] for q,d in s.items()])
add('이 표는 execution 방식 변경 전의 Main64 matched budget이다. Seconds는 batched SMC call의 elapsed-time 합이며 전체 wall-clock이나 GPU busy time이 아니다. 두 quality의 dynamic-probe 계산은 공유되므로 두 행을 합하면 중복된다. Factual-bank 생성, query-summary prediction, I/O, analysis는 이 timer에 포함하지 않는다. Equal particle/MH budget 비교이며 equal-wall-time frontier는 아니다.\n')
add('Bounded correlated Gaussian posterior를 dense quadrature와 비교해 normalized posterior mean의 최대 absolute error 0.001184를 확인했다. 세 family에서 scalar/population factual API와 factual-bank initialization의 동일 input/RNG 결과는 bitwise-identical했고, 서로 다른 factual observations의 batched likelihood도 개별 계산과 bitwise-identical했다. `safe_invalid` proposal-evaluation 변형도 별도 equivalence 검증을 했으나 main/refinement에는 적용하지 않았다.\n')
add('Outer128 일부부터 같은 family의 최대 4 worlds를 함께 처리해 kernel-launch overhead를 줄였다. Target, particle/MH budget, outer draws, stopping criterion은 유지했다. 실제 grouped RNG stream은 group 구성에 따라 달라지며 group IDs/seed를 기록했다. Grouped per-world seconds/rollouts는 batch total의 equal-share attribution이고 실제 batch totals도 저장했다.\n')
add(f"실제 runtime: Python {runtime['python']}, PyTorch {runtime['torch']}, NumPy {runtime['numpy']}, CUDA {runtime['cuda']}, `{runtime['device']}`, float64.\n")
add('사용 allocation은 처음에 2676323(node56)과2676324(node55)였다. Node55에 외부 training PID1799084가 나타나 우리 step2676324.652만 중단했다. 이후 node56에 외부 PID223742가 나타나 우리 step2676323.678만 중단했다. 다른 두 GPU도 외부 training PID1797893/1798018을 사용 중이었다. 기존 allocation과 외부 training step은 변경하지 않았다.\n')
add('Server directory: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tempered_bridge_v6`. `results/resource_state.json`, `results/interrupted_claims`, worker logs, population JSON/NPZ에 상태를 보존했다. 재개 절차는 `RESUME_KO.md`, 설계는 `PROTOCOL.md`, 구현은 `bridge.py`, `run.py`, `grouped.py`, `pipeline.py`에 있다. `server_run_manifest.json`은 파일 hash와 입력 provenance를 기록한다.\n')
add(f'[재개 절차]({root}/RESUME_KO.md) · [실험 설계]({root}/PROTOCOL.md) · [SMC 구현]({root}/bridge.py) · [결과 요약]({root}/results/analysis/summary.json)\n')

add('## 10. 지금 논문 기여로 주장할 수 있는 범위\n')
add('현재 강한 결과는 **proposal mismatch와 weight degeneracy를 구분하고, full-reference에 가까운 conditional posterior를 얻는 검증 가능한 절차**다. Factual-posterior 방향 전환, tempering/rejuvenation, inner MC error, outer ranking uncertainty를 분리한 점이 중요하다.\n')
add('현재 결과만으로 “새로운 SMC algorithm”, “일반적인 observation selection 성능 향상”, “기존 cheap SNIS보다 확실히 우수한 policy”를 주장하기는 어렵다. High에서 strong fixed rule 대비 개선을 확인하지 못했고, pixel_only의 결과도 아직 numerical convergence 검증이 끝나지 않았다. 동일 evidence에서 다른 선택이 나오는 현상도 일부는 Monte Carlo artifact였다.\n')
add('다음 우선순위는 저장된 refinement와 higher-inner-budget audit를 완료하고, 독립적인 새로운 worlds 및 여러 observation-noise realizations에서 결과를 확인하는 것이다. 그 뒤 contact-aware proposal/conditional transport가 같은 accuracy를 더 적은 forward budget으로 달성하는지 equal-compute frontier로 비교할 수 있다.\n')
add('이번 문제는 known camera/family와 calibrated image features를 쓰는 bounded 2D rigid-contact diagnostic이다. Factual vx만 사용해 controlled ambiguity를 만든 설정이며 일반적인 full-video inference와 다르다. Shared factual bank가 놓친 mode나 simulator/sensor mismatch를 모두 검증한 것은 아니다. Real-video 실험은 v6에서 새로 수행하지 않았고, v5 contour audit를 이번 physics validation으로 간주하지 않는다.\n')
add('**Adaptive refinement와 increased-inner-budget audit는 미완료다. GPU가 다시 비워졌다는 사용자 안내 후 기존 checkpoint부터 재개한다.**\n')
dest=(root.parents[2]/'outputs' if str(root).startswith('/Users/') else root/'reports')/'physics_scene_belief_tempered_bridge_v6_interim_report.md';dest.parent.mkdir(exist_ok=True);dest.write_text('\n'.join(L));print(dest)
