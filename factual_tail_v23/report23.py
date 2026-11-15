"""Build the Korean report and a research figure from measured server outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
r=root/"results"
def get(f):return json.loads((r/f).read_text())
def tab(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])
def num(x):return f"{x:.6f}"
def pct(x):return f"{100*x:.2f}%"
def ci(row):return f"{row[0]:+.6f} [{row[1]:+.6f}, {row[2]:+.6f}]"
def ms(xs):return " / ".join(f"{x:.3f}" for x in xs)
s=get("summary.json");d=get("design.json");names=get("methods.json")
p,c=[s["rows"][i] for i in ("precise","degraded")]
cost=np.load(r/"cost_ablation.npz");other=np.quantile(cost["times"].reshape(-1,4),[.5,.9],axis=0)
audit=np.stack([np.load(r/f"audit_{b}.npz")["detail"] for b in range(4)])
scores=np.stack([np.load(r/f"choices_{b}.npz")["score"] for b in range(4)])
flags=scores>=d["cutoff"]
a=np.stack([np.load(r/f"choices_{b}.npz")["acts"][:,0,0] for b in range(4)])-1
selected=audit[np.arange(4)[:,None],np.arange(512)[None,:],a]
slot_count=int(d["gate"][4:].split("m")[0])//4
parts=["# v23 · 잘못된 observation의 비용과 factual evidence에 따른 tail budget 배정",
"""**결과.** Query 위치에서 stick 여부를 예측하는 Coulomb impact worlds를 새로 만들었다. 같은 world에서 차선 observation을 택할 때의 cross-fitted Brier 비용은 0.149986이었다. Sensor precision을 낮추면 0.080741이었고, paired 증가분은 +0.069245 [95% CI +0.059908, +0.078932]였다. 잘못된 observation의 비용을 키우려던 실험 설계는 사전 기준을 통과했다.

**같은 시간에 더 정확했는가?** GPU p90 5.14–5.19 ms에서 factual gate의 평균 loss는 0.077172, top-K는 0.080369, fixed global mixture는 0.079527이었다. Gate가 네 bank 모두에서 두 baseline보다 낮았지만 두 primary confidence intervals가 모두 0을 포함했다. 같은 K에서 항상 tail을 보존한 방법의 loss는 0.077163이었다. 이번 결과는 tail 보존을 선택적으로 적용할 가능성을 보여주지만, factual routing이 성능 개선에 꼭 필요하다는 증거는 약하다.""",
"## 1. 어떤 worlds에서 잘못된 선택이 비싸지는가",
"""한 번의 factual 충돌을 본 뒤 세 actions 중 하나를 고른다. Action 1은 query 위치 주변의 boundary profile, 2는 factual 충돌 위치 주변의 정밀 profile, 3은 다른 입사 속도의 probe다. 예측할 것은 **query 위치 y=0.025에서 incoming velocity (1, 0.425)를 가했을 때의 stick 여부**다. Selected observation을 동일한 full inner-bank decoder에 넣고, 실제 binary outcome에 대한 (p−label)²를 측정했다.

World generator와 candidate bank에는 같은 quadratic boundary와 Coulomb impulse를 썼다. Family는 probability 1/2로 생성한다. Geometry family는 local slope가 음수인 쪽에 놓고 curvature를 ±0.30 주변, friction coefficient를 0.215 주변에서 변화시킨다. Friction family는 local slope가 양수인 쪽에 놓고 curvature를 0 주변, friction coefficient를 0.075 또는 0.365 주변에서 변화시킨다. Offset·slope·curvature·friction·restitution에 연속 변동을 더했다. 두 family에 동일한 physical query를 가한다. 전자는 factual 위치에서 보이지 않는 curvature의 부호, 후자는 약한 factual 충돌만으로 구별하기 어려운 sliding friction이 stick 예측을 바꾸도록 설계했다.

Factual incoming velocity는 (1, −0.10), probe는 (1, −0.85)다. Precise 조건에서 query view, factual view, probe의 각 좌표 noise standard deviation은 각각 0.00035, 0.001, 0.025다. Degraded control은 세 standard deviations를 모두 24배로 하고, 동일 world·factual evidence·query label·표준화한 noise를 유지했다. 모든 방법에 현재의 sensor calibration을 제공했다. Family, 실제 curvature/friction, 추가 observation의 실현값, 실제 stick label은 policy에 전달하지 않았다. Sensor 변경은 design manipulation이며 실제 장비 측정값은 없다.

V22의 실패 사례나 realized loss로 worlds를 선별하지 않았다. Simulator와 기준을 먼저 확정한 후 512개 worlds를 새로 추출했다. 239개가 geometry, 273개가 friction이었다. Stick prevalence는 각각 51.5%, 31.1%였다. Development 96개, timing calibration 48개, test 512개의 seeds는 서로 겹치지 않는다. 네 independent Sobol inner/outer bank pairs에서 각각 16,384개 hypotheses를 사용했다. 동일 world의 32 noise repeats와 네 bank는 paired measures다.""",
"## 2. 사전에 관측할 수 있는 사실로 tail budget을 정하는 방법",
"""Factual data x는 중심부 boundary sample 9개와 충돌 후 velocity 2개다. External bank에서 wᵢ=P(hᵢ|x), zᵢ를 각 candidate hypothesis의 stick label이라 하자. 전체 factual stick probability는 p=Σwᵢzᵢ다. Factual 상위 512개가 갖는 mass를 H, 그 안에서 renormalize한 stick probability를 pH라 두고 **score = 4p(1−p)(1−H) + |p−pH|**를 계산했다. 앞 항은 stick 여부가 불확실하고 tail mass가 클 때 커지며, 뒤 항은 factual truncation이 예측을 이미 바꾸고 있을 때 커진다. 이 값은 추가 observation이 무엇일지 모르는 상태에서 계산된다.""",
f"Development score의 median {d['cutoff']:.6f} 이상이면 K의 3/4은 exact head weights를 보존하고 1/4은 stick/non-stick label별 tail resampling에 균등 배정했다. 각 sample에 그룹 mass를 그룹 slots 수로 나눈 weight를 준다. Threshold 미만이면 모두 factual top-K에 사용했다. 선택된 K에서는 gate-on일 때 tail {slot_count}개, gate-off일 때 0개다. Test gate-on 비율은 {pct(s['gate_rate'])}로, 평균 {slot_count*s['gate_rate']:.1f}개의 tail slots를 사용했다. 총 K는 일정하므로 slots의 감소량을 GPU 시간 절감으로 환산할 수 없다.",
"""Acquisition은 각 action의 hypothetical observation에서 compact decoder의 stick probability q와 external label posterior pY를 계산하고 pY(1−q)²+(1−pY)q²를 평균한다. Risk가 가장 낮은 action을 골랐다. Top-K와 gate는 같은 acquisition 함수를 쓴다. Fixed global은 v18의 original direct kernel로 K128, 4,096 outer observations를 사용하며, external bank에서 뽑은 labels로 expected loss를 계산한다. K의 의미는 global mixture와 external support에서 다르다. 모든 방법의 최종 probability는 같은 full inner bank가 계산하므로, 같은 action을 선택하면 realized Brier도 같다.
 
개발에서 M1024·2048·4096과 K를 함께 바꾼 13개 budgets를 각 family별로 timing했다. Fixed global p90에 가장 가까운 각 4개를 2차 timing하고, 세 pairwise p90 ratios가 0.94–1.06 안인 조합들에서 development Brier 합이 최소인 구성을 선택했다. 별도 48개의 calibration worlds를 24/24로 나눴고 test loss로 budgets를 수정하지 않았다. Primary 비교는 각 방법에 유리한 K,M을 허용한다. Gate의 순수 기여는 K,M을 통일한 강제 분기 ablation에서 평가한다.""",
f"확정된 세 방법은 {d['gate']}, {d['top']}, fixed global이다. Calibration p90은 같은 순서로 {ms([d['second'][key] for key in names[:3]])} ms였다. 모든 test actions를 파일로 고정하고 SHA-256을 기록한 뒤에만 evaluator가 실제 observation과 stick labels를 읽었다.",
"## 3. World 설계의 독립 검증",
"""각 world·bank에 대해 32회 noise 중 16회로 세 action의 Brier 순서를 정하고, 나머지 16회로 그 순서의 Brier 차이를 평가했다. 절반을 맞바꾼 결과도 평균했다. 이는 숨겨진 동일 world에 반복 실험할 수 있는 evaluator만 구성할 수 있는 진단이다. Deployment selector가 factual evidence로 아는 expected action gap과는 차이가 있다. 사전 통과 기준은 precise의 runner-up−best 평균 0.03 이상, precise−degraded 증가분의 paired 95% CI 하한이 0 초과였다.""",
tab(["교차 평가량","Precise [95% CI]","Degraded [95% CI]","Paired 증가 [95% CI]"],[[lab,ci(p["gap"][j]),ci(c["gap"][j]),ci(s["interaction"][j])] for j,lab in enumerate(["차선 − 최선 Brier","최악 − 최선 Brier"])]),
f"두 절반에서 같은 best action을 고른 비율은 precise {pct(p['split_agreement'])}, degraded {pct(c['split_agreement'])}였다. 별도 검증으로 설계 때 정한 family rule(geometry → query view, friction → probe)의 precise Brier는 {num(p['family_map'])}, 이 rule을 뒤집은 Brier는 {num(p['reverse_map'])}였다. Paired 차이는 {ci(p['map_ci'])}다. 두 rule은 평가 때만 family 정보를 받는다. 전체에서 관측한 큰 gap은 모든 world에서 최적 action이 다르다는 뜻은 아니다.",
"## 4. Held-out Brier와 compute",
tab(["정책","Precise Brier","Degraded Brier","Precise p50 / p90 / p95 ms","Degraded p90 ms"],[[name,num(p["mean_loss"][j]),num(c["mean_loss"][j]),ms(np.array(p["timing"])[:,j]),f"{c['timing'][1][j]:.3f}"] for j,name in enumerate(names[:3])]),
tab(["Gate 차이 (음수가 이득)","Precise crossed 97.5% CI","Precise world-only 97.5% CI","Degraded exploratory 95% CI"],[[lab,ci(s["primary"][j]),ci(s["world_ci"][j]),ci(s["degraded_ci"][j])] for j,lab in enumerate(["Gate − top-K","Gate − fixed global"])]),
f"Precise의 gate/top, gate/fixed, top/fixed p90 ratios는 {p['timing'][1][0]/p['timing'][1][1]:.4f}, {p['timing'][1][0]/p['timing'][1][2]:.4f}, {p['timing'][1][1]/p['timing'][1][2]:.4f}로 사전 허용 범위 0.90–1.10을 만족했다. Primary의 두 contrasts에 Bonferroni를 적용하고, world와 bank를 독립적으로 resample한 paired bootstrap 20,000회를 사용했다. Point estimates는 두 baseline 대비 개선 방향이다. 두 upper bounds가 모두 0 미만이어야 한다는 superiority 기준은 통과하지 못했다. Bank 수는 네 쌍이고 new contact law나 physical object로 인한 변동은 이 intervals에 포함하지 않았다.",
tab(["Bank pair seeds","Precise gate−top","Precise gate−fixed"],[[str(tuple(pair)),num(row[0]),num(row[1])] for pair,row in zip(s["seed_pairs"],p["perbank"])]),
"""Batch-1 float64를 한 장의 RTX PRO 6000 Blackwell Server Edition에서 측정했다. 48 held-out worlds × 3 repeats × 4 banks, 조건·방법별 576 timings이며 warm-up은 제외했다. 순서를 섞고 각 call 전후 CUDA synchronization을 수행했다. Factual fit, gate score·routing, acquisition, 선택된 observation의 posterior update가 포함된다. Static bank 및 sample plan 구축, sensor 동작 시간은 제외했다.""",
"### Gate가 실제 observation을 바꿨을 때",
tab(["Precise 비교","다른 선택 / 2,048 bank-worlds","한 번이라도 바뀐 고유 worlds / 512","다른 선택에서 gate−baseline Brier","전체 평균 차이"],[[lab,z['n'],z['worlds'],num(z['mean_delta']),num(s['primary'][j][0])] for j,(z,lab) in enumerate(zip(p['flips'],["Top-K","Fixed global"]))]),
"""공통 decoder 때문에 전체 차이는 ‘다른 action을 선택할 비율 × 그때의 평균 Brier 차이’다. 동률에 가까운 경우가 많았던 v22와 달리, 이번에는 실제로 갈린 선택에 loss 차이가 있었다. 다만 이 subset에서는 우연히 action이 바뀐 bank와 worlds가 반복되므로, subset 평균을 독립적인 효과로 일반화하기 어렵다.""",
"### Tail allocation 자체의 효용",
tab(["Precise 방법","Held-out Brier","Gate−비교 [exploratory 95% CI]","추가 timing p50 / p90 ms"],[[lab,num(v),ci(p['secondary'][i]),ms(other[:,j]) if j is not None else "—"] for i,(lab,v,j) in enumerate([("같은 K,M · 항상 top",p["mean_loss"][3],1),("같은 K,M · 항상 tail 보존",p["mean_loss"][4],2),("같은 비율 · random routing 기대값",p["random_route"],None)])]+[["Factual slope sign rule",num(p["cheap"]),ci(p["secondary"][4]),ms(other[:,3])],["Factual gate",num(p["mean_loss"][0]),"기준",ms(other[:,0])]]),
"""Random routing은 bank별 gate-on 빈도만 같게 두고 각 world에 분기를 무작위로 배정했을 때의 정확한 기대 loss다. 이를 별도의 GPU 정책으로 timing하지 않았다. Slope sign rule은 factual profile의 선형 slope가 음수면 query view, 그 외에는 probe를 택한다. 이 deterministic 규칙은 family를 보지 않으며, 같은 decoder update를 포함했다. 이 두 대조 및 추가 timing은 exploratory analysis다. Forced branches도 같은 K,M에서 비교했으므로 primary의 독립 budget 선택이 만드는 confounding을 확인할 수 있다.""",
f"항상 tail을 쓰는 방법은 매번 {slot_count} tail slots를 사용했고, factual gate는 평균 {slot_count*s['gate_rate']:.1f}개였다. 직접 측정한 p50에서는 gate {other[0,0]:.3f} vs always-preserve {other[0,2]:.3f} ms, p90에서는 {other[1,0]:.3f} vs {other[1,2]:.3f} ms였다. Gate의 Brier는 always-preserve와 +0.000010 차이였다. 동등성 margin을 둔 시험은 수행하지 않았다.",
tab(["Precise family / 분기","bank-worlds","Gate-on","Runner-up gap","Gate / top / fixed Brier"],[[lab,z["n"],pct(z["gate"]),num(z["gap"])," / ".join(num(v) for v in z["mean_loss"][:3])] for lab,z in p["subgroup"].items()]),
f"작은 support K512/M512에서 top의 Brier는 {num(p['mean_loss'][5])}, gate는 {num(p['mean_loss'][6])}였고 차이는 {ci(p['secondary'][3])}였다. Full external label reference(M512)의 Brier는 {num(p['mean_loss'][7])}로, finite bank와 outer sampling에 의존한다. Single-action controls의 precise loss는 query view {num(p['cost_by_action'][0])}, factual view {num(p['cost_by_action'][1])}, probe {num(p['cost_by_action'][2])}였다.",
"### 실제 추가 observation 이후 tail은 어떻게 달라졌는가",
"""Primary가 끝난 뒤, 동일한 test outputs의 precise 조건에서 gate가 사용하는 K를 고정하고 full external bank로 실제 observation별 posterior update를 재계산했다. 아래 factual/updated tail mass는 **factual top-K 밖**의 mass이고, error는 full stick probability 대비 support approximation의 mean absolute error다. V22의 recovery는 factual mass ≤5%, updated mass ≥10%, 동시에 5배 이상 증가한 event다. Factual posterior 자체가 넓으면 첫 조건에서 제외되므로 이 기준과 연속적인 mass/error를 함께 본다.""",
tab(["공통 K · 집단","Factual tail mass","Updated tail mass","Top / 보존 label error","V22 recovery %"],[[lab,num(ar[...,0].mean()),num(ar[...,1].mean()),num(ar[...,3].mean())+" / "+num(ar[...,4].mean()),pct(ar[...,5].mean())] for lab,ar in [("모든 candidate actions",audit),("Gate-on · 모든 actions",audit[flags]),("Gate-off · 모든 actions",audit[~flags]),("Gate가 실제 선택한 action",selected)]]),
"Gate-on에서 이미 factual tail mass가 평균 33.2%였고, 보존을 켜면 full posterior 대비 label error가 0.0570에서 0.0213으로 줄었다. 그러나 v22식 희귀 recovery는 이 집단에서 0건이었다. Gate는 처음부터 넓게 퍼진 posterior를 포착했다. 탈락한 작은 mass가 뒤늦게 살아나는 world를 예측했다는 주장을 이 지표가 뒷받침하지는 않는다.",
"추가 timing에서는 always-preserve가 gate보다 p50과 p90에서 모두 빨랐다. 특히 p90은 4.726 대 5.181 ms였다. 이 구현의 adaptive routing은 tail slots를 덜 쓰면서도 score 계산과 분기 비용을 만회하지 못했다. 현재 비교에서 성능과 latency를 함께 고려하면 always-preserve가 후속 검증의 우선 baseline이다.",
"![Sensor precision별 action gap과 latency-matched Brier 비교](results/v23.png)",
"## 5. 논문으로 주장할 수 있는 부분",
"""**잘못된 선택의 비용을 식별한 실험 설계는 기여 후보가 된다.** Hidden physical family를 사후에 선별하지 않고 생성했으며, 같은 world·query·noise에서 sensor precision만 조작해 선택 비용의 증가를 독립적으로 확인했다. Privileged family mapping, factual slope rule, same-K forced routing을 함께 두어 benchmark가 단순 cue만으로 풀리는 정도와 numerical approximation의 기여를 분리할 수 있다.

**새 selector의 효율 우위는 아직 입증되지 않았다.** 강한 두 baseline보다 낮은 point estimate가 모든 bank에서 반복됐지만, 같은 512개 worlds를 사용했으며 unique disagreement worlds는 67개와 36개였다. 특히 always-preserve가 거의 같은 loss를 내므로, 현재의 두 항짜리 factual score만을 독립적인 algorithmic novelty로 내세우기는 어렵다. 반대로 작은 평균 차이만 보고 observation 선택 자체가 중요하지 않다고 판단할 수도 없다. 잘못된 action의 잠재 비용과 실제 경쟁 정책끼리 다른 action을 고르는 빈도가 다르기 때문이다.

**다음 검증에서 우선할 것.** 이 결과의 독립 replication에는 ① sensor calibration·physical prior를 바꾼 새 worlds에서도, ② 고정한 factual gate가 always-preserve와 비교해 사전 설정한 noninferiority margin 안의 Brier를 유지하고, ③ 비용 목표인 p50 또는 p90을 실제로 줄이는지를 함께 평가하는 것이 적절하다. Margin은 응용에서 허용하는 오류 비용으로 먼저 정해야 한다. Geometry와 friction을 family에 종속시킨 현재 prior에는 강한 slope cue가 있다. 두 변인을 독립적으로 조절하는 factorial control과 물체별로 분리한 real intervention data가 있어야 일반적인 scene belief에 대한 주장으로 확장할 수 있다.""",
"## 구현 및 재현",
"""서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/factual_tail_v23`이다. Simulator는 [physics23.py](physics23.py), gate는 [policy23.py](policy23.py), calibration·evaluation은 [run23.py](run23.py), 통계는 [stats23.py](stats23.py)에 있다. [사전 protocol](PROTOCOL.md), [latency calibration 원자료](results/design.json), [집계 결과](results/summary.json), [choices freeze](results/choices_freeze.json)를 함께 보존했다. GPU worker에는 allocation 2744468의 node55 한 장을 사용했다. 무결성 검사를 통과한 소스와 선택 파일의 hashes는 results/sources.json 및 freeze records에 있다.

재실행할 때는 이전 버전의 상대 경로를 유지한 새 output directory에서 Slurm allocation과 빈 GPU를 확인한다. 이전 v18 guard를 통해 run23.py를 실행한 후 compute node CPU에서 stats23.py를 실행한다. 추가 진단과 timing은 같은 GPU에서 audit23.py와 cost23.py로 순차 실행하며 report23.py는 CPU에서 표와 그림을 만든다. 최초 추가 진단 호출은 entry-point 경로 오류로 계산 전에 종료됐다. 경로를 수정한 재실행으로 네 bank 모두 검증했다.""",
(root/"LITERATURE_KO.md").read_text()]
fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
for j,(name,row) in enumerate([("Degraded",c),("Precise",p)]):
    mean,lo,hi=row['gap'][0]
    ax[0].bar(j,mean,color=["#95adb5","#296784"][j])
    ax[0].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],color="black",capsize=4)
ax[0].set(xticks=[0,1],xticklabels=["Degraded","Precise"],title="Cost of choosing the runner-up",ylabel="Cross-fitted excess Brier (95% CI)")
for j,(mean,lo,hi) in enumerate(s['primary']):
    ax[1].errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt="o",capsize=4,color="#296784")
ax[1].axvline(0,ls=":",color="0.5");ax[1].set(yticks=[0,1],yticklabels=["vs top-K","vs fixed global"],ylim=(-.5,1.5),title="Precise sensors, matched GPU latency",xlabel="Gate minus baseline Brier (97.5% CI)")
fig.savefig(r/"v23.png",dpi=155)
(root/"REPORT_KO.md").write_text("\n\n".join(parts)+"\n")
