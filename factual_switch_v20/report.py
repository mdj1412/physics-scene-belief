"""Build the Korean report and loss/latency figure from the frozen server outputs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
r=root/"results"
def read(file):return json.loads((r/file).read_text())
def fmt(v):return f"{v:+.6f}"
def interval(v):return "—" if v is None else f"{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]"
def table(cols,rows):
    return "\n".join(["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])
summary=read("summary.json")
audit=read("audit.json")
timing=read("timing.json")
cfg=read("gate.json")
main=summary["0"]
shift=summary["1"]
teacher=cfg["routed"]["method"]
sections=["# v20 · Factual evidence로 observation을 바꿀 world를 고를 수 있는가", (root/"FINDINGS_KO.md").read_text(),
"""## 무엇을 독립적으로 바꿨는가

Factual evidence는 경계 주변에서 얻은 9개 boundary coordinates와 충돌 후 속도 2개다. 같은 evidence를 설명하는 scene hypotheses로부터 query 지점에서 stick이 일어날 확률을 예측한다. 연구자가 추가로 고를 수 있는 observation은 query view, factual view, probe 세 가지다. Probe를 기준으로 다른 view를 선택했을 때의 이득을 **probe의 Brier loss에서 그 view의 Brier loss를 뺀 값**으로 정의했다. 양수이면 바꾼 편이 낫다.

C는 replica당 256개 hypotheses를 resampling한 compact posterior mixture, E는 독립적인 16,384-hypothesis external posterior bank다. 첫 축은 hypothetical observation Y의 marginal predictive law다. 두 번째 축은 같은 Y를 주었을 때 binary label의 조건부 확률 p를 제공하는 source다. 두 축에 C 또는 E를 적용하므로 CC·EE는 일관된 joint distribution, CE·EC는 의도적으로 교차한 distribution이다. 세 번째 축은 p로 Bernoulli label을 하나 뽑아 점수를 계산하는 sample 방식과 두 label을 해석적으로 적분하는 mean 방식이다.

모든 cell은 같은 compact decoder의 출력 q를 채점한다. Sample 방식은 (Z−q)², mean 방식은 p(1−q)²+(1−p)q²를 사용한다. Source가 C이면 p=q여서 mean score는 q(1−q)가 된다. E이면 매 hypothetical Y에 대해 external posterior의 Gaussian likelihood update로 p를 계산한다. E에서 observation만 뽑는 비용과 E에서 conditional probability까지 계산하는 비용이 다르다. 세 action이 outer random coordinates를 공유하고 sampled cells는 공통의 독립 IID uniforms를 사용했다. V19의 sampled label은 generating component에서 읽었지만 이번에는 지정한 source의 p로부터 다시 뽑아 교차 조작을 가능하게 했다.

각 replica에 4 scrambles × 128 outer samples, 독립적인 compact representatives 두 벌을 썼다. V18 fixed global mixture는 기존의 K128/M4096 direct kernel로 다시 실행했다. 새 여덟 cell은 M512이므로 factorial 내부에서 한 축을 바꾼 비교가 그 요인의 효과다. Fixed global과 비교하면 budget·compression 세부 구현·scoring이 함께 달라진다. 모든 방법은 선택 후 같은 full bank decoder로 실제 synthetic observation을 처리한다.
""",
"""## 어떤 world에서 바꿀지 예측하는 방법

Train 1,024 worlds에서는 각 world에 세 observation을 모두 가한 simulated potential outcomes를 16개 noise repeats로 평가했다. 각 view의 target은 해당 world에서 probe를 사용했을 때와의 평균 Brier 차이다. 실제 적용 때 입력은 observation 전 factual evidence, 이 evidence로 계산한 query context, posterior stick probability, ESS, contact-law masses, 여섯 continuous parameters의 posterior mean과 SD뿐이다. Hidden world parameter, actual intervention 결과, unselected observation은 inference 시 입력에 들어가지 않는다.

Ridge regression은 public features와 제곱항으로 두 view의 gain을 추정한다. Train 통계로 표준화한 값을 ±8에 제한하고 penalty를 0.05×N으로 고정했다. 두 예측 중 큰 값을 gate score로 사용한다. Direct는 score가 threshold를 넘으면 예측한 view로 바로 바꾼다. Routed는 같은 score로 factorial selector를 호출할지 정하고, 호출한 selector가 probe를 다시 선택할 수도 있다. 나머지 worlds에서는 probe를 수행한다.

Validation 384개에서 후보 threshold와 selector 조합의 mean Brier loss를 비교했다. Direct는 모든 world에 probe를 쓰는 규칙을, routed는 E-C-mean과 validation score 하위 quartile 경계를 선택했다. Routed의 threshold는 −0.022512이며 validation에서 288/384번 호출했다. Validation은 threshold와 selector를 고른 자료이므로 성능 평가는 새 worlds에서 진행했다. Gate를 먼저 고정하고 matched와 shifted 조건의 test actions를 모두 저장한 후 private test outcomes를 열었다.

Held-out 1,024 worlds 각각에 실제 noise를 16번 반복했다. Shifted condition은 같은 worlds와 noise에서 flexible contact law만 바꾸며 inference는 nominal로 유지했다. Primary endpoint는 matched stick에서 routed minus baseline의 paired mean loss difference다. 두 baseline에 대한 family-adjusted 97.5% CI를 world 단위 bootstrap 20,000회로 계산했다. 음수면 routed가 유리하다. Secondary 95% intervals는 exploratory 비교에 사용했다.
""","## Held-out Brier loss와 online compute"]
names=["probe","fixed_global","direct","routed"]+[a+"-"+b+"-"+c for a in ["C","E"] for b in ["C","E"] for c in ["sample","mean"]]
sections.append(table(["Policy 또는 Y–label–mode","Matched Brier ↓","Shifted Brier ↓","Matched query/factual/probe","GPU mean / p50 / p90 ms"],[
[n,f"{main['rows'][n]['loss']:.6f}",f"{shift['rows'][n]['loss']:.6f}"," / ".join(map(str,main['rows'][n]['actions'])),f"{timing[n]['mean']:.2f} / {timing[n]['p50']:.2f} / {timing[n]['p90']:.2f}"] for n in names]))
sections.append(table(["Routed minus comparator","Matched 97.5% CI","Shifted 95% CI"],[
[n,interval(main["contrasts"][n]),interval(shift["contrasts"][n])] for n in ("probe","fixed_global")]))
sections.append("""GPU timing은 RTX PRO 6000 Blackwell에서 held-out 첫 32 matched worlds를 3회씩 측정했다. 순서를 섞고 batch 1·float64에서 GPU synchronization을 time boundary에 적용했다. Factual fit, 필요한 full/compact acquisition, gate feature의 CPU 전송·ridge prediction, 선택된 observation update를 포함한다. Static bank 및 QMC plan 생성, offline simulator supervision·ridge fitting, 실제 sensing/reset 시간은 제외했다. 표의 비용은 online per-world 비용이다. Timing에서 산출한 모든 action은 사전에 고정한 actions와 일치했다.

![Held-out Brier loss, GPU latency and gain calibration](results/v20_results.png)

## Factorial contrasts

각 행은 8개 cell의 **선택 후 실제 held-out loss**에서 계산했다. Main effect는 다른 축을 균등 평균하고 E−C 또는 mean−sample을 취했다. Interaction은 대응하는 차이의 차이다. 음수는 해당 방향으로 바꿀 때 loss가 낮아지는 경향이다. 한 bank와 한 acquisition randomization에 조건부인 secondary intervals다.""")
sections.append(table(["Factor contrast","Matched paired 95% CI","Shifted paired 95% CI"],
[[k,interval(v),interval(shift["effects"][k])] for k,v in main["effects"].items()]))
sections.append(table(["한 축만 바꾼 transition","Matched 선택 변경 /1024","Shifted 선택 변경 /1024"],
[[k,v,shift["flips"][k]] for k,v in main["flips"].items()]))
sections.append("""## Factual score가 예측한 switch value

Validation score quartile 경계를 test에 그대로 적용해 각 구간의 gain을 측정했다. Direct-view gain은 gate threshold를 적용하기 전 ridge가 제안한 view를 항상 택할 때의 이득이다. Selector gain은 선택된 E-C-mean을 그 구간 모두에서 호출할 때의 이득이다. 배포한 routed policy는 gate를 통과한 world에만 이 selector를 쓴다. 아래 구간별 값은 subgroup 평균으로, 한 world의 intervention 성공을 보장하지 않는다.""")
sections.append(table(["조건 / score 구간","n","평균 predicted gain","실측 direct-view gain [95% CI]","실측 selector gain [95% CI]","실제 switches","Median posterior minority mass"],
[[f"{law} Q{i+1}",v['n'],fmt(v['predicted']),interval(v['direct']),interval(v['teacher']),v['gate_switch'],f"{v['minority_median']:.5f}"]
for law,c in [("matched","0"),("shifted","1")] for i,v in enumerate(summary[c]["strata"])]))
for cname,label in [("0","Matched"),("1","Shifted")]:
    g=audit["groups"][cname];m=summary[cname];sw=m["rows"]["routed"]
    sections.append(f"{label}에서 routed selector를 {m['teacher_called']}/1024번 호출하고 실제로 {sw['switch']}번 probe를 다른 observation으로 바꿨다. 변경한 worlds의 paired benefit는 {interval(sw['benefit_if_switched'])}였다. Factual score로 ridge 제안 view의 realized gain이 양수인지 판별한 AUROC는 {g['auc']:.3f}였고, Q4−Q1 direct-view gain 차이는 {interval(g['upper_lower_ci'])}였다. AUROC의 positive label은 16 repeats 후의 strict gain>0이어서 작은 numerical 차이에도 민감하다.")
sections.append("""## 추가 독립 검증 · 호출할 selector의 gain을 직접 학습하면 달라지는가

주 실험을 완료한 뒤 두 가지 한계를 발견했다. 첫째, ridge가 예측한 값은 direct view의 이득인데 routed rule은 다른 selector를 호출한다. 둘째, primary threshold grid에는 always-probe가 포함됐지만 모든 worlds에서 selector를 호출하는 always-call endpoint는 빠져 있었다. Ungated selector를 test baseline으로 별도 보고했으나 validation에서 routing 자체의 가치가 과대평가될 여지가 있다.

추가 protocol을 먼저 기록하고, 같은 development 자료에서 E-C-mean과 fixed global 각각의 실제 train gain을 target으로 ridge 두 출력을 학습했다. Threshold 후보에 always-call과 never-call을 모두 넣었다. 기존 matched test와 겹치지 않는 새 1,024 worlds(16 repeats)를 생성해 맞춤 gate 두 개를 평가했다. 이 추가 분석의 설계는 주 결과를 본 뒤 정했고, 추가 test의 choices는 그 outcomes를 열기 전에 고정했다. Target과 threshold grid를 함께 바꿨으므로 두 수정의 효과를 각각 추정한 비교는 아니다.""")
extra=read("aligned/summary.json")
et=read("aligned/timing.json")
sections.append(table(["새 worlds의 policy","Mean Brier loss","Switches /1024","Matched GPU p50 / p90 ms"],
[[n,f"{v['loss']:.6f}",v['switch'],f"{et[n][0]:.2f} / {et[n][1]:.2f}"] for n,v in extra.items()]))
sections.append(table(["Matched selector target / comparator","Loss difference [98.75% family-adjusted CI]"],
[[n+" minus "+k,interval(v)] for n,entry in extra.items() if "contrasts" in entry for k,v in entry["contrasts"].items()]))
sections.append("""Validation에서 두 gate 모두 모든 worlds에서 selector를 호출하는 endpoint를 선택했다. 새 test에서 E-C-mean gate는 probe 대비 −0.000362, fixed global 대비 −0.000288이었고 두 adjusted CI가 모두 0을 포함했다. Fixed-global gate는 원래 fixed global과 정확히 같은 observation을 골랐다. Gate feature 계산을 실제로 포함한 p90은 E-C-mean에서 4.98 ms, fixed global에서 7.82 ms였다. 이처럼 gate가 일정하면 배포 시 해당 계산을 생략할 수 있다. 표에는 실제 측정한 경로의 비용을 기재했다. 주 test와 추가 test는 표본이 달라 각 policy의 절대 loss가 달라질 수 있으며 비교는 동일한 worlds 안에서만 수행한다.

주 test에서 Q1/Q4의 posterior minority mass median은 약 0.237/0.0075였다. 강한 harm은 query event가 더 불확실한 하위 score 집단에서 관찰됐다. 그러나 단일 feature가 그 차이를 일으켰다고 판단할 수 없다. Matched test에서 direct view의 realized gain이 양수인지를 판별한 AUROC도 0.462였다. 몇 건의 큰 손해가 집단 평균을 움직일 수 있으므로, gain의 부호를 잘 맞히는 것과 평균 손실을 줄이는 것을 각각 확인해야 한다.""")
sections.append("## 검증 범위와 재현\n\n원본 simulator의 Gaussian kernels를 physical coordinates에서 다시 계산해 validation 한 world의 세 action·여덟 cell risk를 비교했다. 최대 절대 차이는 "+f"{audit['risk_max_abs_error']:.3e}였다. "+"""Source·gate·test-choice freeze hashes를 확인했다. 양 측정 조건은 같은 test worlds를 공유하며 bank seed와 acquisition seed는 고정됐다. 실제 카메라 입력·full trajectory·robot control loss는 이 실험에서 직접 측정되지 않았다. Hindsight best action은 같은 repeats를 보고 최솟값을 고르므로 배포 가능한 성능으로 해석할 수 없다.

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/factual_switch_v20`다. `PROTOCOL.md`, `policy.py`, `experiment.py`, `analyze.py`가 사전 설계와 구현이며 `audit.py`는 독립 계산과 추가 strata 진단이다. `results/ridge.npz` 및 `gate.json`에는 fitted policy, choices_0/1에는 public-side decisions, loss_0/1에는 held-out per-action losses가 있다. 같은 상대 경로에 v18/v19 dependencies를 둔 새 폴더에서 할당된 idle GPU에 v18 guard를 적용해 experiment.py를 실행하고 compute-node CPU에서 analyze.py, audit.py, report.py 순으로 실행한다.""")
sections.append((root/"REVIEW_KO.md").read_text())

fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout="constrained")
short=["probe","fixed_global","routed",teacher]
ax=axes[0]
for n,s in zip(short,["always probe","fixed global","routed","ungated selector"]):
    x,y=timing[n]["p90"],main["rows"][n]["loss"]
    ax.scatter(x,y,s=55);ax.annotate(s,(x,y),xytext=(3,6),textcoords="offset points",fontsize=8)
ax.margins(x=.25,y=.25);ax.set(xlabel="GPU p90 per world (ms)",ylabel="Held-out mean Brier loss",title="Matched stick event")
ax=axes[1]
for key,label,off in [("direct","Direct view",-0.06),("teacher","Selector",.06)]:
    vals=[v[key] for v in main["strata"]];x=[i+off for i in range(1,5)]
    ax.errorbar(x,[v[0] for v in vals],yerr=[[v[0]-v[1] for v in vals],[v[2]-v[0] for v in vals]],marker="o",capsize=3,label=label)
ax.axhline(0,color="0.4",ls=":");ax.set(xticks=[1,2,3,4],xlabel="Predicted gain quartile (validation cut points)",ylabel="Probe loss minus selected observation loss",title="Held-out subgroup gains (95% CI)");ax.legend()
fig.savefig(r/"v20_results.png",dpi=160)
(root/"REPORT_KO.md").write_text("\n\n".join(sections)+"\n")
