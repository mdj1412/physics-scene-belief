from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
results = root / "results"
def read(name):
    return json.loads((results / name).read_text())
summary = read("summary.json")
timing = read("timing.json")
diag = read("diagnostics.json")
audit = read("outer_audit.json")
stick = summary["0_1"]["rows"]

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), layout="constrained")
colors = {"conditional":"#a54d22","self_label":"#387098","external":"#337859"}
for name, color in colors.items():
    rows = [f"{name}/{m}" for m in [512, 2048, 4096]]
    ax[0].plot([timing[r]["p90"] for r in rows],[stick[r]["loss"] for r in rows],
               marker="o", color=color, label=name.replace("_"," "))
    for size, r in zip([512,2048,4096],rows):
        ax[0].annotate(str(size),(timing[r]["p90"],stick[r]["loss"]),xytext=(3,4),textcoords="offset points",fontsize=7)
ax[0].axhline(stick["probe"]["loss"],color="#555",ls=":",label="always probe")
ax[0].set(xlabel="Stick GPU p90 latency (ms)",ylabel="Mean observed Brier loss",title="512 matched worlds")
ax[0].legend(fontsize=8)
for i,name in enumerate(("conditional","self_label")):
    ax[1].plot([512,4096],audit["rms_within_seed_sd"][i],color=colors[name],marker="o",label=name.replace("_"," "))
ax[1].set(xlabel="Outer samples per replica",ylabel="RMS within-world SD of probe–query gap",
          title="64 worlds, fixed representatives, 8 outer seeds",xticks=[512,4096])
ax[1].legend(fontsize=8)
fig.savefig(results/"tradeoff.png",dpi=160)
plt.close(fig)
def tbl(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(map(str,r))+" |" for r in rows])

parts = ["""# v19 · Conditional Brier risk로 contact observation을 고르는 실험

2026년 11월 14일. 새로 생성한 512 worlds에서 **hypothetical label을 적분하면 고정된 scene hypotheses 안의 numerical noise는 줄었지만, 실제로 더 좋은 observation을 선택한다는 증거는 얻지 못했다.** 같은 compact mixture로 비교한 matched stick-event에서 바뀐 선택은 14/512건이었다. Independent outer bank로 평가 조건을 바꾸면 113/512건이 바뀌었다.

이 보고서의 앞부분은 서버 실험, 뒷부분은 이 실험에 근거를 제공한 네 원문의 비교·knowledge map·gaps·연구 질문 10개·project review다. ‘실제 loss’는 simulator가 감춰 둔 world에서 선택 후 측정한 event-probability Brier loss를 가리킨다.

## 어떤 observation을 고르는가

Factual boundary profile과 충돌 직후 속도를 본 뒤, 멀리 떨어진 지점의 boundary profile을 보는 query view, 가까운 boundary를 정밀 측정하는 factual view, 다른 속도로 접촉시켜 outgoing velocity를 측정하는 probe 중 한 가지를 고른다. 세 observation의 sensing cost는 같다고 두었다. 관심 event는 query 지점의 contact 여부와 접촉 후 stick 여부다. 주 검정은 이전에 순위가 불안정했던 stick에 두었다.

## 실험을 고른 이유와 구현

V18은 action gap에 두 종류의 numerical error가 섞인다는 점을 보였다. Component-only 대비 seed agreement가 높아졌지만, stick GPU median은 adaptive 18.52 ms, fixed region mixture 6.14 ms였다. 이번에는 매번 계산량을 늘리는 대신 하나의 계산을 바꿨다. Dinkel et al.이 finite design EIG 비교에서 사용한 conditional averaging에 착안해 binary predictive Brier risk에서 label을 정확히 평균했다. 이 실험은 Dinkel et al.의 EIG estimator나 전체 ADE 알고리즘을 재현한 것이 아니다. [Dinkel et al., §2.3](https://arxiv.org/html/2607.16933v1#S2.SS3)

Factual posterior에서 replica마다 256개 global representatives를 취해 discrete mixture를 만들었다. 각 hypothetical observation에 대하여 mixture decoder가 계산한 stick probability를 q라 하자. Self-label은 그 observation을 생성한 component의 label Z를 사용해 (Z−q)²를 계산한다. Conditional은 같은 observation에서 Z=0과 Z=1의 두 경우를 합친 q(1−q)를 사용한다. 동일한 mixture가 생성과 예측을 담당할 때 E[(Z−q)² | Y]=q(1−q)이다. 한 pilot case의 모든 해당 entries에서 직접 합산한 값과 공식의 최대 차이는 5.55×10⁻¹⁷이었다. 독립적인 outer law에서 실제 조건부 확률이 p라면 기대 손실은 p(1−p)+(p−q)²다. 이 차이가 self-consistent risk의 적용 범위를 정한다.

두 replicas와 4개의 outer scrambles를 사용했다. M은 replica당 512, 2,048 또는 4,096이며 두 replicas가 outer random coordinates를 공유한다. Conditional/self-label의 representatives와 hypothetical observations가 같으므로 이 비교가 label 적분만 바꾼 ablation이다. External-label은 같은 compact decoder를 사용하되 별도의 16,384-particle outer bank에서 hypothetical scene과 label을 가져온다. 이때 sampling law, targets, 외부 posterior fitting 비용이 함께 바뀐다.

모든 방법은 하나의 observation을 선택하고 공통 16,384-particle full decoder로 예측한다. 이전 global/region fixed mixtures, adaptive allocation, full nested2048을 새 banks와 같은 worlds에 다시 실행했다. 두 fixed rules도 포함했다. Primary acquisition에서 inner/outer bank seeds는 183911/193913이다. 두 acquisition seeds는 213901/243911이다. Matched control의 첫 64 worlds에서 별도 bank pair와 별도 outer seed로 nested4096 reference를 계산했다. 이 reference도 finite-sample estimate다.

## Matched stick 결과

Loss는 synthetic world당 16개의 paired observation-noise repeats를 먼저 평균했다. Agreement는 representatives와 outer scrambles를 함께 바꾼 두 acquisition seeds에서 동일한 action을 고른 world 비율이다. Reference regret는 첫 64개 worlds에서 선택한 action의 reference risk에서 최소 reference risk를 뺀 값의 평균이다. GPU timing은 첫 32개 test worlds에서 방법 순서를 섞어 3회 측정했고, batch 1·float64에서 factual fit, 선택, 선택된 observation update까지 포함한다. Static bank/plan 생성과 physical sensing은 제외했다."""]

parts.append(tbl(["Method","Brier loss ↓","Seed agreement ↑","Reference regret ↓","GPU p50 / p90 (ms)","8 ms 초과율"],[
    [name,f"{row['loss']:.6f}",f"{row['agreement']:.2%}" if name not in ("probe","query") else "고정 rule",
     f"{row['reference_regret']:.6f}",f"{timing[name]['p50']:.2f} / {timing[name]['p90']:.2f}",f"{timing[name]['over8']:.1%}"]
    for name,row in stick.items()]))
parts.append("""별도 pilot 12개 worlds에서 방법별 p90이 8 ms 이하인 가장 큰 M을 미리 선택했다. 세 mixture 방법 모두 M4096을 선택했고 test p90도 8 ms 이내였다. 따라서 같은 시간 상한으로 고른 비교와 M4096 직접 비교는 동일한 관측값이다. 이 실험은 각 방법을 8 ms까지 채웠을 때의 최적 성능을 측정하지 않았다. 세 후보 중 더 큰 M을 시험했을 때의 효과는 열려 있다.

아래 차이는 conditional/4096의 loss에서 baseline의 loss를 뺀 값이다. 4개의 사전 가설에 Bonferroni를 적용한 98.75% paired world-bootstrap CI를 사용했다(20,000 resamples). 같은 상한에서의 비교가 외부 bank M4096과 중복돼 고유한 수치는 세 개다.""")
parts.append(tbl(["Primary matched stick contrast","Mean difference","Adjusted CI"],
  [[name,f"{v[0]:+.7f}",f"[{v[1]:+.7f}, {v[2]:+.7f}]"] for name,v in summary["0_1"]["contrasts"].items()]))
parts.append("""Conditional 대 self-label의 unadjusted 95% interval은 [−0.0002733, −0.0000037]이었으나 사전 family 기준에서는 0을 포함한다. 많은 comparisons 중 한 결과만 고르면 결론이 달라질 수 있다. 두 acquisition seeds의 Top-1 agreement 차이는 −1.37 percentage points였으며 paired 95% CI는 [−3.13, +0.39]였다. Conditional과 always-probe의 loss 차이는 +0.000360, exploratory 95% CI [−0.003286, +0.004454]였다. Always-probe는 GPU p90 1.73 ms여서 이 환경에서 특히 중요한 baseline이다.

![Matched stick loss, GPU latency and conditional numerical variability](results/tradeoff.png)

왼쪽 그림은 point estimates다. 오른쪽은 아래의 추가 진단 결과다. Hypothetical sample 수를 늘려도 realized Brier loss의 순위가 단조롭게 변하지 않았다.

## 어디에서 차이가 생겼는가

**고정된 representatives에서의 outer integration.** 결과를 본 뒤 추가한 진단으로 첫 64개 matched stick worlds에서 representatives를 고정하고 별도의 outer plans 8개를 사용했다. 미리 선택했던 probe-minus-query risk gap에 대해 각 world 안의 sample SD를 구하고 그 RMS를 보고한다. M512에서 self-label 0.003822가 conditional 0.001854로, M4096에서는 0.000533이 0.000257로 줄었다. M4096에서 같은 world의 8개 seeds로 만들 수 있는 28쌍의 평균 Top-1 agreement는 오히려 self-label 97.77%, conditional 97.21%였다. 고정된 gap 하나의 변동 감소와 세 action 중 최선의 선택은 다른 측정값이다.

**Mixture가 지운 label support.** Matched stick의 122/512 worlds에서는 양쪽 replicas 각각에서 representatives의 event label이 한 값뿐이었다. 이 122개가 conditional의 세 action risk가 모두 같아진 122개와 정확히 일치했다. Full factual bank의 minority probability는 이들에서 median 0.000023, 최대 0.00613이었다. 12개만 0.001을 넘었고 0.01을 넘는 case는 없었다. 고정된 tie rule은 query view를 선택했다. Support loss가 실제로 발생했지만, 이 수치만으로 평균 decision loss의 주원인이라고 판정할 수 없다.

**어떤 변경이 action을 바꿨는가.** Same-mixture label 적분으로 14개가 바뀌었고 external generation으로 바꾸면 113개, v18 fixed global로 바꾸면 119개가 달라졌다. 첫 64 worlds의 independent-bank nested reference에서 conditional/4096 regret는 0.000436, external/4096은 0.000029였다. 별도 approximation을 사용하는 reference인 만큼 reference agreement와 realized Brier loss를 함께 해석해야 한다. 이 결과는 다음 검증에서 representative support와 predictive law를 먼저 점검할 이유를 준다.

## Contact control과 shifted contact law""")
parts.append(tbl(["Method","Matched contact","Matched stick","Shifted-law stick · nominal inference"],[
    [m]+[f"{summary[c]['rows'][m]['loss']:.6f}" for c in ("0_0","0_1","1_1")]
    for m in ("conditional/4096","self_label/4096","external/4096","v18_global","v18_region","v18_adaptive","nested2048","query","probe")]))
parts.append("""Matched contact에서 conditional과 self-label은 512개 모두 같은 action을 선택했다. Conditional은 508건에서 query view를 골랐다. Shifted stick에서는 conditional 대 self-label의 차이가 +0.000174였고 secondary 95% CI는 [−0.000016, +0.000540]이었다. 두 조건은 같은 hidden worlds와 noise를 공유하는 paired conditions다. Shifted 결과에서 새로운 독립 표본 512개를 추가로 얻은 것으로 계산하지 않았다.

## 논문 기여와 다음 결정

**현재 뒷받침되는 주장은 좁고 구체적이다.** 이 simulator에서 binary conditional averaging은 고정된 compact belief 안에서 지정한 action-gap estimate의 outer 변동을 약 절반으로 줄였다. Strong fixed mixture 또는 always-probe 대비 실제 decision loss 개선은 확인되지 않았다. Conditional과 self-label의 계산 시간도 거의 같아, external 방법과의 속도 차이를 label 적분 자체의 효과로 귀속할 수 없다. Simulator, bank, device가 달라진 v18과 v19 사이의 절대 loss 차이를 성능 향상으로 해석하지 않는다.

**다음으로 선택할 질문은 “어느 world에서 다른 observation을 고를 만큼 predictive law가 달라지는가?”다.** Public factual evidence로 minority mass와 compact/external disagreement를 정의하고 disjoint pilot에서 threshold를 고정한다. Held-out worlds에서 global·label-stratified·independent-bank generation을 동일한 GPU budget으로 비교하고, 관측 전 예측한 이득과 관측 후 actual loss의 관계를 측정한다. Primary endpoint는 always-probe와 strong fixed mixture 양쪽에 대한 paired Brier loss difference로 둔다. 고정된 action-gap의 variance 감소만 재현된다면 numerical diagnostic으로 정리하는 편이 현재 증거에 맞다.

이 판단은 analytic local impulse model, 같은 비용의 세 observation, nominal inference, batch 1 GPU에서 얻었다. 실제 영상의 geometry estimation, 장치의 reset cost, robot intervention 성공률은 별도의 검증 대상이다. Acquisition replication은 두 seeds, independent-bank reference와 고정 representatives 진단은 64 worlds로 제한했다. 8 ms는 공학적 비교를 위해 선택한 상한이다.

## 실행 기록과 재현

서버 디렉터리는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/conditional_risk_v19`이다. 처음 node56에서 pilot 이후 다른 작업이 시작돼 guard가 이번 worker를 중단했다. 기록을 `interrupted_node56/`에 보존했다. 최종 표의 pilot·test·timing은 node55, allocation 2744468, RTX PRO 6000 Blackwell 한 장에서 처음부터 다시 측정했다. 완료된 worker의 종료 코드는 0이다.

`selector.py`는 public evidence를 받아 action을 정한다. `run.py`는 data generation, pilot, policy freeze, 실측 timing과 공통 decoder evaluation을 실행한다. `analyze.py`는 사전 contrasts, `diagnose.py`와 `outer_audit.py`는 exploratory diagnostics를 담당한다. `PROTOCOL.md`, `results/sources.json`, `results/freeze.json`에 분석 계획과 hash를 저장했다. 재현할 때는 새 디렉터리에 같은 상대 경로의 v18 dependencies를 두고 할당된 GPU에서 v18의 guard로 run.py를 실행한 뒤 compute-node CPU에서 analyze.py와 diagnose.py를 실행한다. Outer audit은 같은 GPU에서 별도 guard 디렉터리로 실행한다.""")
parts.append((root/"LITERATURE_KO.md").read_text().replace("# Contact observation selection에서 계산을 줄이면 실제 예측도 좋아지는가","## 문헌 검토와 연구 질문"))
(root/"REPORT_KO.md").write_text("\n\n".join(parts)+"\n")
