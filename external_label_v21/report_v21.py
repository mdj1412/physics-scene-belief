from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
r=root/"results"
def load(n):return json.loads((r/n).read_text())
def table(cols,rows):
    return "\n".join(["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])
def ci(v):return f"{v[0]:+.6f} [{v[1]:+.6f}, {v[2]:+.6f}]"
def csv(v,precision=6):return " / ".join(f"{x:.{precision}f}" for x in v)
s=load("summary.json");d=load("design.json");p=load("parity/summary.json");pd=load("parity/design.json")
laws=s["laws"];names=s["methods"];cheap=d["selected"];timed=s["timed"]
loss=np.array(s["loss"]);bs=np.array(s["seed_effect"]);st=np.array(s["seed_timing"]);loo=np.array(s["leave_one"])
sections=["# v21 · Independent bank seeds와 unseen contact laws에서 external label posterior의 가치", (root/"FINDINGS_KO.md").read_text(),
"""## 연구 질문과 비교의 단위

Factual boundary profile과 collision velocity를 관측한 뒤 query view, 더 정밀한 factual view, 반대 방향의 probe 중 하나를 고른다. 관심 대상은 query 위치에서 물체가 stick할 확률이다. 모든 방법은 선택한 observation을 받아 동일한 full inner bank로 이를 예측한다. Brier loss는 predicted probability와 실제 binary stick label 차이의 제곱이며 낮을수록 좋다. 여기서 ‘실제’는 hidden simulator world에서 실현한 observation과 event를 가리킨다.

V20의 주된 단서는 hypothetical Y를 받았을 때 label probability를 어느 posterior가 제공하느냐에 있었다. Compact posterior C는 full inner bank에서 resample한 256 hypotheses로 구성하고 두 replicas를 쓴다. External posterior E는 별도의 16,384 hypotheses에 같은 factual evidence를 적용한다. (1) Y를 C/E 중 어디서 생성하는지, (2) Y에 대한 label probability p를 C/E 중 어디서 계산하는지, (3) Bernoulli label을 한 번 뽑는지 또는 정확히 평균하는지를 교차한 여덟 cells를 V20과 같은 M512에서 재실행했다. 모든 cells에서 채점하는 예측 q는 compact decoder가 낸다. Conditional mean loss는 p(1−q)²+(1−p)q²다.

**Label-source effect**는 같은 Y law와 label mode에서 E−C로 바꾼 뒤의 *선택 후 held-out Brier loss 차이*를 계산하고, 네 조합에 같은 가중치를 준 값이다. **Deployment contrast**는 고정한 저비용 policy의 loss에서 fixed global mixture(K128/M4096)의 loss를 뺀 값이다. 두 효과는 서로 다른 질문에 답한다.

## Worlds, contact laws와 독립성

개발용 nominal worlds 128개와 held-out worlds 512개를 분리했다. 각 test world에는 공통 actual observation noise를 16회 적용했다. 하나의 world에서 nominal과 두 stress laws를 모두 생성해 geometry·material draws와 noise가 같은 paired comparison을 만들었다. Inference bank와 decoder는 모두 nominal model을 사용했다. 각 bank pair에는 서로 다른 scrambled Sobol inner·outer banks를 만들었고, 다섯 pairs 간에도 다른 seeds를 사용했다. QMC acquisition plans와 representative quantiles는 고정해 bank가 바뀔 때의 변동을 측정했다.

| Data-generating law | Flexible class의 tangential impulse | 이 stress가 바꾸는 것 |
|---|---|---|
| nominal | vₜ sigmoid[b₀+b₁ log((abs(vₜ)+0.02)/0.5)] | Inference와 일치하는 control |
| speed_threshold | sign(vₜ) min(abs(vₜ), μ max(jₙ,0)), μ=0.05+0.4 sigmoid(b₀+1)/(1+abs(vₜ)/(0.15+0.5 sigmoid(b₁))) | 속도가 증가하면 friction coefficient가 감소하고 flexible class에도 exact sticking이 생긴다. |
| directional_damping | vₜ sigmoid[b₀+1.2 tanh(3vₜ)+b₁vₙ/(1+abs(vₙ))] | 진행 방향의 부호와 normal velocity가 감쇠 비율을 바꾼다. |

두 stress laws는 pilot과 model fitting에서 제외했다. Base frictionless/Coulomb classes는 같은 physics를 유지한다. ‘Unseen 평균’은 두 지정 stress에 같은 가중치를 준 값이다. 두 결과가 모든 contact laws를 대표한다고 해석할 수는 없다.""",
f"512 worlds 중 flexible class는 {s['flex_n']}개다. 세 law 순서로 전체 stick rate는 {csv(s['rates'],3)}, flexible class만의 stick rate는 {csv(s['flex_rates'],3)}였다. Base classes에서는 세 조건의 hidden label, action choices, actual losses가 같음을 검증했다.",
"""## External posterior의 비용을 줄인 방법

각 world에서 full external bank의 factual weights를 한 번 계산한다. 그중 weight가 큰 K개 hypotheses를 GPU topk로 취하고, query/factual/probe 각각의 hypothetical observation에 대해 선택된 hypotheses에서 원래 prior weight와 likelihood를 곱해 다시 정규화한다. 얻은 p로 conditional Brier risk를 계산한다. Hypothetical Y는 계속 full external bank에서 생성하므로 label posterior의 support만 줄인다. Factual weight가 작더라도 특정 Y에는 likelihood가 클 수 있어 omitted mass와 action agreement를 따로 측정했다. K=N으로 둔 validation world에서 full implementation과 비교했다.

초기 protocol은 K={128,512,2048}, M={512,2048} 후보 중 fixed global p90 이하이면서 개발 loss가 가장 작은 구성을 선택했다. Pilot timing은 별도 bank pair와 24개 개발 worlds, warmup 후 방법별 3회씩 측정했다. 개발 loss는 128 worlds에서 평가했다. 이 개발 평균으로 최종 방법을 정한 후 test outcomes를 열었다.""",
table(["개발 후보","개발 mean Brier","Pilot GPU p90 ms"],[[n,f"{v:.7f}",f"{t:.2f}"] for n,v,t in zip(d["methods"],d["development_loss"],d["p90"])]),
f"선택된 구성은 {cheap}이다. 전체 bank를 유지했을 때 full risk와의 최대 차이는 {d['max_error']:.3e}였다.",
"""## Label-source replication과 primary endpoints

World와 bank pair를 독립적으로 재추출하는 crossed paired bootstrap을 20,000회 수행했다. 각 draw 안에서 law·method·noise pairing을 유지하며 noise repeats를 독립 worlds로 세지 않았다. 사전 정의한 네 contrasts의 CI는 Bonferroni 98.75%이다. Bank 수가 다섯이므로 seed 모집단에 대한 bootstrap tail 추정은 거칠다. 옆에 bank를 고정한 world-only interval과 seed별 값을 함께 제공한다.""",
table(["Primary contrast","Crossed 98.75% CI","World-only 98.75% CI"],[[n,ci(v),ci(w)] for n,v,w in zip(["Label E−C · nominal","Label E−C · unseen 평균","Cheap−fixed · nominal","Cheap−fixed · unseen 평균"],s["primary"],s["world_only"])]),
table(["Bank inner / outer","Label E−C · nominal / threshold / directional","Cheap−fixed · 같은 순서"],[
[f"{a} / {b}",csv(x[:,0]),csv(x[:,1])] for (a,b),x in zip(s["seedpairs"],bs)]),
table(["Method","Nominal Brier ↓","Speed threshold Brier ↓","Directional damping Brier ↓"],
[[n]+[f"{v:.6f}" for v in loss[:,i]] for i,n in enumerate(names)]),
table(["Secondary contrast","Nominal · 95% CI","Speed threshold · 95% CI","Directional damping · 95% CI"],
[[label]+[ci(x) for x in s["secondary"][i*3:i*3+3]] for i,label in enumerate(["Label E−C","Cheap−fixed","Top-K−full E · E-Y/mean M512","Top-K−compact C · E-Y/mean M512"])]),
f"같은 M512에서 top-K와 full E의 Top-1 agreement는 세 조건에서 {csv(np.array(p['full_control_agreement'])*100,2)}%였다. Action별 risk에서 공통 offset을 제거한 RMS 차이는 {csv(p['full_control_centered_rms'])}였다. Full-vs-compact pair는 같은 Y, q, mean mode에서 label source만 다르다.",
f"선택 K의 omitted factual mass median은 {csv(s['omitted_quantiles'][0],5)}, p90은 {csv(s['omitted_quantiles'][1],5)}, 최대는 {csv(s['omitted_quantiles'][2],5)}였다. Flexible class만 본 E−C는 {csv(s['flex_effect'])}, cheap−fixed는 {csv(s['flex_delta'])}였다. 이 subgroup은 mechanistic diagnostic이다.",
"![Bank sensitivity and deployment contrast](results/v21.png)",
"""## Online compute와 latency 근접 대조군

RTX PRO 6000 Blackwell 한 장에서 batch 1, float64로 각 방법을 섞어 실행했다. 각 bank·law에서 held-out 첫 16 worlds×2회 측정했다. 매번 factual fit, 필요한 bank update, top-K와 risk 계산, observation 선택 및 선택된 actual observation update까지 포함했다. Static bank construction, sensing 및 reset 비용은 제외했다. Timing을 측정한 모든 action은 미리 저장한 policy choices와 일치했다. Unseen p90은 두 law의 측정치를 합쳐 산출했다.""",
table(["조건","방법","GPU p50 ms","GPU p90 ms"],[[law,n,f"{v['p50']:.2f}",f"{v['p90']:.2f}"] for law,vals in s["timing"].items() for n,v in vals.items()]),
"""첫 development 결과의 clock time이 cap에 여유가 있어, held-out losses를 확인하기 전에 별도 latency protocol을 고정했다. 다른 calibration bank에서 M을 4,096까지, K를 4,096까지 넓혀 실제 p90이 fixed global 이하이며 가장 가까운 구성을 택했다. Test에서는 두 방법을 같은 timing pass에서 재측정했다. 사전에 정한 근접 구간은 p90 ratio 0.90–1.00이다. 이 추가 분석은 같은 test worlds와 같은 다섯 banks를 사용한다.""",
table(["추가 calibration 후보","GPU p90 ms"],[[n,f"{v:.2f}"] for n,v in zip(pd["options"],pd["p90"])]),
f"추가 선택은 {pd['selected']}였다. Nominal/두 unseen 순서로 이 방법의 Brier loss는 {csv(p['loss'])}, fixed global은 {csv(p['fixed'])}였다.",
table(["Held-out 조건","후보 p50 / p90 ms","Fixed p50 / p90 ms","p90 ratio","Brier 차이와 CI"],
[[law,f"{v['p50'][0]:.2f} / {v['p90'][0]:.2f}",f"{v['p50'][1]:.2f} / {v['p90'][1]:.2f}",f"{v['p90'][0]/v['p90'][1]:.3f}",ci((p['ci'][0] if law=="nominal" else p['ci'][1]) if law in ('nominal','unseen') else p['per_law'][laws.index(law)])] for law,v in p['timing'].items()]),
"추가 비교의 nominal·unseen 평균은 두 endpoint에 대해 보정한 97.5% CI, 각 stress law는 descriptive 95% CI다. Pilot의 근접 latency가 새 worlds에서 같은지와 Brier 우위가 있는지를 각각 확인해야 한다.",
table(["Bank","추가 후보−fixed Brier · nominal / threshold / directional","주 후보 GPU p90 ms · 같은 순서","주 비교 fixed GPU p90 ms"],
[[i+1,csv(x),csv(st[i,:,0],2),csv(st[i,:,1],2)] for i,x in enumerate(p["bank_delta"])]),
"각 bank를 한 번씩 제외한 E−C nominal 평균의 범위는 "+csv([loo[:,0,0].min(),loo[:,0,0].max()])+"였고, 두 unseen의 동일가중 E−C 평균 범위는 "+csv([loo[:,1:,0].mean(1).min(),loo[:,1:,0].mean(1).max()])+"였다.",
"""## 실행 및 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/external_label_v21`이다. `PROTOCOL.md`와 `LATENCY_PROTOCOL.md`에 두 비교의 설정을 기록했다. 코드 `v21_policy.py`가 posterior truncation을, `data_v21.py`가 private world 생성을, `run_v21.py`와 `parity.py`가 public policy selection 및 평가를 담당한다. `results/summary.json`과 `results/parity/summary.json`에 두 비교의 수치가 있고 `policy_*.npz`, `eval_*.npz`, `paired_losses.npz`에 재분석용 데이터를 보관했다.

재현 시 같은 상대 경로에 이전 버전의 dependencies를 두고 새 directory에서 Slurm allocation 안의 idle GPU로 v18 guard를 통해 run_v21.py를 실행한다. Compute-node CPU에서 analyze_v21.py를 실행한 뒤 같은 GPU의 별도 guard directory에서 parity.py를 실행하고 CPU에서 parity_stats.py, report_v21.py 순서로 진행한다. 실험은 synthetic local impulse model의 단일 observation 선택이며 raw RGB, physical robot trajectory 또는 실측 contact law를 포함하지 않는다.""",
(root/"REVIEW_KO.md").read_text()]
f=load("fine/summary.json");fd=load("fine/design.json")
at=next(i for i,x in enumerate(sections) if x.startswith("## 실행 및 자료"))
sections[at:at]=["## Fine calibration으로 실제 latency 근접 여부 재확인",
"""위의 coarse calibration은 p90 ratio 0.885로 사전 근접 범위에 도달하지 못했다. 어느 held-out loss도 확인하기 전에 마지막 latency-only grid를 기록했다. 또 다른 별도 calibration bank와 32개 nominal development worlds에서 M2048, K2048부터 4352까지 256 간격을 평가하고, cap 이하인 후보 중 0.97에 가장 가까운 비율을 우선했다. 이후에는 test timing에 따라 다시 조정하지 않았다. 이 추가 비교는 같은 held-out 사례를 쓰는 exploratory 분석이다.""",
f"선택된 후보 {fd['selected']}의 pilot ratio는 {fd['p90'][fd['options'].index(fd['selected'])]/fd['p90'][-1]:.3f}였다. Held-out에서는 각 조건의 새 interleaved fixed baseline과 다시 비교했다.",
table(["조건","Brier 후보 / fixed","후보 / fixed p90 ms","p90 ratio","후보−fixed Brier CI"],[
[law,(f"{f['loss'][0]:.6f} / {f['fixed'][0]:.6f}" if law=="nominal" else f"{np.mean(f['loss'][1:]):.6f} / {np.mean(f['fixed'][1:]):.6f}" if law=="unseen" else f"{f['loss'][laws.index(law)]:.6f} / {f['fixed'][laws.index(law)]:.6f}"),csv(v["p90"],2),f"{v['p90'][0]/v['p90'][1]:.3f}",ci(f["ci"][0 if law=="nominal" else 1] if law in ["nominal","unseen"] else f["per_law"][laws.index(law)])] for law,v in f["timing"].items()]),
"Nominal 및 unseen 평균의 interval은 이 추가 비교 두 개에 대한 97.5% CI이며, coarse/fine과 주 실험 전체를 묶은 simultaneous CI는 아니다. Calibration 및 test timing은 별도 측정으로 GPU clock, scheduling, world 구성에 따른 변동을 포함한다. p90 budget과 근접 구간은 이번 연구를 위해 정한 engineering 기준이다. Fine 결과는 results/fine의 design.json과 summary.json에 저장했다."]
fig,ax=plt.subplots(1,2,figsize=(10.3,4.3),layout="constrained")
xx=np.arange(3)
for j in range(5):
    ax[0].plot(xx,bs[j,:,0],marker=".",alpha=.55,label=f"Bank {j+1}")
ax[0].plot(xx,bs[:,:,0].mean(0),color="black",marker="o",label="Mean",lw=2.4)
ax[0].axhline(0,color=".5",ls=":");ax[0].set(xticks=xx,xticklabels=["Nominal","Threshold","Directional"],ylabel="Brier difference (external minus compact)",title="Label source across bank pairs");ax[0].legend(ncol=2,fontsize=7)
for label,i,c in [("Selected under cap",0,"#246292"),("Latency matched",1,"#bd602d")]:
    ds=s["secondary"][3:6] if i==0 else f["per_law"]
    a=np.array(ds);ax[1].errorbar(xx+(i-.5)*.09,a[:,0],yerr=[a[:,0]-a[:,1],a[:,2]-a[:,0]],label=label,fmt="o",capsize=3,color=c)
ax[1].axhline(0,color=".5",ls=":");ax[1].set(xticks=xx,xticklabels=["Nominal","Threshold","Directional"],title="Deployment loss (descriptive 95% CI)",ylabel="Brier difference vs fixed global");ax[1].legend(fontsize=8)
fig.savefig(r/"v21.png",dpi=160)
(root/"REPORT_KO.md").write_text("\n\n".join(sections)+"\n")
