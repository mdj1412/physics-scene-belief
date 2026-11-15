from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def load(n):return json.loads((root/"results"/(n+".json")).read_text())
s=load("summary");d=load("design");sub=load("subgroups")
laws=["nominal","speed_threshold","directional_damping"]
names=s["names"];top,keep=names[3:5];labels=["Query view","Factual view","Probe"]
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])
def val(x,spec=".6f"):return "—" if x is None else format(x,spec)
def pct(x):return val(None if x is None else x*100,".2f")
def ci(x):
    if x is None:return "—"
    return f"{x[0]:+.6f} [{x[1]:+.6f}, {x[2]:+.6f}]"
def ratio(time,law,a,b):
    return time[law][a][1]/time[law][b][1]
def tci(a,b):return [ci(s["secondary"][j]) for j in range(a,b)]
part=["# v22 · Top-K가 제외한 scene hypotheses의 posterior mass 회복과 observation 선택",
(root/"FINDINGS_KO.md").read_text(),
"""## 무엇을 ‘회복’으로 측정했는가

Factual boundary profile과 충돌 직후 속도를 본 뒤 생긴 16,384개 external hypotheses의 가중치를 w라 하자. 높은 w의 K개를 H, 나머지를 T로 고정했다. 추가 observation Y를 본 뒤에도 집합 H와 T의 구성은 그대로 두고, full external bank의 posterior weights를 likelihood로 다시 계산했다. 각 action에서 factual tail mass와 updated tail mass를 대응시켰다.

**Collective recovery**는 T의 factual mass가 5% 이하이면서 Y 이후 10% 이상이고 동시에 5배 이상 증가한 경우다. **Individual recovery**는 T에 속했던 개별 hypothesis가 Y 이후 mass 1% 이상이고 자신의 factual weight 대비 10배 이상이 된 경우다. 두 정의는 다른 현상을 센다. 여러 작은 hypotheses가 함께 회복되거나, tail 전체의 mass는 작아도 그 안의 한 hypothesis가 두드러질 수 있다. 사건의 빈도는 action별 5 bank pairs × 512 worlds × 16 noise repeats를 분모로 쓴 descriptive rate이며, 이 40,960개가 독립 물체라는 가정은 하지 않았다.

Stick probability의 변화에는 간단한 항등식이 있다. T의 updated mass를 m, updated head와 tail 내부의 stick probabilities를 각각 pH, pT라 하면 **p(full)−p(top) = m × (pT−pH)**이다. 모든 진단 updates에서 이 등식을 float64 오차 10⁻⁹ 이내로 검산했다. 큰 mass가 돌아와도 두 집단이 같은 label을 예측하면 top-K의 label probability는 거의 바뀌지 않는다. 반대로 m과 label 분리가 함께 크면 작은 factual weight가 중요해진다. 모든 mass와 label probability는 지정된 finite nominal bank의 값이다.

## 보존 방식과 실험 설계

총 K slots 중 상위 3K/4개는 원래 weight로 보존한다. 나머지 hypotheses를 각 candidate scene에서 계산한 query stick/non-stick label로 나눈다. 각 그룹에서 K/8개를 factual weights에 비례하는 inverse-CDF stratified resampling으로 뽑고, 각 slot에 그 그룹의 factual mass를 K/8로 나눈 값을 부여한다. 복원 추출이므로 일부 slots는 같은 hypothesis를 나타낸다. 전체 factual mass와 두 label masses를 보존하며, 그룹 내부의 observation likelihood 분포는 표본으로 근사한다. Pilot에서 전체 mass 및 label별 mass를 10⁻¹⁰ 이내로 검산했다. Tail sample plan은 고정했으며 실제 Y나 숨겨진 world label은 selector에 제공하지 않았다.

평가에는 새 worlds 512개, 다른 seed의 independent inner/outer bank pairs 다섯 쌍, actual noise repeats 16개를 사용했다. Nominal, speed threshold, directional damping은 V21과 같은 세 law다. 모든 조건에서 nominal model로 inference했다. Geometry와 noise를 같은 world에서 공유하고 flexible contact class에만 stress를 적용했다. 변경하지 않은 base classes에서 선택과 실측 loss가 같은지 확인했다. 개발용 nominal worlds 32개와 별도 calibration bank는 test에서 사용하지 않았다.

**Mechanism comparison**은 K512/M512에서 top, 보존 방법, full external label evaluator를 비교했다. 세 방법 모두 동일한 hypothetical observations, compact q, conditional Brier score를 쓰고 label posterior support만 바꾼다. Full evaluator도 M512와 finite bank를 쓰는 numerical reference다. 선택 후에는 모든 방법이 같은 full inner decoder로 해당 actual observation을 받아 stick probability를 예측한다. 따라서 아래의 decision loss는 approximation이 action을 바꾸면서 생긴 실현 Brier 차이다. Truncated decoder를 실제 배포한 loss와는 다르다.

**Deployment comparison**은 top과 보존 방법의 M을 2,048로 고정하고 K를 각각 latency로 선택했다. Fixed global은 원래 K128/M4096 direct kernel이다. Pilot 16 worlds에서 넓은 K grid를 측정하고, 다음 16 worlds에서 가까운 후보를 다시 측정해 세 방법의 p90이 서로 가깝도록 선택했다. 이때 Brier loss를 사용하지 않았다. Test에서는 bank·law별 첫 24 worlds × 2회, 방법 순서를 섞어 GPU를 동기화했다. Factual fit, support 구성, acquisition, 선택된 observation update를 포함한 batch-1 float64 latency다. Plan·bank의 static setup과 실제 sensing/reset 시간은 제외했다. 허용한 근접 범위는 각 비교 쌍의 p90 ratio 0.90–1.10이며, 실제 test 비율을 아래에 제시한다.""",
f"개발에서 확정한 top은 {top}, 보존 방법은 {keep}다. 두 번째 pilot의 p90은 각각 {d['second'][top]:.3f}, {d['second'][keep]:.3f}, fixed global {d['second']['fixed']:.3f} ms였다.",
"## 어느 observation에서 mass가 돌아왔는가",
"""아래 집계는 모든 action을 실제 simulator에서 짝지어 측정한 counterfactual evaluation이다. Surprise는 factual predictive distribution 아래에서 actual Y의 negative log marginal density를 whitened coordinates로 계산했다. 차원이 다른 action끼리 절대값을 비교하지 않고 각 law·action 안의 하위/상위 quartile을 비교했다. ‘회복 시 label error’는 해당 조건에서 full external probability와의 mean absolute difference다. 두 방법의 실현 Brier와는 별개의 진단이다."""]
for g,lab in [(0,"진단용 top K512"),(1,"latency에 맞춘 top K")]:
    rows=[]
    for law in laws:
        for i,a in enumerate(labels):
            e=s["when"][f"{law}/{g}"][str(i+1)];h=e["high"]
            rows.append([law+" · "+a,h["n"],pct(e["rate"]),pct(e["atom"]),pct(e["low_surprise_rate"])+" / "+pct(e["high_surprise_rate"]),val(h["post"],".3f"),val(h["p_top_err"],".4f")+" / "+val(h["p_keep_err"],".4f"),pct(h["retained_fraction"])])
    part.extend(["### "+lab,table(["Law · action","회복 건수 / 40960","Collective %","Individual %","Surprise Q1 / Q4 회복 %","회복 시 mean mass","회복 시 top / 보존 label error","보존 support에 포함된 회복 mass %"],rows)])
part.append("마지막 열은 full posterior의 실제 tail mass 중 보존 방법이 한 번 이상 뽑은 hypotheses에 속한 비율이다. Resampled weight의 합과는 정의가 다르다. 표의 — 는 해당 event가 없어 조건부 평균을 계산할 수 없다는 뜻이다.")
part.append(table(["Law · K512","Base / flexible 회복 % (query)","Base / flexible 회복 % (factual)","Base / flexible 회복 % (probe)","실제 top 선택의 회복 %","실제 보존 선택의 회복 %"],
[[law]+[pct(s["when"][f"{law}/0"][str(j)]["base_rate"])+" / "+pct(s["when"][f"{law}/0"][str(j)]["flex_rate"]) for j in (1,2,3)]+
 [pct(s["mechanism"][law][k]["rate"]) for k in ("top_chosen","keep_chosen")] for law in laws]))
part.append(table(["Law","K512 sensitivity (≥5%,2× / ≥10%,5× / ≥20%,10×) %","K512 회복 중 post label error≥0.05 % (query / factual / probe)"],
[[law," / ".join(pct(x) for x in [sub[law]["0"]["rates_alt"][0],sub[law]["0"]["pooled_rate"],sub[law]["0"]["rates_alt"][1]])," / ".join(pct(s["when"][law+"/0"][str(a)]["high"]["abs_change_ge_05"]) for a in (1,2,3))] for law in laws]))
part.append("### 큰 회복의 실제 예시")
part.append(table(["조건 · bank/world/repeat (0-based)","Action","Tail factual → updated","제외된 최대 atom의 factual rank","그 atom의 post mass / 증가배수","Full / top / 보존 stick probability","선택 (top/보존/full)"],
[[f"{v['law']} · {v['bank']}/{v['world']}/{v['repeat']}",labels[v['action']-1],f"{v['metrics'][0]:.4f} → {v['metrics'][1]:.4f}",int(v['metrics'][4]),f"{v['metrics'][3]:.4f} / {v['metrics'][5]:.0f}×"," / ".join(f"{v['metrics'][i]:.4f}" for i in (6,7,8))," / ".join(labels[j-1] for j in v['choices'])] for v in s["examples"]]))
part.append("각 law에서 K512 material recovery 중 updated mass가 가장 큰 사례를 사후에 골랐다. 빈도와 평균은 위의 전체 표로 판단해야 한다.")
part.extend(["## 회복이 실제 선택과 loss에 미친 규모",
"""Full-vs-top 비교에서는 같은 hypothetical Y에서 score를 계산했으므로, 이 numerical reference 대비 flip은 해당 support substitution에 귀속할 수 있다. 실제 Y는 action을 고른 뒤에 얻어진다. 따라서 아래의 회복 subgroup과 flip의 연관은 선택 전에 그 Y를 볼 수 있다는 의미가 아니다.

각 bank-world에서 첫 8개 repeats의 세 potential actions 중 하나라도 material recovery가 있으면 회복 집단으로 분류했다. 나머지 8개 repeats로만 집단별 actual Brier를 평가했다. 이 분리는 동일 noise draw로 집단을 정의하고 loss를 측정하는 문제를 줄인다. 동일 world를 공유하므로 개별 observation에서의 인과적 매개효과를 추정하는 설계는 아니다. 집단별 interval은 bank와 world를 재추출한 conditional mean의 descriptive 95% bootstrap이다."""])
rows=[]
for law in laws:
    for key in ("recovery","other"):
        z=s["strata"][law+"/0"][key];extra=sub[law]["0"]
        rows.append([law+" · "+key,z["n"],pct(z["top_full_flip"]),pct(z["keep_full_flip"]),pct(z["top_keep_flip"]),val(z["top_minus_full"]),ci(extra["recovery_ci" if key=="recovery" else "other_ci"])])
part.append(table(["조건 · subgroup","bank-world 수","Top/full flip %","보존/full flip %","Top/보존 flip %","Top−full Brier","보존−top Brier [95% CI]"],rows))
part.append(table(["Law · 전체 worlds, K512","Top/full flip %","보존/full flip %","Top → 보존 risk error RMS","Top−full Brier [95% CI]","보존−full Brier [95% CI]","보존−top Brier [95% CI]"],
[[law,pct(s['mechanism'][law]['top_full_flip']),pct(s['mechanism'][law]['keep_full_flip']),f"{s['mechanism'][law]['top_risk_rmse']:.6f} → {s['mechanism'][law]['keep_risk_rmse']:.6f}"]+tci(3*l,3*l+3) for l,law in enumerate(laws)]))
part.append("Risk error는 action-common offset을 뺀 RMS다. Full과 선택이 동일하면 공통 actual decoder가 받는 observation도 같아 두 방법의 world loss 차이는 0이다. Flip 횟수가 많아도 거의 동률인 action 사이에서 주로 일어나면 평균 Brier는 크게 변하지 않을 수 있다.")
part.append(table(["Law","Top/full flip에서 full action gap median","일치에서 gap median","다른 action 선택 시 top−full Brier · 마지막 8회 [95% CI]","다른 action 선택 시 보존−top Brier · 마지막 8회 [95% CI]"],
[[l,val(sub[l]["0"]["median_full_gap_flip0"]),val(sub[l]["0"]["median_full_gap_same0"]),ci(sub[l]["0"]["switched_actual_delta0"]),ci(sub[l]["0"]["disagreement_ci"])] for l in laws]))
part.append("## Held-out Brier loss와 GPU 비용")
part.append(table(["Policy","Nominal","Speed threshold","Directional damping"],[[n]+[f"{s['loss'][l][j]:.6f}" for l in range(3)] for j,n in enumerate(names)]))
part.append(table(["Primary paired contrast","Crossed 98.75% CI","World-only 98.75% CI"],[[l,ci(v),ci(w)] for l,v,w in zip(["보존−top · nominal","보존−fixed · nominal","보존−top · unseen 평균","보존−fixed · unseen 평균"],s['primary'],s['world_ci'])]))
part.append("Primary family의 네 contrasts에 Bonferroni를 적용했다. 20,000 bootstrap draws에서 bank pair와 world를 독립적으로 resample하고 law·method·sensor noise의 pairing을 유지했다. Unseen은 V21에서 정의한 두 stress laws의 동일가중 평균이다. Stress laws 자체를 모집단에서 resample한 interval은 아니다.")
part.append(table(["조건","Top p50 / p90 ms","보존 p50 / p90 ms","Fixed p50 / p90 ms","Probe p50 / p90 ms","보존/top p90","보존/fixed p90","top/fixed p90"],
[[l]+[" / ".join(f"{x:.3f}" for x in v[n]) for n in names[3:]]+[f"{ratio(s['timing'],l,a,b):.3f}" for a,b in [(keep,top),(keep,'fixed'),(top,'fixed')]] for l,v in s["timing"].items()]))
part.append(table(["Bank","보존−top · nominal / threshold / directional","보존−fixed · 같은 순서"],[[i+1," / ".join(f"{v[j][0]:+.6f}" for j in range(3))," / ".join(f"{v[j][1]:+.6f}" for j in range(3))] for i,v in enumerate(s['seed_deltas'])]))
part.append("![Recovery by action and held-out loss](results/v22.png)")
part.extend(["## 검증 자료와 범위",
"""원래 simulator는 local impulse, 하나의 추가 observation, 동일 sensing cost를 사용한다. 복원한 mass는 full external posterior가 부여한 값이며 실제 hidden physics가 정답이라는 뜻은 아니다. Synthetic stress에서 nominal posterior가 크게 확신하는 잘못된 hypotheses가 나타날 수도 있다. Real imagery, 개입 비용, trajectory reward로 일반화하려면 별도 검증이 필요하다.

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tail_recovery_v22`이다. `PROTOCOL.md`, `policy22.py`, `run22.py`, `stats22.py`에 설계·구현·primary analysis를 기록했다. `subgroup22.py`의 threshold sensitivity와 conditional CIs는 추가 진단이다. `choices_*.npz`는 public-only 선택과 risks, `evaluation_*.npz`는 paired losses와 실제 intervention별 mass, `columns.json`은 그 열의 정의다. 각 단계의 hash는 results 아래 plan.json, choices_freeze.json, sources.json에 있다. 재현할 때는 이전 v18–v21 dependencies의 상대 경로를 유지하고 새로운 실험 폴더에서 v18 guard를 통해 run22.py를 실행한 뒤 compute-node CPU에서 stats22.py, subgroup22.py, example22.py를 실행한다. 추가 대조는 같은 GPU에서 control22.py를 실행한 뒤 CPU에서 control_stats.py를 실행하고, 마지막에 report22.py로 표와 그림을 생성한다.""",
(root/"REVIEW_KO.md").read_text()])
extra=load("examples_checked")
rows=[]
for v in extra:
    rows.append([v["law"],v["rank"],f"{100*v['atom_before']:.5f}% → {100*v['atom_after']:.2f}%",
                 f"{v['whitened_rms_winner']:.2f} / {v['whitened_rms_factual_map']:.2f}",
                 f"{v['mean_likelihood_ratio']:,.0f}×"])
pos=part.index("## 회복이 실제 선택과 loss에 미친 규모")
part[pos:pos]=[table(["Law","회복 atom의 rank","Atom factual → updated","실제 Y에 대한 whitened RMS residual · 회복 atom / factual MAP","Tail 대 head의 평균 likelihood ratio"],rows),
"Physical coordinates에서 세 사례의 likelihood를 독립적으로 재계산했다. Nominal query view에서 factual rank 673의 non-stick hypothesis는 actual Y와 0.50 noise standard deviations의 RMS 차이를 보였고, factual MAP의 차이는 5.39였다. 이때 제외 집합 전체가 2.38%에서 99.49%로 커졌다. 그러나 이 world에서 사전에 고른 action은 세 selector 모두 probe였다. 따라서 이 query view의 큰 posterior 변화는 세 정책 사이의 실현 loss 차이를 만들지 않았다."]
ab=load("unstratified/summary")
actions=np.stack([np.load(root/f"results/choices_{b}.npz")["acts"] for b in range(5)])
risks=np.stack([np.load(root/f"results/choices_{b}.npz")["risk"] for b in range(5)])
assert np.array_equal(actions[...,3],actions[...,4])
rows=[]
for l,law in enumerate(laws):
    vals=risks[:,:,l,:3,:]
    tie=np.ptp(vals,axis=-1)<=1e-12
    alts=np.where(tie,3,vals.argmin(-1)+1)
    flip=actions[:,:,l,0]!=actions[:,:,l,2]
    rows.append([law,int(flip.sum()),int((flip&tie[...,0]).sum()),pct((alts[:,:,0]!=alts[:,:,2]).mean()),pct((alts[:,:,1]!=alts[:,:,2]).mean()),
                 int((actions[:,:,l,4]!=actions[:,:,l,5]).sum())])
pos=part.index("## Held-out Brier loss와 GPU 비용")
part[pos:pos]=["### Tie 처리와 tail sampling 대조 실험",
"사전 규칙은 세 risk의 최대−최소가 10⁻¹² 이하면 query view를 고른다. 아래 sensitivity는 저장한 세 risk를 그대로 두고 해당 tie에서 probe를 고르도록 바꾼 사후 분석이다. Full-bank reference에도 같은 규칙을 적용했다. 마지막 열은 원래 배포 규칙에서 보존과 fixed global이 서로 다른 action을 선택한 횟수다. Law당 분모는 2,560 bank–world 조합이다.",
table(["Law","K512 top/full 불일치 건수","그중 top이 three-way tie인 건수","Tie→probe일 때 top/full 불일치 %","Tie→probe일 때 보존/full 불일치 %","배포 보존/fixed 불일치 건수"],rows),
"또한 같은 GPU에서 K512/M512, 동일한 exact head와 tail slots, 같은 resampling jitter로 tail을 label별 분할 없이 하나의 weighted distribution에서 뽑았다. 동일한 세 potential actions의 held-out loss로 평가했다. 이 exploratory ablation은 primary experiment가 끝난 뒤 추가했다. 이 대조의 조건은 동일한 K와 M이며 별도 latency calibration은 수행하지 않았다.",
table(["Law","Pooled-tail Brier","Top−pooled Brier [95% CI]","Label별 보존−pooled Brier [95% CI]","Pooled와 label별 보존의 선택 불일치 %"],
[[l,val(ab["loss"][j]),ci(ab["comparisons"][2*j]),ci(ab["comparisons"][2*j+1]),pct(ab["flips"][j][1])] for j,l in enumerate(laws)]),
"Pooled-tail의 Brier 점추정이 세 law에서 모두 label별 보존보다 낮았고, 세 paired intervals 모두 0을 포함했다. 두 방식은 선택의 약 11–15%에서 서로 달랐다. Label별 보존이 Brier를 줄이는 고유한 원인이라고 판정할 수 없다."]
fig,ax=plt.subplots(1,2,figsize=(10.6,4.5),layout="constrained")
xx=np.arange(3)
for i,law in enumerate(laws):
    ax[0].bar(xx+(i-1)*.23,[100*s['when'][law+'/0'][str(j)]['rate'] for j in (1,2,3)],width=.22,label=law)
ax[0].set(xticks=xx,xticklabels=labels,ylabel="Material recovery (% of actual draws)",title="Factual top 512");ax[0].legend(fontsize=8)
for j,lab in [(0,'vs top-K'),(1,'vs fixed global')]:
    v=np.array([s['primary'][j],s['primary'][j+2]])
    ax[1].errorbar(np.array([0,1])+(j-.5)*.08,v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt="o",capsize=3,label=lab)
ax[1].axhline(0,color=".4",ls=":");ax[1].set(xticks=[0,1],xticklabels=["Nominal","Unseen mean"],ylabel="Preserving minus baseline Brier",title="Latency-matched comparison (98.75% CI)");ax[1].legend(fontsize=8)
fig.savefig(root/"results/v22.png",dpi=160)
(root/"REPORT_KO.md").write_text("\n\n".join(part)+"\n")
