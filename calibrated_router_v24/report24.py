"""Render report directly from verified artifacts; all scientific quantities from the server run."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
r=root/"results"
pilot=root.parent/"cheap_tail_v24"/"results"
def get(d,n):return json.loads((d/(n+".json")).read_text())
def tab(head,rows):
    return "\n".join(["| "+" | ".join(head)+" |","|"+"|".join(["---"]*len(head))+"|"]+
                     ["| "+" | ".join(map(str,rr))+" |" for rr in rows])
def num(v):return f"{v:.6f}"
def signed(v):return f"{v:+.6f}"
def rate(v):return f"{100*v:.1f}%"
def fmtrow(x):return " / ".join(num(v) for v in x)
def yn(b):return "충족" if b else "미충족"
def counts(n,mask):return f"{n} / {mask}"
def rng(x,k=3):return f"{min(x):.{k}f}–{max(x):.{k}f}"
def triple(v):return f"{v[0]:+.6f} [{v[1]:+.6f}, {v[2]:+.6f}]"
def rho(x,y):
    if len(np.unique(x))==1 or len(np.unique(y))==1:return "정의되지 않음"
    return f"{spearmanr(x,y).statistic:.3f}"
def joint_static(s,j):
    return all(row["ub"][j]<.003 and row["ratio_ub"][j]<.95 for row in s["rows"].values())
local=get(r,"localization");s=get(r,"summary");cfg=get(r,"selected");pilot_s=get(pilot,"summary");pilot_cfg=get(pilot,"selected")
rows=s['rows'];cal=list(rows)
methods=["Factual gate","Always-preserve K2048",f"Always-low K{cfg['low']}",f"Always-high K{cfg['high']}",f"Fixed budget K{cfg['static']}","Fixed global mixture","Blind routing"]
diag=[np.load(r/f"routing_{i}.npz") for i in range(5)]
over=np.stack([d["times"] for d in diag]);sig=np.stack([d["signals"] for d in diag])
us=np.quantile(over.reshape(-1,2),[.5,.9],axis=0)
all_b=all(v['brier_pass'] for v in rows.values());all_t=all(v['p90_pass'] for v in rows.values())
statpass=[methods[j] for j in [2,3,4] if joint_static(s,j)]
parts=["# v24 · 독립적인 curvature와 friction에서 factual gate의 noninferiority와 GPU p90",
f"**동시 판정은 {yn(s['joint'])}이다.** Camera-favouring, probe-favouring, balanced sensor calibration 세 조건에서 absolute Brier 증가의 one-sided 95% upper bound를 +0.003과 비교하고, 같은 입력의 GPU p90 ratio upper bound를 0.95와 비교했다. Brier 기준은 {sum(v['brier_pass'] for v in rows.values())}/3개, p90 기준은 {sum(v['p90_pass'] for v in rows.values())}/3개 calibration에서 통과했다. 대상은 새로운 synthetic worlds 1,152개, 별도의 bank pairs 5쌍과 한 장의 RTX PRO 6000이다.",
tab(["Held-out calibration","Gate Brier","Always-preserve Brier","Gate−reference · 95% upper","Gate / reference p90 (ms)","p90 ratio · 95% upper","동시 판정"],
[[c,num(v['loss'][0]),num(v['loss'][1]),signed(v['delta'][0])+" · "+signed(v['ub'][0]),f"{v['p90'][0]:.3f} / {v['p90'][1]:.3f}",f"{v['ratio'][0]:.3f} · {v['ratio_ub'][0]:.3f}",yn(v['brier_pass'] and v['p90_pass'])] for c,v in rows.items()]),
"## 1. V23에서 어떤 가정을 바꿨는가",
"""V23은 curvature로 stick을 판단해야 하는 family와 friction을 봐야 하는 family의 factual slope 분포가 달랐다. V24는 그 family를 제거했다. Quadratic boundary의 curvature 중심 −0.29, 0, +0.29와 Coulomb friction coefficient 중심 0.12, 0.32를 모두 교차했다. 각 셀에서 curvature에는 독립 Uniform(−0.06,+0.06), friction에는 독립 Uniform(−0.0375,+0.0375) jitter를 주었다. Offset, local slope, restitution도 별도로 추출했다. 셀마다 192개 world를 같은 가중치로 평가했다. 새로운 worlds는 v23·v24 development에서 관찰한 loss로 선별하지 않았다.

Factual evidence는 중심부 boundary profile의 관측값 9개와 incoming velocity (1,−0.10)인 충돌 직후의 velocity 2개다. 선택지는 query view, factual view, incoming velocity (1,−0.85)의 probe다. Target은 y=0.025, incoming velocity (1,0.425)에서의 stick이다. 공통 16,384-hypothesis inner decoder에 선택한 observation을 한 번 넣고, 실제 stick label에 대한 (p−label)²를 평가했다. Simulator와 inference는 같은 Coulomb contact law를 사용했다. Calibration 간에는 같은 world, label 및 표준화한 observation noise를 공유한다.""",
f"실제 held-out continuous curvature와 friction의 Pearson correlation은 {s['correlation'][1][2]:+.4f}였다. 독립성은 이 sample correlation의 크기에서 추론한 것이 아니라 generator의 factorial assignment와 독립 jitter에서 온다.",
tab(["Calibration","Factual profile σ","Factual velocity σ","Query view σ","Factual view σ","Probe σ"],[
["Development · camera","0.0021","0.12","0.00040","0.0010","0.060"],["Development · probe","0.0033","0.18","0.00130","0.00075","0.023"],
["Held-out · camera","0.0018","0.15","0.00030","0.0012","0.075"],["Held-out · probe","0.0037","0.11","0.00165","0.00055","0.018"],["Held-out · balanced","0.0027","0.21","0.00076","0.00090","0.042"]]),
"모든 policy는 해당 센서의 올바른 noise model을 받는다. 여기서 unseen calibration은 두 development 설정과 다른 noise 표준편차 조합을 뜻한다. 잘못 보정된 센서나 새로운 contact law를 시험한 실험은 별도로 필요하다.",
"## 2. 저비용 gate를 어떻게 만들었는가",
"""**첫 구현(A).** Factual feature에서 작은 K와 큰 K의 realized Brier 차이를 예측하는 ridge를 학습했다. K768–2048에서 training labels가 모든 budget pair에서 0이었다. Validation 두 calibration에서도 모든 입력이 high로 갔다. 이 결과만 확인한 시점에 A를 고정 K 대조로 남겼다. A의 held-out loss를 열기 전에 B의 training target·budget 후보·seeds와 판단 기준을 별도 protocol로 고정했다. A와 B는 test worlds·bank pairs를 공유하지 않는다.""",
f"**독립 확인 실험(B).** Ridge는 full-reference K2048과 low K에서 세 action의 acquisition risk 차이를 계산한 뒤, action 공통 offset을 제외한 RMS error의 log를 학습한다. Target은 log(error+10⁻⁷)이다. 실제 stick label 없이도 candidate posterior로 만들 수 있다. Budget 선택에는 별도의 validation Brier를 사용했다. Training 288개, validation 240개, test 1,152개의 worlds가 독립적이다.",
"""Online에서 gate는 아직 GPU posterior를 만들기 전에 host에 있는 factual profile에서 선형 slope와 quadratic curvature를 추정한다. 두 factual velocity, log(query-view σ / probe σ), factual-profile σ까지 6개 features를 만들고, 표준화한 값·제곱·calibration interactions를 합한 20개 항에 ridge coefficients를 적용한다. Gate는 hidden physics, 셀 번호, query label, 추가 observation을 받지 않는다. 두 분기 모두 kernel·M·decoder는 같고 K만 다르다. K의 3/4은 큰 factual weights를 그대로 유지하고, 1/4은 stick/non-stick label별 residual mass에 배정한다. Reference는 이를 매번 K2048/M2048로 실행한다.""",
f"K 후보 쌍은 (64, 768), (128, 1024), (256, 1536)이었다. Development의 40·60·80 percentile cutoffs 중 양쪽 validation calibrations에서 high 빈도 10–90%를 우선하고 worst validation Brier 증가가 +0.0015 이하인 후보에서 평균 K를 최소화했다. 선택된 low/high는 K{cfg['low']} / K{cfg['high']}, 각각 tail slots {cfg['low']//4} / {cfg['high']//4}개다. Cutoff는 training score의 {cfg['quantile']*100:.0f} percentile({cfg['model']['cut']:.4f})이고 validation high 빈도는 {rate(cfg['val_rate'][0])}, {rate(cfg['val_rate'][1])}였다. Threshold나 budget을 test 결과로 다시 맞추지 않았다.",
("양쪽 validation calibration 모두에서 high route 비율 10–90%인 후보가 없었다. 사전 fallback 기준으로 위 구성을 선택했다. 이는 두 개발 sensor에서 적응성이 고르게 학습됐다는 해석을 제한한다." if not cfg["valid"] else "양쪽 validation calibration에서 사전 분기 빈도 조건을 만족했다."),
"## 3. Noninferiority와 GPU p90의 동시 판정",
"""각 calibration에서 Δ = Brier(gate)−Brier(always-preserve), R = p90(gate)/p90(always-preserve)로 정의했다. Δ의 one-sided 95% upper bound < +0.003이고 동시에 R의 upper bound < 0.95여야 통과한다. 세 calibrations가 전부 통과해야 전체 성공이다. 어느 한 endpoint라도 실패하면 성공을 주장할 수 있는 intersection–union test가 아니므로, 전체 성공에는 여섯 조건을 모두 요구했다. +0.003은 v23의 runner-up 비용 약 0.150의 2%로 정한 연구용 tolerance이며 실제 사용자가 허용할 비용은 측정하지 않았다. GPU 기준은 측정 오차를 넘는 5% 이상의 실질적 p90 절감이다.

같은 world에서 32번 반복된 noise로 action별 Brier를 평균한 후 비교했다. Availability of additional observations는 simulation evaluation에만 허용했다. 모든 test choices를 먼저 hash로 고정한 뒤 private labels와 observation을 열었다. Brier uncertainty는 6개 cells 안에서 worlds를 resample하고 5개 banks를 교차 resample한 paired bootstrap 10,000회로 계산했다. Timing은 cell마다 12 worlds, 4 repeats, 5 banks로 방법·calibration당 1,440회다. 동기화한 batch-1, float64, 무작위 interleaved 측정에서 raw factual host input의 전송, gate, support 생성, acquisition, 선택한 observation의 전송·update를 포함한다. Bank 생성·학습, physical sensor 취득 시간은 제외했다. 동일한 bank, world, repeat block을 paired resample한 5,000개 p90 ratios의 95 percentile을 upper bound로 사용했다. 한 서버·한 GPU에 대한 조건부 결과다.""".replace("어느 한 endpoint라도 실패하면 성공을 주장할 수 있는 intersection–union test가 아니므로, 전체 성공에는 여섯 조건을 모두 요구했다.","이 intersection–union test에서는 여섯 조건의 결합만 primary claim으로 검정한다."),
tab(["민감도 · 통과한 calibration 수 / 3","Δ < +0.001","Δ < +0.002","Δ < +0.003","Δ < +0.005","R < 1.00","R < 0.95"],[["95% upper 기준"]+[sum(v['ub'][0]<u for v in rows.values()) for u in (.001,.002,.003,.005)]+[sum(v['ratio_ub'][0]<u for v in rows.values()) for u in (1,.95)]]),
"## 4. Routing이 어디에서 도움이 되었는가",
tab(["Calibration","High 사용률","평균 K","다른 action · bank-world %","다른 action이 있었던 고유 worlds / 1152","다른 선택에서 평균 Δ Brier","High / low route에서 low−high Brier"],[
[c,rate(v['rate']),f"{v['mean_k']:.1f}",rate(v['flip']),v['world_flips'],signed(v['flip_delta']) if v['flip_delta'] is not None else '해당 없음',
" / ".join('해당 없음' if a is None else signed(a) for a in (v['low_minus_high_on'],v['low_minus_high_off']))] for c,v in rows.items()]),
"Online decoder가 같아 action이 같으면 두 정책의 실현 Brier도 같다. 따라서 전체 Δ는 다른 action을 골랐을 비율과 그때의 평균 Δ의 곱이다. High/low 집단의 단순 loss 차이는 gate-on 효과에 대한 인과 추정치가 아니다. 비교에는 같은 world에서 강제로 분기를 고정한 두 결과를 사용했다.",
tab(["비교 방법","Camera Brier / p90 ms","Probe Brier / p90 ms","Balanced Brier / p90 ms"],[[name]+[num(v['loss'][j])+" / "+(f"{v['p90'][j]:.3f}" if j<5 else "미측정") for v in rows.values()] for j,name in enumerate(methods)]),
tab(["Gate−control · paired 95% exploratory CI","Camera","Probe","Balanced"],[[methods[j]]+[triple(v['gate_comparisons'][j]) for v in rows.values()] for j in (2,3,4,6)]),
f"같은 primary 기준을 사후에 정적 controls에도 적용했을 때 세 calibration을 모두 통과한 구성은 {', '.join(statpass) if statpass else '없음'}이었다. Fixed budget은 validation 평균 K에 가장 가까운 사전 후보를 골랐다. Blind routing은 calibration별 동일 high 빈도로 무작위 배정했을 때의 기대 손실이다. 별도 kernel로 실행하지 않았다. Fixed global은 추가 accuracy 대조이며 이번 primary latency comparator는 always-preserve다.",
tab(["Calibration","Calibration 다수 분기","같은 분기인 worlds / 1152","다수 분기 Brier","Gate−다수 분기 · 95% CI"],[[c,z['majority'],z['majority_worlds'],num(z['loss']),triple([z['delta']]+z['ci'])] for c,z in local.items()]),
"이 추가 대조는 각 calibration의 test factual inputs에서 더 자주 쓰인 분기를 그 calibration 전체에 적용했다. Labels를 보지 않는 transductive diagnostic이다. 실제로 calibration metadata만으로도 비슷한 결과가 나오는지 점검하며, 사전 확정된 deployable primary baseline과는 구분한다.",
tab(["Calibration","Gate action 횟수 Q / F / P","Reference action 횟수 Q / F / P","항상 query view / factual view / probe Brier"],[[c," / ".join(map(str,z['gate_actions']))," / ".join(map(str,z['reference_actions'])),fmtrow(z['single_action_loss'])] for c,z in local.items()]),
"Q, F, P는 각각 query view, factual view, probe다. 한 calibration당 5,760개 bank–world choices를 셌다. 하나의 action이 지배적인 조건이라면 관측된 noninferiority는 budget 선택이 어려운 worlds에서의 성과와 구별해야 한다.",
"### Gate만의 비용과 근사 오차",
f"동일한 factual input에 대해 cheap host-side score와 v23의 posterior top-512 score를 따로 timing했다. p50은 {us[0,0]:.1f} / {us[0,1]:.1f} µs, p90은 {us[1,0]:.1f} / {us[1,1]:.1f} µs였다(cheap / v23). V23에는 이미 fitted state를 제공했고 GPU 결과를 host에서 분기 가능하게 반환하는 synchronization을 포함했다. 위 end-to-end primary 측정과 별개의 추가 진단이다. Factual input이 처음부터 GPU에만 있는 시스템에서는 host gate 비용이 달라질 수 있다.",
tab(["Calibration","Score와 관찰된 low risk error의 Spearman ρ","Low / high centered risk error 평균","은행별 Δ 범위","은행별 p90 ratio 범위"],[[c,rho(sig[:,j,:,2].ravel(),sig[:,j,:,0].ravel()),f"{sig[:,j,:,0].mean():.5f} / {sig[:,j,:,1].mean():.5f}",rng(v['bank_delta'],5),rng(v['bank_ratio'],3)] for j,(c,v) in enumerate(rows.items())]),
"Spearman은 timing에 사용한 72 worlds와 5개 bank의 descriptive association이다. 추가 observation의 실제 label이나 효과를 score의 입력으로 쓰지 않았다.",
"### Factorial cell별 결과",
tab(["Curvature / friction 중심","Camera Δ / high %","Probe Δ / high %","Balanced Δ / high %"],[[f"{[-.29,0,.29][k//2]:+.2f} / {[.12,.32][k%2]:.2f}"]+[signed(v['strata'][k]['diff'])+" / "+rate(v['strata'][k]['high']) for v in rows.values()] for k in range(6)]),
"Cell별 Δ는 exploratory point estimate다. Primary 판정은 여섯 cells의 동일 가중 평균에 적용했다. 개별 물체나 모든 cell에 noninferiority가 보장된다는 뜻으로 확대할 수 없다.",
"### 최초 gate(A)의 독립 기록",
tab(["A · calibration","High 빈도","Gate−reference Δ · upper","p90 ratio · upper"],[[c,rate(v['rate']),signed(v['delta'][0])+" · "+signed(v['ub'][0]),f"{v['ratio'][0]:.3f} · {v['ratio_ub'][0]:.3f}"] for c,v in pilot_s['rows'].items()]),
f"A의 development 결과는 K{pilot_cfg['low']} / K{pilot_cfg['high']}였지만 모든 test 입력이 같은 분기를 탔다. A의 속도 또는 loss 결과를 adaptive routing의 근거로 합산하지 않았다.",
"![Calibration별 Brier noninferiority와 GPU p90 ratio](results/v24.png)",
"## 5. 논문 기여 가능성과 판단",
(root/"FINDINGS_KO.md").read_text() if (root/"FINDINGS_KO.md").exists() else "최종 해석 대기",
(root/"LITERATURE_KO.md").read_text(),
"## 재현 자료",
"""서버의 확인 실험 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/calibrated_router_v24`이다. [사전 protocol](PROTOCOL.md), [독립 simulator](../cheap_tail_v24/physics24.py), [factual features와 ridge](../cheap_tail_v24/gate24.py), [최종 실행 및 branch 배정](router24.py), [통계 분석](../cheap_tail_v24/stats24.py), [gate 비용 진단](overhead24.py), [후보별 validation](results/development.json), [primary 통계 및 각 cell 결과](results/summary.json), [최초 실행 결과](../cheap_tail_v24/results/summary.json)를 보존했다. Raw choices, losses, timing blocks, seeds, source·plan hashes는 각 실행의 results 아래에 있다. 같은 sibling directories와 기존 v18–v23 dependencies를 유지하고 빈 Slurm GPU에서 v18 guard로 run24.py(A), router24.py(B)를 각각 실행한 후 각 폴더에서 stats24.py를 compute-node CPU로 실행한다. B의 overhead24.py는 별도 guard log 폴더에서 실행한다. Paid sensor latency를 포함하는 real-world 실험 자료는 없다.""".replace("Paid sensor latency","실제 sensor 동작 시간")]
fig,axes=plt.subplots(1,2,figsize=(9.6,3.7),layout="constrained");colors=["#276776","#a66835","#524d80"]
for j,(key,v) in enumerate(rows.items()):
    for ax,point,bound in zip(axes,[v['delta'][0],v['ratio'][0]],[v['ub'][0],v['ratio_ub'][0]]):
        ax.plot([point,bound],[j,j],color=colors[j],lw=2);ax.plot(point,j,"o",color=colors[j]);ax.plot(bound,j,"|",ms=12,color=colors[j])
for ax,title,lim in zip(axes,["Gate minus reference Brier","Gate / reference GPU p90"],[.003,.95]):
    ax.axvline(lim,ls="--",color="0.4",label="Decision threshold");ax.set(yticks=range(3),yticklabels=cal,title=title,xlabel="Estimate to one-sided 95% upper bound",ylim=(-.6,2.6));ax.legend(fontsize=8)
fig.savefig(r/"v24.png",dpi=160)
(root/"REPORT_KO.md").write_text("\n\n".join(parts)+"\n")
