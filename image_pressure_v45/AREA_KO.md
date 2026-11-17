## 3. 같은 접촉 위치에서 pressure area의 효과

세 물리 조건은 정확히 같은 x에서 접촉을 시작했다. 정규화한 원형 pressure profile은 접촉 동안 반지름이 0에서 peak까지 커졌다가 줄어든다. Point control, peak 반지름 0.25 mm의 narrow, 2 mm의 broad를 비교했다. 매 순간 pressure centroid와 총 scalar normal impulse(물체별 0.910–2.048 N·s)는 맞췄다. 같은 하중을 더 넓게 분포시키므로 peak pressure는 낮아진다. Local normal의 방향이 달라질 수 있어 합성된 impulse vector 방향은 자유롭게 두었다. Effective pressure area는 (∫p dA)²/∫p² dA이며, 곡면의 면적 요소로도 계산했다. 이 개입은 최초 접촉점까지 움직였던 v44의 rigid sole 변화와 구별된다. [개입 식과 neutral control](PRESSURE_PROTOCOL_KO.md) · [모형 변수 검산](pressure_audit45.json)

D는 해당 면적에서 **항상 projection했을 때의 Brier − 항상 original을 썼을 때의 Brier**다. 앞 절의 선택 정책 효과와 구분한다. Narrow·broad D 구간은 네 비교에, broad−narrow와 형상 interaction의 구간은 세 비교에 대해 보정했다. [사전 지정 pressure 비교](score45.json)

| Shape | 곡면 effective area, narrow → broad (mm²) | Narrow D [동시 95% CI] | Broad D [동시 95% CI] | Broad−narrow D [동시 95% CI] |
|---|---:|---:|---:|---:|
| Fold | 0.18095 → 11.65051 | +0.009794 [+0.008909, +0.010679] | +0.013531 [+0.012592, +0.014470] | **+0.003738** [+0.003403, +0.004072] |
| Ripple | 0.18060 → 11.55776 | +0.002356 [+0.001745, +0.002967] | +0.002314 [+0.001691, +0.002937] | **−0.000042** [−0.000141, +0.000056] |

Fold−ripple interaction은 **+0.003780 [+0.003436, +0.004124]**였다. Fold에서는 면적을 넓힐 때 stick→slip과 slip→stick이 각각 D에 +0.002323과 +0.001415를 보탰다. Ripple에서는 두 전환의 기여가 +0.000563과 −0.000605였다. 즉 ripple에서도 약 3.58%의 label이 바뀌지만, 평균 Brier 손익은 상쇄된다. 사후 factorial ablation에서 위치별 normal만 바꿔도 fold의 면적 효과 +0.003730이 재현됐고, normal을 고정하고 위치별 friction만 남기면 +0.000009였다. 이 결과는 처방한 physics model 안에서 패치가 서로 다른 기울기를 포함하게 되는 경로를 지지한다. [전환별 분해와 ablation](PRESSURE_RESULTS_KO.md)

**결과를 바꿀 수 있는 정의.** Stick proxy는 충돌 후 중심 접선 속도 ≤0.020 m/s로 정의했다. 회전을 고정하고 하나의 translational slip mode에 대해 Coulomb capacity를 적분했다. Flat·uniform surface에서는 이론대로 면적에 따른 속도 차이가 사라졌다. 2,654,208개 충돌 조건 전체에서 수치 적분 해상도를 두 배로 높이면 label 956개(0.0360%)가 바뀌었고, fold의 broad−narrow D는 +0.003650으로 유지됐다. Ripple은 0.010 m/s 기준에서 −0.000138, 0.040 m/s에서 +0.000247로 방향이 바뀌었다. 따라서 ripple의 평균 손익이 작다는 해석에는 threshold를 붙여야 한다. 실제 pad의 탄성, 자연 형성되는 pressure, local incipient slip은 별도 측정이 필요하다. [정밀도·threshold 재검산](pressure_mechanism45.json) · [실물 partial slip 측정](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)
