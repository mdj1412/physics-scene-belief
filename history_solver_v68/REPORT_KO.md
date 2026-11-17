# v68 · 첫 calibration부터 접촉 법칙이 바뀌면 어떤 형상 정보를 남겨야 하는가?

**외부 엔진 PyBullet에서는 v67의 두 결과가 재현됐다.** 두 마찰 법칙 모두에서 projection axis를 약화하면 Wendland의 예측이 좋아지고 Dispersive의 예측이 나빠졌다. 부호 있는 좌표를 되돌려 주면 두 형상 평균 held-out Brier loss도 낮아졌다. **반면 접촉 부위가 눌리는 정도로 하중을 계산한 independent compliant solver에서는 재현 범위가 좁아졌다.** 특히 rate-strengthening에서는 사전 지정한 기본 해상도에서 형상 간 손익 반전과 평균 좌표 복원 이득을 모두 확인하지 못했다. Rate-weakening에서는 평균 복원 이득이 해상도를 높여도 남았지만, Wendland에서 축을 약화한 손익의 부호가 바뀌어 형상별 반전은 유지되지 않았다. 아래에서 Brier loss는 낮을수록 좋다. [GPU 예측과 절대 점수](GPU_RESULTS_KO.md) · [PyBullet full-history 결과](bullet_history/REPORT_KO.md) · [물리·수치 검산](SOLVER_KO.md)

## 1. v67보다 강해진 검정

v67은 동일한 관측 기록 뒤 마지막 impact의 법칙만 바꿨다. 이번에는 calibration 물체 네 개의 factual contact와 네 번의 유료 probe, 새 물체의 factual contact, 실제로 선택한 probe, 마지막 impact까지 **해당 시행의 같은 law와 solver**를 사용했다. 두 law는 진입 접선 속도와 마찰계수의 관계가 반대인 rate-strengthening과 rate-weakening이다. 각 충돌에 진입할 때의 속도로 계수를 정하고 그 충돌 중에는 값을 유지했다. 같은 형상·표면·조명·noise와 observation 예산을 짝지었다. 미구매 observation과 실제 law 이름은 predictor에 들어가지 않는다. [사전 계획과 정보 경로](STUDY_KO.md)

v67의 네 source axis와 여덟 readout은 고정했다. v68의 주 predictor는 v38에서 학습한 optimizer로 네 calibration 측정의 공통 log-odds 보정량을 계산한다. 각 환경에서 새 factual observation으로 128개 물체의 acquisition을 다시 계산해 13개에 probe, 115개에 contour query를 배정했다. 평가에는 v66에서 source와 분리된 12개 catalog와 총 6,144개 prototype draw를 사용했다. 같은 물체를 다시 사용해야 이력과 label의 변경을 짝지어 분석할 수 있다. PyBullet에서는 128개 모두로 구매를 정하되 사전 지정한 환경별 첫 8개에서 outcome을 평가했다. 이 외부 대조는 12개 catalog 총 384개 draw이며 서로 다른 설정을 재생한 terminal contact 110,592개 모두 유효했다.

## 2. 무엇이 재현됐는가

두 개의 서로 다른 비교를 수행했다. **부분 억제**는 같은 likelihood에서 축의 가중치를 약화한다. **좌표 복원**은 같은 구조의 source-trained readout 두 개에 해당 축의 좌표를 제거하거나 부호까지 돌려주는 비교다. 따라서 Wendland에서 약화의 이득과 좌표 복원의 손해가 동시에 나올 수 있다.

| 첫 calibration부터 사용한 solver | Law | Wendland, 부분 억제 − 원래 likelihood | Dispersive, 부분 억제 − 원래 likelihood | 두 형상 평균, restored − dropped |
|---|---|---:|---:|---:|
| Prescribed pressure | Strengthening | −0.003089 | +0.006714 | −0.012642 |
| Prescribed pressure | Weakening | −0.003596 | +0.007442 | −0.022122 |
| Compliant · 9 contact sites | Strengthening | −0.001036 [−0.001426, −0.000646] | −0.000128 [−0.000818, +0.000561] | +0.000559 [−0.000548, +0.001667] |
| Compliant · 9 contact sites | Weakening | −0.001525 [−0.001903, −0.001148] | +0.000965 [+0.000150, +0.001780] | −0.006015 [−0.007432, −0.004599] |
| Compliant · 17 contact sites, 이력과 정답 모두 재생 | Strengthening | −0.000724 [−0.001057, −0.000390] | +0.000162 [−0.000506, +0.000830] | +0.000044 [−0.001114, +0.001202] |
| Compliant · 17 contact sites, 이력과 정답 모두 재생 | Weakening | +0.000351 [+0.000014, +0.000687] | +0.001091 [+0.000338, +0.001844] | −0.006546 [−0.007952, −0.005140] |
| PyBullet · rigid sphere/mesh | Strengthening | −0.002339 [−0.003850, −0.000829] | +0.005745 [+0.003823, +0.007667] | −0.016961 [−0.020639, −0.013283] |
| PyBullet · rigid sphere/mesh | Weakening | −0.003062 [−0.004247, −0.001877] | +0.006553 [+0.004570, +0.008536] | −0.023440 [−0.027601, −0.019279] |

독립 단위는 catalog다. 기본 compliant 여섯 primary contrast의 구간과 PyBullet 여섯 secondary contrast의 구간은 각각 묶음 안에서 동시 95%로 보정했다. 17-site 결과는 초기 해상도 진단 뒤 추가한 sensitivity analysis다. 이 구간에도 동일한 여섯 비교 보정을 적용했으며, 기본 검정과 형식이 같은 별도 확인 실험으로 해석하지 않는다. 좌표 복원의 사전 실용 기준은 두 형상 평균 차이의 구간 상한이 −0.002보다 낮은 것이다. [Catalog별 Brier](score_main68.json) · [PyBullet source 및 catalog별 결과](bullet_history/external_history_result.json)

![세 모델과 두 compliant 해상도에서의 Brier 차이](full_history_v68.png)

각 점은 catalog별 paired Brier 차이의 평균이다. 가로선은 해당 모델·해상도의 여섯 비교 안에서 보정한 95% 구간이다. 음수는 각각 부분 억제 또는 좌표 복원 쪽이 유리함을 뜻한다. PyBullet은 같은 catalog의 사전 지정 부분 표본이다. [벡터 그림](full_history_v68.pdf)

**두 형상 평균만 보면 좌표 복원에 대한 설명을 잘못할 수 있다.** 기본 compliant의 weakening에서 Wendland는 dropped **0.233880** → restored **0.236469**로 나빠졌고, Dispersive는 **0.201818 → 0.187199**로 좋아졌다. Strengthening에서도 Wendland의 차이는 +0.004711, Dispersive는 −0.003592였다. Strengthening에서는 네 source seed 중 두 개에서만 평균 복원 차이가 음수였다. 이에 비해 PyBullet에서는 두 law에서 모두 형상 평균의 복원 이득이 −0.002 기준을 넘었다. 모델이 기존 source에서 학습한 좌표를 새 엔진에서 어떻게 사용하는지에 관한 결과다. 새로운 target에서 취득 가능한 최적 정보량을 직접 측정하지는 않았다.

## 3. 같은 물체에서 이력과 정답을 교체한 검산

1. **법칙 변화는 calibration에 도달했다.** Compliant에서 짝지은 두 law 간 calibration probe 속도의 성분별 RMS 차이는 약 0.1395 m/s였다. 측면 진입 속도가 0인 factual 충돌에서는 약 0.0000061 m/s였다. 광학 profile은 같았다. 다른 법칙의 calibration만 넣고 나머지 관측과 label을 그대로 두자 full predictor의 stick 확률은 형상·law별 평균 절대값으로 약 0.028–0.031 달라졌다. Calibration은 예측에 실제로 사용됐다. [물리 입력 대조](input_audit68.json) · [구매·네 번째 cue 검산](information68.json)
2. **사용한 보정이 항상 유용하지는 않았다.** Compliant에서 두 형상 평균 full predictor의 calibration 적용−생략 Brier 차이는 strengthening **+0.002677**, weakening **+0.002039**였다. 다른 law의 calibration으로 바꾼 대조에서는 두 형상 평균 full predictor의 올바른 이력−교체한 이력 차이가 각각 +0.000965, −0.000400이었다. 두 교체 차이의 exploratory paired 95% 구간은 모두 0을 포함했다. 보정을 생략해도 기본 compliant의 strengthening에서 pooled restoration이 개선되지 않았다. 기존 optimizer를 이 엔진의 최적 보정으로 간주할 근거는 부족하다. [교체 조건별 절대 점수](crossover68.json)
3. **상대 성능이 움직인 주요 경로는 terminal label이었다.** Raw predictor를 고정하고 v67→compliant의 입력 이력과 terminal label을 모두 교차했다. Dispersive·strengthening의 soft−full 차이는 총 −0.006964 움직였다. 바꾸는 순서를 양쪽으로 평균하면 label 교체 몫은 −0.006968, 관측 이력 교체 몫은 +0.000004였다. 같은 조건에서 restored−dropped의 총 변화는 +0.016818, 그중 label 교체 몫은 +0.016830이었다. 실제로 두 주 solver는 family·law별로 16.70~23.37%의 stick/slip 판정이 달랐다. 이는 이 benchmark의 같은 물체에 대한 분해다. 다른 관측 정책이나 물리 장치의 인과 기여율까지 정하지 않는다. [사전에 고정한 교차 설계](CROSSOVER_PLAN_KO.md) · [수치 재생](crossover68.py)
4. **정답을 같게 둔 PyBullet 대조도 수행했다.** PyBullet label을 공통으로 쓰고 입력만 compliant→PyBullet로 바꾸면 어느 물체에서 관측을 살지는 평균 14.35% 바뀌었다. 그럼에도 strengthening의 평균 복원 차이는 −0.017636→−0.016961, weakening은 −0.024254→−0.023440이었다. 같은 물체에서 어느 관측을 샀는지가 달라지는 빈도와 최종 예측 비교의 크기는 다른 양이다. [같은 Bullet label 대조](bullet_score68.json) · [처음부터 Bullet로 계산한 대조](bullet_history/REPORT_KO.md)

## 4. 수치적으로 어떤 결론까지 믿을 수 있는가

**해상도는 작은 projection 손익의 부호를 바꿨다.** 처음 계획한 9-site solver의 step은 약 4.63 μs였다. 각 환경의 첫 32개 물체에서 step을 절반으로 줄이고 site를 17개로 늘렸을 때 Wendland·weakening의 soft−full 부호가 바뀌었다. 이를 본 뒤 전체 128개에서 terminal label과 calibration 이력을 모두 더 미세한 설정으로 다시 생성했다. 가중치·law·cutoff와 구매 예산은 고정했다. Fine label 생성과 fine GPU 예측을 병렬 수행했고, 추가 GPU 작업은 첫 32개 subset의 결과가 이미 알려진 상태에서 수행한 사후 진단이다. 해당 예측을 봉인한 뒤 전체 fine 결과를 채점했다. [고해상도 이력 사전 명시 및 실행](fine_history/PLAN_KO.md) · [Full-history refinement](fine_history/FINE_HISTORY_KO.md)

- **Wendland·weakening**에서 9-site의 −0.001525가 17-site에서 **+0.000351**이 됐다. 기존 이력을 두고 terminal label만 17-site로 바꾸면 이미 **+0.000376**이었다. 추가로 이력까지 17-site로 바꾼 효과는 −0.000025였다. 이 차이는 주로 terminal 접촉 판정에서 왔다. 동일한 17-site에서 Dispersive도 +0.001091이므로 두 형상 모두에서 축 억제가 손해였다.
- **좌표 복원**은 17-site에서도 weakening에서 **−0.006546**, strengthening에서는 **+0.000044**였다. 전자는 두 형상의 평균이며 17-site에서도 Wendland의 절대 Brier는 0.242292→0.244891, Dispersive는 0.196488→0.180798이었다. 별도로 각 환경 첫 8개에 같은 9-site 예측을 두고 terminal을 33-site까지 올려도 weakening 평균 복원 이득은 −0.006954 [−0.010302, −0.003605]였다. 이 작은 subset의 33-site 조건에서는 두 law의 네 형상별 soft−full 구간이 모두 0을 포함했다. 세 해상도 결과로 연속체의 참값을 결정할 수는 없다. [해상도별 catalog 결과](refined68.json) · [이력과 label을 따로 바꾼 결과](fine_history/fine_scores.json)
- **종료 판정.** 새 예측을 봉인한 뒤 일부 compliant trajectory에서 모든 국소 평면으로부터 분리됐는지 확인되지 않았다. 원래 상한 3.33 ms에서 종료되지 않은 경우만 33.33 ms까지 연장하고, 그래도 미확정이면 그 시점의 속도와 별도 표시를 남겼다. 기본 1,769,472개 중 175개, 17-site 전체 중 86개였다. 불확실한 binary label을 모두 어느 쪽으로든 허용해도 weakening의 17-site 평균 복원 차이는 **[−0.006546, −0.006544]**였고, 동시 구간의 가장 보수적인 상한도 −0.005137이었다. Wendland의 손익 반전이 무너진 원인은 이 미확정 사례를 임의 처리한 데 있지 않았다. [종료 기준 수정 기록](SOLVER_STATUS_KO.md) · [기본 해상도의 최악 경계](unresolved_bounds68.json)
- **PyBullet의 step과 stick 기준.** 같은 모든 PyBullet 물체에서 terminal step을 20→10 μs로 줄이자 110,592개 중 4,154개(3.76%)의 label이 달라졌다. 두 law에서 projection 반전은 유지됐고 평균 복원 차이는 −0.016111 및 −0.023697이었다. 이 대조에서는 history 해상도는 원래 값이다. 별도로 기본 compliant solver에서 stick cutoff를 0.010/0.020/0.040 m/s로 바꾸자 weakening의 평균 복원 차이는 −0.002209/−0.006015/−0.015001이었고, Dispersive soft−full은 −0.00105/+0.00097/+0.00563이었다. 지정한 terminal event가 어떤 역할을 하는지도 결과에 포함해야 한다. [PyBullet 전체 step 검산](bullet_halfstep68.json) · [Cutoff 분석](threshold68.json)

## 5. 논문 기여와 다음 판정

이번 결과로 지지할 수 있는 기여는 **하나의 고정된 형상 표현이 다른 접촉 모델로 전이될 때, 어떤 관측·물리 endpoint에서 그 예측 가치가 바뀌는지 분해한 것**이다. Prescribed pressure와 PyBullet에서는 v67의 방향이 남았다. 하중을 국소 침투로 정한 compliant 모델에서는 접촉 해상도에 따라 작은 projection 이득의 부호가 달라졌고, signed coordinate를 복원한 이득은 주로 Dispersive와 rate-weakening에 집중됐다. 형상 정보 보존과 likelihood 억제를 각각 실험해야 하는 이유를 같은 물체에서 보여준다.

외부 타당성의 병목은 실제 접촉 중 하중 위치와 terminal slip을 어느 solver가 더 잘 설명하는지다. 세 엔진은 contour와 진입 속도 기반 마찰 법칙을 공유하지만 pressure footprint, shoe와 rigid tip, 사용하는 contact normal, 종료 방식이 다르다. 다음에는 **같은 물체·표면의 반복 impact에서 contact patch와 접촉 직후의 접선 속도를 함께 측정**하고, solver별 연속 속도 오차와 사전 등록한 stick cutoff의 Brier를 함께 평가하는 것이 우선이다. 같은 실제 사실에 두 엔진을 맞춰야 어느 물리 가정이 두 형상에서 실제로 정보 손실을 만드는지 판별할 수 있다. [원 연구 6편 비교·증거의 강도·후속 연구 질문 10개](LITERATURE_KO.md) · [실측 충돌 연구](https://proceedings.mlr.press/v78/fazeli17a.html) · [Cube와 Cassie의 solver 실측 비교](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

재현 자료는 datacenter3의 기존 프로젝트 아래 history_solver_v68에 있다. 기본 GPU 예측은 유휴 상태를 확인한 Slurm 2744471의 NVIDIA RTX PRO 6000에서 실행했다. 24개 shard의 예측·검증·저장에 283.5초가 걸렸다. PyBullet과 fine-history GPU 실행도 각각 빈 장치를 확인했다. Contact dynamics와 label의 CPU 계산은 Slurm compute node에서 수행했다. 본 보고서에는 실제 물리 장비에서 얻은 결과가 없다. [GPU 실행 기록](gpu_run.json) · [예측 봉인](FORECASTS_SEALED) · [실험별 재현 코드](score_main68.py)
