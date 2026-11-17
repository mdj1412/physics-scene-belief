# Butterfly Super Anti · 공개 영상에서 상대 속도와 회전 예측

Preston 영상의 24개 top/back 조건에서 광학 판독 기준을 충족한 10개 impact를 평가했다. 최초 운동 구간의 20개 free-flight window, 인접 frame 간 435개 구간에서 중복·정지 frame이 발견되지 않았다. Camera capture rate와 편집 전후의 실제 시간 배율은 기록에서 확인되지 않았다. 같은 impact의 입사와 이탈에서 시간 배율이 같다는 조건에서, 각 velocity를 입사 normal speed로 나누면 공통 시간 단위가 소거된다. Angular velocity에도 같은 처리를 적용하려고 image ball radius와 각속도를 곱한 rim speed를 썼다. Source 여섯 제품도 같은 방식으로 변환해 모델을 다시 fit했다. 따라서 아래의 두 오차는 무차원이고 Feint OX의 m/s·rad/s와 합산할 수 없다. [광학 기준](../audit/preston/quality_protocol.md) · [창별 cadence](../audit/preston/validation/cadence.csv) · [프로토콜](ANTI_PROTOCOL_KO.md)

## 같은 test와 calibration 횟수

들어오는 normalized slip의 크기와 방향을 따라 정렬한 후 8개의 후보를 균등 간격으로 골랐다. 나머지 두 impact는 Ref 1(top), Ref 24(back)다. 이 공통 holdout은 K=4와 K=8 모두에 적용된다. K=4는 후보 8개 중 모든 70개 조합, K=8은 한 조합이다. 실제로 구매한 수는 불확실한 sign이 나와도 줄이지 않는다. Majority vote의 분모에서는 optical uncertainty interval이 0을 포함하는 sign을 보류한다. 유효한 vote가 동률이거나 전혀 없으면 원래 routing을 유지한다.

| 정책 | K=4 normalized velocity / spin rim error | K=8 normalized velocity / spin rim error | K=8 sign이 판독된 유일한 held Ref 1의 output sign |
|---|---:|---:|---|
| Majority vote | 0.172 / 0.355 | 0.172 / 0.355 | 일치 |
| Continuous-loss gate | 0.221 / 0.036 | 0.221 / 0.036 | 불일치 |
| Single affine EB | 0.206 / 0.238 | 0.219 / 0.224 | 일치 |
| Elastic impulse | 0.106 / 0.180 | 0.101 / 0.178 | 일치 |

고정된 두 impact에서 velocity vector는 elastic impulse가, spin rim speed는 continuous-loss gate가 상대적으로 작은 오차를 냈다. K=4와 K=8 모두 majority vote는 original, continuous-loss gate는 opposite routing을 선택했다. Calibration 8개 중 sign을 실제로 판독한 것은 Ref 17 하나다. K=4의 70개 조합 중 절반은 sign을 한 건도 판독하지 못했고 나머지 절반도 한 건뿐이다. [각 정책의 수치](anti_audited/interval/result.json) · [각 held impact의 관측과 예측](anti_audited/interval/k8_held.csv)

**이 제품에서 검출률을 확정할 수 있는가?** Optical perturbation에서 이탈 부호가 유지된 impact는 10개 중 Ref 1과 17 두 개였다. Ref 1의 이탈 상대속도 범위는 −2.26~−0.02 px/frame, Ref 17은 −8.25~−4.77 px/frame이다. Ref 17의 입사 부호는 불명확해 입사·이탈 간 물리적 부호 반전을 진단하기도 어렵다. Ref 24의 이탈 범위는 −1.11~+1.04 px/frame이어서 부호 판정에서 보류했다. 따라서 두 방법의 flip 빈도를 이 rubber의 실제 반전 검출률로 일반화할 수 없다. Interval 전체가 0에서 0.1 px/frame 이상 떨어져야 한다는 더 강한 기준에서는 공통 held 2점 중 판독 가능한 것은 0점이다. [독립 optical interval](../audit/preston/sign_sensitivity.csv) · [더 강한 판독 기준](anti_audited/margin_010/result.json)

## 분할과 시간 배율에 대한 민감도

- 서로 다른 256개 random 8-shot prefix에서 앞 4회와 8회를 비교하고 공통으로 남긴 두 impact를 평가했다. K=8의 평균 velocity / spin rim error는 majority 0.352 / 0.488, continuous-loss gate 0.392 / 0.184, single affine EB 0.416 / 0.254, elastic impulse 0.309 / 0.296이었다. 동일한 10개 impact의 재조합이다.
- 한 window의 전·후반 normal displacement 비율이 0.8~1.25에 들어오는 경우만 남긴 민감도에서는 Ref 11이 빠졌다. 9개에서 분할을 다시 고정하면 held는 Ref 9 하나가 되고 이탈 sign은 판독되지 않는다. 이 한 점의 K=8에서는 elastic impulse가 두 지표 모두에서 가장 작았다. 동일 분할을 여러 개 보는 paired random 결과에서는 velocity는 elastic impulse, spin rim error는 continuous-loss gate가 가장 작았다. [9개 민감도](anti_audited/steady_normal/result.json)
- Post-impact의 모든 속도·각속도에 0.5 또는 2를 곱해 미확인 상대 시간 배율에 대한 stress test를 했다. 0.5일 때 K=8의 최소 velocity error는 EB 0.092이고 elastic impulse는 0.118이었다. 2일 때는 elastic impulse가 0.159로 최소였다. Continuous-loss gate는 각 조건에서 spin rim error 0.035, 0.051로 가장 작았다. 이 배율은 실제 편집 속도의 추정치가 아니다. [0.5배](anti_audited/post_half/result.json) · [2배](anti_audited/post_double/result.json)

모든 조건을 빈 GPU에서 실행하고 CPU의 별도 NumPy 구현으로 같은 common-holdout 결과를 재현했다. 이 비교는 두 성분의 signed optical estimate를 확보한 한 영상의 조건부 검증이다. 제품 간 release 특징의 차이, 절대 속도 오차, 새로운 session에서 순위가 유지된다는 주장을 추가하려면 독립 capture timing과 더 많은 시편·실험 조건이 필요하다. [GPU 프로그램](anti.py) · [CPU 검산 결과](anti_audited/interval/audit.json)
