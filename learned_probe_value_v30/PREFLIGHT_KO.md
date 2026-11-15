# v30 · 학습 준비와 서버 CPU 검증

**확인된 연구 결과.** V29에서 실제로 측정한 Brier 감소량을 그대로 v30의 학습 label로 쓸 수 있다. 그러나 source 환경별로 어느 곳에서 이득이 큰지를 매기는 순위가 64개 held-out worlds의 선택에 따라 크게 변했다. 반면 32회 반복한 sensor noise가 만드는 오차는 작았다. 미래 물체 수와 독립 calibration futures 수가 scorer 검증의 중요한 표본 단위다.

| 확인 항목 | 2→4회 | 4→8회 |
|---|---:|---:|
| 개발용 환경 수 | 192 | 192 |
| 동일 예산 선택을 유지한 두 64-world 부분표본 간 gain 순위 상관 (Spearman, block bootstrap 95%) | 0.099 [−0.008, 0.211] | 0.010 [−0.133, 0.139] |
| 두 부분표본 간 gain의 평균 절대 차이 | 0.01211 | 0.01088 |
| 전체 128개에서 얻은 환경별 gain의 표준편차 | 0.00854 | 0.00683 |
| Worlds·label·선택을 고정한 32회 sensor noise Monte Carlo 표준오차의 RMS | 0.00027 | 0.00025 |

두 부분표본은 전체 128개에서 이미 정한 top 13 관측 선택을 공유하므로 독립 deployment batch 두 번을 재현한 것은 아니다. 확인한 것은 같은 source data에서 환경별 realized gain 순서의 불안정성이다. 이 진단으로 새로운 scorer의 실패를 확정하거나 unseen law Brier를 예측할 수는 없다. [서버 CPU 결과](source_target_noise.json) · [재현 코드](target_noise30.py)

## 구현한 비교

Source는 v29의 Coulomb law 24개 block, 192개 환경이다. 각 stage의 target은 해당 prefix에서 중지했을 때의 common-decoder Brier에서 2회 또는 4회를 더 관측했을 때의 Brier를 뺀 값이다. 학습은 실제 센서 응답과 posterior의 통계량 15개를 입력받는 두 개의 ridge regressions으로 구현했다. Ridge penalty는 source block 단위 nested grouped validation으로 고르게 했다. [Scorer](score30.py)

새 물체에서 정밀 boundary profile인 query view와, 다른 입사속도의 접촉으로 얻는 2D outgoing velocity인 probe 중 하나를 고른다. 평가 대상은 다른 충돌의 stick probability다. Learned, fixed 4, uncertainty-only가 한 block에서 같은 calibration probes 32회를 산다. Primary batch 설정에서는 환경당 128개의 미래 물체 중 13개에서만 probe를 받아 총 test probes도 104회로 같게 했다. Primary decoder는 population coupling을 알고 source contact law를 가정하며, law-aware common decoder는 사후 sensitivity용이다. 두 held-out laws에서 각각 새 blocks 16개를 쓰는 설계다. 결과 판정은 learned − fixed 4 및 learned − uncertainty-only 두 대비의 simultaneous 95% interval에 따른다. [Run 설정](design.json) · [사전 분석 계획](ANALYSIS_PLAN.md) · [GPU runner](runner30.py)

## 실제로 바뀐 물리 현상

서버 CPU에서 최종 test와 다른 diagnostic seeds로 같은 숨은 물체에 두 가지 새 contact law를 적용했다. Static–kinetic split에서는 Coulomb law의 static threshold를 1.2배, kinetic impulse의 계수를 0.8배로 바꿨다. Speed weakening에서는 tangential speed에 따라 계수가 연속적으로 변한다. Source 관측을 단순히 다시 샘플링한 것이 아니다.

| CPU 진단 | Static–kinetic split | Speed weakening |
|---|---:|---:|
| Source 대비 factual 관측의 평균 변화 | 0 | 0 |
| Probe velocity 차이, 평균 L2 norm | 0.08501 | 0.14378 |
| Source와 다른 query stick label | 10.25% | 5.62% |

기존 접촉의 factual evidence는 같지만 새 probe의 측정값과 최종 정답이 모두 달라질 수 있음을 확인했다. 320개 신규 world 및 numerical bank seeds의 중복 검사를 통과했다. 이 표의 값은 policy 성능이 아니다. [검산 원본](preflight.json) · [두 contact laws](contact30.py)

## GPU 측정 상태

이 사전 검증본 작성 시 세 allocation 모두 사용 중이었다. 2744468에는 반복 실행되는 Blender GPU step이 있었고 나머지 두 장에도 각각 약 40 GB를 쓰는 Python process가 있었다. 새 모델 training과 inference는 보류했다. 따라서 이 문서가 보장하는 범위는 서베이, 구현, 학습 label과 물리 법칙의 CPU 검산까지다. 세 정책의 unseen-law held-out Brier, GPU p90, 방법의 우월성은 아직 측정되지 않았다. GPU 실행 시점에 device process와 memory를 다시 확인하는 진입점을 준비했다. [실행 파일](run_gpu.sh)

근거가 남는 논문 질문은 *실제 loss 감소량을 학습해도 어떤 상황에서 환경 간 순위를 안정적으로 정하기 어려운가*이다. Error-reduction regressor 자체는 LAL에 선례가 있으며, common-decoder 교차 평가 또한 Action-BED에 선례가 있다. 실험에서 두 baseline을 이기고 그 결과가 target noise, decoder mismatch, 환경 수에 얼마나 의존하는지 확인해야 방법 기여를 판단할 수 있다. [네 연구자 서베이 및 10가지 후속 질문](SURVEY_KO.md)
