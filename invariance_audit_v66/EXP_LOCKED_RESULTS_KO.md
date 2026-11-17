# V66 · 새 contour family의 label-blind 확인 실험

같은 source checkpoint를 새로운 두 family에 옮겼을 때, **restored − dropped**의 전체 정책 Brier 차이는 **-0.043678 [-0.045457, -0.041899]**였다. 두 retained 조건의 사전 판단과 실제 결과를 아래에 함께 적었다. [사전 가설](EXP_LOCKED_PLAN_KO.md) · [서버 결과](exp_locked_results.json)

## 비교 대상과 시간 순서

새 compact Wendland 계열과 dispersive shoulder 계열을 각 3개 variant로 생성하고 support 980–991의 12개 catalog에서 평가했다. 기존과 같은 inlay/woven과 rational/gaussian_rate를 쓴다. 따라서 검증한 이동은 새 contour·catalog로의 transfer다. 기존 v46 분석을 본 후 가설을 정했으며, 신규 solver label은 모든 locked forecast를 봉인한 후 생성했다. 두 높이와 3 offset, 2 finite radius, noise 4회는 대응 조건이고, 각 catalog에 같은 가중치를 부여했다. 새 object 6,144회 추출의 고유 prototype은 5,945개다. [형상과 사전 split](locked/LOCK_KO.md) · [중복 검증](locked/world_audit.json)

Source 학습은 v45의 8개 training·4개 validation catalog에서 이미 완료됐다. 3 source-fit seed × 3 representation의 9개 checkpoint, feature statistics, epoch, axis, common hard-projected physics feature를 변경하지 않았다. 단일 GPU에서 각 구매 조건의 prediction을 작성하고 12개 target 파일의 SHA를 봉인했다. 이어서 독립적으로 생성한 binary label과 결합했다. 3개의 fit은 하나의 ensemble로 합치지 않고 개별 seed의 Brier를 평균했다. Probe를 산 13/128에서는 다섯 방법 모두 같은 forecast를 사용했다. [소스 checksum](exp_locked_plan.sha256) · [예측 봉인](exp_locked_seal.sha256) · [실행 코드](exp_locked.py)

| Predictor | 전체 Brier | Query 구입 시 | Wendland 전체 | Dispersive 전체 |
|---|---:|---:|---:|---:|
| Frozen physics · full | 0.166229 | 0.158251 | 0.183312 | 0.149145 |
| Frozen physics · dropped | 0.191364 | 0.186228 | 0.185135 | 0.197594 |
| Retrained · full | 0.135604 | 0.124164 | 0.160769 | 0.110438 |
| Retrained · dropped | 0.182129 | 0.175948 | 0.185320 | 0.178937 |
| Retrained · restored | 0.138451 | 0.127333 | 0.164419 | 0.112482 |

두 비교의 95% simultaneous CI는 catalog-paired t interval에 Bonferroni 보정을 적용했다. 비교별 단측 예측을 했어도 양측 구간을 사용했다. 0.002는 운영상 중요도를 가늠하기 위해 target label 생성 전에 정한 Brier 차이다. 로봇의 실제 utility로 보정한 값은 아니다.

| Primary contrast | 이전 terrace/dimple | 신규 두 family | 사전 판정 |
|---|---:|---:|---|
| Restored − dropped | -0.027100 [-0.028674, -0.025526] | -0.043678 [-0.045457, -0.041899] | 예측한 방향 및 사전 제안한 0.002 기준 충족 |
| Full − dropped | -0.027131 [-0.028877, -0.025384] | -0.046525 [-0.048234, -0.044816] | 예측한 방향 및 사전 제안한 0.002 기준 충족 |

가족별 restored − dropped는 Wendland -0.020901 [-0.022164, -0.019637], dispersive -0.066456 [-0.069234, -0.063677]였다. 두 family 각각의 방향은 사전에 확정하지 않은 보조 분석이다. Restored − full은 +0.002847 [+0.002536, +0.003158]였다. 이 비교에는 equivalence margin이 없다.

## 어떤 입력 이동을 실제로 관찰했나

Source training의 7개 paid range channel을 각각 training 평균·표준편차로 정규화했다. |z|>3인 channel 측정값의 비율은 training 0.0102%, source validation 0.0094%, 기존 v46 0.0697%, 이번 locked target 0.1197%였다. Locked target 최대 |z|는 3.707였다. 이 진단은 심한 marginal extrapolation의 한 측면을 보며 joint support와 label conditional의 일치를 보장하지 않는다. [Source·기존 target 진단](exp_shift.json) · [신규 target label-free 기록](exp_locked_forecasts.json)

## 연구 해석

동일한 acquisition과 같은 학습 예산에서 상대 risk를 검정했다. 삭제한 signed coordinate를 보존하는 것이 어디서 이득인지에 대한 증거는 이 simulator의 surface·law·표본 가중치를 조건으로 한다. 같은 feature에서 가능한 모든 readout을 최적화한 Bayes risk, 새로운 물성, 실측 영상·impact에서의 효과는 여기서 추정하지 않았다. 압축된 예측기의 loss도 finite learning·likelihood error를 포함한다. Exact-alignment twin의 conditional floor와 실제 두 readout의 held-out Brier 차이는 서로 다른 질문으로 해석한다. [능동적 정보 소실 진단](EXP_RESULTS_KO.md) · [전체 catalog·seed별 값](exp_locked_results.json)

**가역 좌표의 해석.** Full과 restored는 같은 정보를 보존해도 서로 다른 source normalization과 coordinate에서 최적화한 finite readout의 transfer risk는 달라질 수 있다. 두 arm 간 차이에 대해 정보량의 차이를 주장할 수 없다. 학습·정규화·최적화 중 어느 요소가 어느 정도 기여했는지는 개별 ablation이 필요하다.

Forecast seal의 파일명 끝에 붙은 LF를 원래 verifier가 경로에 포함해 첫 실행이 중단됐다. 별도 entrypoint가 LF만 제거하고 frozen scoring 함수를 그대로 호출한다. 원래 seal에는 sha256sum 검증을 먼저 적용했다. 이 수정 뒤 target을 다시 예측하거나 통계식을 바꾼 사항은 없다. [수정 내용](exp_locked_score_fix.py) · [기존 실행 로그](exp_locked_score.log) · [완료 로그](exp_locked_score_fixed.log)

![두 target의 paired Brier 차이와 별도 exact-alignment twin 진단](exp_readout66.png)

그림 A와 B의 horizontal interval은 두 primary contrast에 대한 동시 95% 구간이다. C의 값은 별도 query twin 모집단의 absolute projected Brier floor이고, 구간은 catalog 95% interval이다. [벡터 PDF](exp_readout66.pdf)
