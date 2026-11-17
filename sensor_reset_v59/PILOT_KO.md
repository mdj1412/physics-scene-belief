# v59 · 무작위 sensor reset과 독립 camera로 같은 전자 판독의 원인을 구별하기

**선택한 센서의 출력을 초기화하는 것만으로는 physical persistence와 common-mode error를 구별할 수 없었다.** 두 전자 센서가 동일한 upstream 신호를 받기 때문이다. 한 surface에 쓸 수 있는 여덟 impact에서 독립 camera를 더하면 원인 판별률은 50.00%에서 55.17%로 올라갔다. Camera를 사용하면서 둘째 물체 직전에 무작위 선정한 한 센서를 reset하면 55.81%가 됐다. [사전 계획](PILOT_PLAN_KO.md) · [서버 결과](diagnostics59.json)

## 원인의 구성과 독립성

A와 B의 true reference–impact mismatch 평균은 q=29%다. Physical hypothesis(P)에서는 한 session의 서로 다른 두 물체가 같은 latent e를 공유한다. A에서 두 물체가 모두 mismatch인 확률 m은 .145, B에서는 .0925다. Common-mode hypothesis(C)에서는 물체별 true mismatch가 독립이다. 대신 확률 c=(m−q²)/(q(1−q))로 두 센서 모두에 물리 부호와 무관한 동일한 session-wide phantom을 제공한다. c는 A에서 29.58%, B에서 4.08%다. 이 두 원인과 A/B는 독립적으로 교차했다. Policy가 고를 행동은 A/B에 달려 있으며 어떤 센서가 고장 났는지를 곧바로 action label로 쓰지 않았다.

두 원인에서 upstream pair의 네 가지 출력 pattern은 동일한 확률을 갖는다. 센서의 초기 상태와 local latch, 판독오차, randomized reset을 여기에 동일하게 적용해도 이 동등성은 유지된다. 실험에서는 common-mode 환경의 upstream pair가 P의 true pair가 되도록 paired simulation을 구성했다. 따라서 물리 표본은 다르면서도 매 episode에서 두 원인의 전자 센서 판독이 같은 조건을 직접 비교할 수 있었다. [Pilot 생성](pilot59.py) · [독립 likelihood](route59.py)

전자 sensor마다 26.58% 확률로 저마다의 local latch에 걸린다. Reset은 지정 센서의 local latch만 완전히 해제한다. 2.0 N reference와 impact의 부호에는 각 2% 독립 error를 주었다. 물리 변형을 읽을 수 있는 비율은 50%다. Analytic camera는 active 물체의 80%에서 유효하며 reference와 impact 부호를 각각 3% 확률로 잘못 읽는다. 같은 contact에서 별도로 취득하는 camera의 signal source, noise와 visibility는 전자 센서의 오류와 독립이다. 설정값은 실물 장비의 추정값이 아니다.

각 catalog의 두 material에서 2,048개의 독립 pilot을 생성해 24개 catalog, 실험 cell당 98,304개의 8-impact pilot을 평가했다. 393,216개 물체 쌍에서 객체 순서는 50.06%가 BA, reset target은 49.92%가 sensor 1이었다. No-reset에서도 같은 target ID를 가상 배정해 동일한 물체와 sensor noise에서 paired contrast를 구했다. 한 표면의 네 session마다 서로 다른 물체 두 개, 8 reference, 8 impact, 16 contact resets를 지불했다. Reset arm에서는 접촉을 추가하지 않고 electronic sensor reset 네 번을 더 쓴다. [Generation freeze](pilot_code59.sha256) · [Raw 판독 검산](diagnose59.py)

## Reset으로 무엇이 바뀌었는가

아래 double mismatch는 양쪽 물체를 해당 instrument로 읽은 경우만 분모에 넣은 비율이다. Reset은 둘째 물체를 관측하기 전에 수행했다. 선택한 센서의 유효한 둘째 물체 판독 중 실제 mismatch bit가 바뀐 비율은 A/B에서 각각 11.00%/10.93%였다.

| 환경과 원인 | 선택한 sensor의 두 물체 no-reset → reset | Bystander sensor, 양 arm 공통 | Camera가 본 double mismatch, 양 arm 공통 |
|---|---:|---:|---:|
| A · physical persistence | 17.77% → 13.07% | 17.89% | 14.56% |
| A · common-mode error | 17.77% → 13.07% | 17.89% | 9.72% |
| B · physical persistence | 14.50% → 9.79% | 14.70% | 10.43% |
| B · common-mode error | 14.50% → 9.79% | 14.70% | 9.72% |

각 물체의 marginal 곱을 빼도 reset은 선택한 센서의 within-session covariance를 A −4.703 pp [−4.816, −4.595], B −4.700 pp [−4.812, −4.597] 줄였다. 이 감소는 두 원인에서 동일했다. Bystander는 부호 한 개도 바뀌지 않았다. 따라서 local latch가 실제로 해제됐다는 검산에는 충분하지만 upstream에서 온 반복의 원인을 알려주지는 않는다. 물체 순서에 따른 camera double mismatch 차이의 모든 95% CI는 0을 포함했다. [원인·위치·순서별 집계](diagnostics59.json)

**원인과 연결된 변화는 첫째 물체의 camera와 둘째 물체의 reset sensor를 교차했을 때 나타났다.** 이 두 mismatch bit의 covariance가 reset 후 얼마나 변하는지 측정했다. Physical A에서는 +1.329 pp [+1.222, +1.441], physical B에서는 +0.177 pp [+0.080, +0.274]였다. Common-mode A에서는 +0.048 pp [−0.062, +0.153], B는 +0.002 pp [−0.098, +0.101]였다. Physical 원인이면 둘째 전자 센서가 첫 물체의 물리 상태라는 공통 단서에 더 가까워진다. Common-mode 원인이면 첫 물체의 독립 camera는 공유 전자 오류를 읽지 않는다. 첫째 물체는 애초 reset을 받지 않으므로 반대 방향의 cross-sensor contrast에는 reset 변화가 없었다.

Camera만으로 두 물체 간 true mismatch covariance도 추정했다. Camera의 개별 net sign error d=2×.03×.97에 따른 attenuation (1−2d)²를 제거하면, physical A에서 0.06190 [0.05926, 0.06461], B에서 0.00898 [0.00675, 0.01133]이었다. 설정한 값은 0.06090과 0.00840이다. Common-mode에서는 두 환경에서 같은 true 물리 draw를 공유했고 추정값은 0.00017 [−0.00196, 0.00232]였다. 이 정밀도는 24개 catalog 전체를 모은 추정치에 해당한다. [관측 기반 원인 분해](diagnostics59.json)

## 한 surface에서의 원인 판별력

| 4×2 pilot | 원인 balanced accuracy [95% CI] | A에서의 원인 판별률 | B에서의 원인 판별률 |
|---|---:|---:|---:|
| No reset, 전자 센서만 | 50.000% [50.000, 50.000] | 50.00% | 50.00% |
| Reset, 전자 센서만 | 50.000% [50.000, 50.000] | 50.00% | 50.00% |
| No reset + camera | 55.168% [55.076, 55.260] | 59.01% | 51.33% |
| Reset + camera | 55.813% [55.723, 55.906] | 60.14% | 51.49% |

Camera가 있는 상태에서 reset의 incremental gain은 **+0.645 pp [+0.603, +0.683]**였다. 두 원인과 A/B의 prior는 균등하고 개별 episode의 실제 이름은 숨겼다. Camera 없이 발생한 모든 tie는 C로, 나머지는 posterior MAP로 분류했다. Reset + camera는 실제 P의 76.80%를 P로 고르고 실제 C의 65.18%도 P로 잘못 골랐다. B에서 공동 오류가 rare하므로 여덟 번 만으로 원인을 강하게 구별하기 어렵다. 구간은 24개 catalog를 5,000회 paired bootstrap한 pointwise 95% CI다. [원인별 오판과 CI](diagnostics59.json)

**무엇이 camera 이득을 설명하는가.** 사후 대조에서 camera 단독의 원인 판별률은 51.497%였다. Reset + camera에서 세 instrument의 동시 판독은 허용하되 물체 간 joint likelihood를 두 물체의 marginal likelihood의 곱으로 바꾸면 원인 판별률은 55.454%였다. 물체 간 dependence까지 모델링할 때의 추가 이득은 +0.359 pp [+0.324, +0.398]였다. 같은 contact에서의 전자·camera disagreement만으로도 원인 판별력의 상당 부분이 유지됐다. 반면 A/B 정책 판별률은 같은 대조에서 54.072%에서 55.227%로 +1.156 pp [+1.087, +1.221] 올랐다. 같은 regime·material·원인 안에서 두 번째 물체를 다른 pilot의 것으로 바꾸어 pairing을 깨면 full likelihood의 해당 성능은 원인 54.762%, A/B 52.867%였다. 따라서 camera를 쓴 개별 pilot의 판별 향상 전부를 session persistence 측정에 귀속하기는 어렵다. [대조의 설계](cues59.py) · [결과](cues59.json)

**Camera가 같은 오류를 물려받는 경우.** 유료 pilot 중 0%, 25%, 50%, 100%의 camera가 독립적인 물리 source 대신 두 전자 센서와 같은 upstream을 읽도록 하고 원래 likelihood를 고정했다. Sensor 고유 noise와 occlusion은 그대로 유지했다. Reset + camera 원인 판별률은 각각 55.813%, 54.360%, 52.906%, 50.000%였다. 100%일 때 실제 원인이 달라도 세 채널의 기록은 두 원인에서 모두 byte 단위로 같았다. 이 모형에서 reset의 해제 범위 및 camera 장착·clock·동기화의 독립성은 실물 검증 대상이다. [Sensitivity와 음성 대조](diagnostics59.json)

## 재현과 남은 제약

새 파일은 서버 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/sensor_reset_v59`에 있다. 독립적으로 구현된 selector가 24개 raw archive의 부호에서 복원한 code와 policy용 compact archive의 code가 모든 cell에서 같았다. Audit 저장소의 latent true signs는 downstream selector에 주지 않았다. Simulator의 물리 순서에는 wear나 drift를 유발하는 추가 carryover가 없고 camera analytic 모형에서는 물리 상태와 무관하게 visibility를 뽑는다. [Pilot 명세](PILOT_PLAN_KO.md) · [Raw·compact archive digest](diagnostics59.json)
