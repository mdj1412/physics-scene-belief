# v53 · 마지막 reference를 impact 방향 측정으로 바꿀 가치가 있는가?

**네 번째 calibration 물체에서 impact 방향을 읽으면 같은 관측을 받은 post-only simple sign vote보다 continuous joint mixture의 평균 held-out Brier와 session loss p90이 모두 낮았다.** 그러나 교체 전 기록까지 활용하는 v52의 세 행동을 같은 새 관측에 적용하면 결론이 갈렸다. 기본 solver에서 세 행동의 likelihood selector가 **평균 Brier 0.158458**로 continuous joint의 **0.158519**보다 좋았다. 반면 continuous joint의 **session p90은 0.182310**으로 같은 selector의 **0.182443**보다 낮았다. 문자 그대로의 v52 원본(네 쌍 모두 reference)과 비교해도 평균 차이는 −0.0000006으로 불확실했다. 따라서 연구 질문의 두 지표가 강한 대조군 모두에서 동시에 개선됐다는 가설은 지지되지 않았다.

## 1. 무엇을 바꾸고 무엇을 고정했는가

Calibration 물체 한 개에서 0.4 N으로 측정한 **weak displacement의 방향이 후속 impact에서 유지되는가**가 관심 대상이다. 그 방향을 알면 다른 배포 물체에서 받은 weak displacement를 어느 쪽으로 보정할지 결정할 수 있다. 물체마다 영향이 달라서 방향을 알고도 특정 배포 물체의 stick/slip 정답이 주어지지는 않는다.

| 단계 | 네 reference 쌍 · 4R | 세 reference 쌍과 한 impact 쌍 · 3R+I |
|---|---|---|
| 교체 전 | 네 weak/reference 쌍 | 같은 측정 |
| 교체 직후 물체 1–3 | 0.4 N weak + 2.0 N reference | 같은 물체와 같은 측정 |
| 교체 직후 물체 4 | 0.4 N weak + 2.0 N reference | 같은 0.4 N weak + impact 순간 변형 방향 |
| 배포 | 별도 128개 물체에서 각각 query-local weak preload 한 번 후 stick probability 예측 | 같은 물체, 관측, decoder와 outcome |
| 교체 후 예산 | 8 contacts, 8 resets | 8 contacts, 8 resets |

첫 번째 독립 변수는 **어떤 측정을 구매했는가**다. 두 번째는 **구매한 측정을 예측에 어떻게 반영하는가**다. Continuous joint는 보정 방향을 유지한 예측과 뒤집은 예측을 posterior mean으로 혼합한다. 같은 정보에 맞춘 v52 selector는 교체 전 보정 유지, 교체 후 보정 재추정, unsigned 중에서 source의 Brier risk로 선택한다. 같은 세 행동의 vote는 displacement 크기를 쓰지 않고 부호 일치만 센다. Post-only simple sign vote는 교체 후 부호만 세어 방향 혼합 비율을 정한다. 원래의 v52 정책도 4R 그대로 재생했다.

혼합 조건의 첫 번째 실험에서 impact 방향은 **simulator가 지정한 impact 상태에서 두 지점의 변형 차이를 읽은 이상적 센서값**이다. 네 번째 reference는 정책에서 가렸다. 독립적인 calibration 물체를 사용했으며 그 표면의 이름, 실제 reversal rate, deployment stick/slip은 입력에서 제외했다. 정책별 산출을 검산했고 사용하지 말아야 할 네 번째 reference 값을 교란해도 mixed policy 출력이 같았다. 표면·물체 support는 v50–52와 공유하므로 이 연구에서 held-out은 기존 모델의 training/deployment 분리를 뜻한다. [실험 계획](PLAN_KO.md) · [통계 계획](SCORE_PLAN_KO.md) · [정책 검산](POLICY_KO.md)

## 2. 같은 관측의 정책 비교와 같은 정책의 관측 비교

아래 그림의 점은 3R+I continuous joint의 loss에서 각 대조의 loss를 뺀 값이다. 왼쪽이 continuous joint에 유리하다. 위 두 행은 두 endpoint와 두 comparator를 함께 보정한 95% interval, 나머지는 탐색적인 개별 95% interval이다. 같은 정보에서 selector를 이기는지, 같은 decoder에서 네 번째 물리 관측을 바꿀 가치가 있는지는 다른 비교다.

![정책 변화와 네 번째 관측 교환에 따른 Brier 차이](comparisons53.png)

**Impact 방향의 추가 가치와 교환 가치가 달랐다.** 같은 세 reference에 네 번째 impact의 부호를 가린 조건보다 평균 Brier가 **0.0001620** 낮아졌다. 그러나 네 번째 reference를 포기하는 비용까지 포함하면 4R continuous 대비 개선은 **0.0000074**였다(개별 95% CI −0.0000125 ~ −0.0000023). 같은 비교의 session p90 차이는 **−0.0000568**(−0.0000851 ~ −0.0000028)이었다. 숨겨 둔 impact sign을 보여 주는 비교만으로는 센서 교체의 이점을 크게 추정하게 된다.

**어느 환경에서 이득이 났는가.** 아래는 drift rate와 초기 retention을 평균한 기본 solver의 held-out Brier다. Strong vote와 selector보다 좋아지는지에 앞서, 방향 정보를 쓰는 것 자체가 이로운 환경인지가 중요하다.

| Reference와 impact의 active sign 불일치율 e | 3R+I continuous | 4R continuous | 3R+I selector | Unsigned |
|---|---:|---:|---:|---:|
| 0 | 0.156213 | 0.156169 | 0.155954 | 0.159297 |
| 0.25 | 0.158667 | 0.158526 | 0.158654 | 0.159297 |
| 0.50 | 0.160678 | 0.160883 | 0.160766 | **0.159297** |

마지막 행에서는 impact 관측으로 바꾸자 잘못된 방향 보정의 손실을 일부 줄였지만 unsigned가 여전히 더 좋았다. 또 e의 prior 평균을 v52에서 가져온 0.10에서 평가 혼합의 평균인 0.25로 바꾸면, 3R+I가 4R보다 **평균 Brier 0.0000210 높아졌다**. Prior의 평균만 일치시킨 민감도 검사이며 true e를 제공한 대조가 아니다. [채점 결과](score53.json)

## 3. 한 번 본 impact로 무엇을 식별했는가

Weak→reference 방향 유지율을 k, reference→impact 반전 확률을 e라 하면 weak→impact 유지율은 q = e + (1−2e)k다. Fourth witness 하나에서 weak와 impact만 읽으면 q에 대해 배울 수 있지만 k와 e를 서로 분리하기 어렵다. 네 번째 물체의 **49.23%**에서 변형 방향이 0이 아니었다. 나머지에서는 방향에 관한 증거를 얻지 못했다. 실제 e를 0, 0.25, 0.50으로 바꿔도 posterior mean e의 평균은 **0.0994, 0.1002, 0.1010**이었다. 반면 q의 posterior mean은 impact sign을 가렸을 때보다 절대값으로 평균 0.0300, 0.0327, 0.0353만큼 움직였다. 한 번의 측정으로 예측을 조정하는 것과 표면의 error rate를 추정하는 일 사이의 격차다. [Calibration 결과](calibration53.json) · [식별성 검산](identifiability53.json)

### 이상적인 부호를 영상으로 읽으면

네 번째 calibration 물체의 family·높이·충돌 상태에 맞춰 **충돌 전/충돌 순간 × 두 marker × 49 grayscale pixels**를 별도로 렌더링했다. 두 marker는 접촉 중심으로부터 각각 1.5 mm 떨어져 있다. 각 영상의 intensity centroid로 두 marker의 움직임 차이를 읽고 ±0.15 mm 안은 무응답으로 처리했다. 광학 배율은 0.1 mm/pixel, 두 시점에 공통인 translation과 노이즈 모델은 모든 정책 비교에서 공유했다. 아래 셋은 동일한 continuous posterior를 사용한다.

| Impact readout | 평균 held-out Brier | Session loss p90 |
|---|---:|---:|
| 이상적 profile sign | 0.158519 | 0.182310 |
| Marker jitter 0.20 pixel | 0.158519 | 0.182310 |
| Marker jitter 0.85 pixel | 0.158558 | 0.182360 |

0.85-pixel 조건에서는 실제 inactive인 관측 중 약 **37.4%**에 방향이 잘못 검출됐다. 그 결과 평균 손실이 원래의 4R continuous(0.158526)보다 높아졌다. 이 sensor는 지정된 충돌 순간과 보이는 marker를 가정한다. 실제 영상에서의 onset 검출, 가림, 재료 내부의 변형까지 해결한 실험은 아니다. 표의 confusion count는 같은 물체·노이즈를 여러 환경 조건에 재사용한 simulation count다. 독립 물체 수와 다르다. 네 session의 synthetic line-scan 판독은 별도 CPU timing에서 p90 **0.0355 ms**였다. [픽셀 판독 코드](optical53.py) · [측정 오차와 손실](optical53.json) · [별도 readout timing](scan_latency53.json)

