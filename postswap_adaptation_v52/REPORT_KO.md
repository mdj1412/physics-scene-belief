# v52 · 표면 교체 뒤 기존 calibration을 언제 유지하고, 다시 추정하고, 포기할 것인가?

**새 표면에서 weak/reference를 네 쌍 측정해 세 action을 선택하자, 같은 네 쌍과 같은 action을 받은 simple sign vote보다 평균 held-out Brier loss와 session loss p90이 모두 낮아졌다.** 평균 Brier는 **0.158520 대 0.158552**, session당 Brier의 p90은 **0.182592 대 0.182656**이었다. 두 차이는 각각 약 0.000033과 0.000065다. 더 단순한 post-only vote와 비교할 때보다 우위가 작다.

**세 action 중 하나를 반드시 고르는 방식이 가장 좋은 정책은 아니었다.** 교체 전후의 증거를 joint posterior로 연속 혼합하면 평균 Brier는 0.158526으로 selector와 차이를 확정하기 어려웠고, session loss p90은 **0.182367**로 더 낮았다. 논문에서 주장할 수 있는 범위는 적은 probe로 교체 전 증거를 어느 정도 재사용할 수 있으며, 그 선택이 평균과 상위 손실에 서로 다르게 작용한다는 것이다.

## 1. 실제 deployment에서 정책이 볼 수 있는 것

1. **교체 전 네 쌍은 공통 관측이다.** 서로 다른 calibration 물체 네 개를 각각 0.4 N으로 누른 뒤 reset하고, 같은 위치의 2.0 N reference displacement를 읽은 뒤 다시 reset한다. 두 displacement의 부호가 맞을 확률을 교체 전 κ라고 둔다.
2. **같은 contact law에서 표면을 바꾼다.** Mosaic에서 nap으로, nap에서 mosaic으로 교체한다. 교체 후 새 표면의 calibration 물체 0·1·2·4·8개에서 같은 쌍을 얻는다. 교체 전후의 서로 다른 surface 행을 사용하고, calibration repetition을 별도 순열로 짝지었다. Deployment 물체와 calibration 물체는 서로 다른 support에서 생성했다. V50에서 보관한 12개 catalog와 6,144개 held-out deployment 물체를 재사용하여 물체 차이를 통제했다. 실제 장비로 새 재료를 측정하지는 않았다.
3. **정책이 선택하는 대상은 최종 보정 방식이다.** Weak readout d를 두 방향으로 해석한 frozen decoder 출력 p(d), p(−d)를 각각 얻는다. Keep은 교체 전 네 쌍의 κ̂로, refit은 교체 후 쌍의 κ̂로 이 두 확률을 혼합한다. Unsigned는 source에서 별도로 학습한 |d| 기반 predictor다. 세 경우 모두 배포 물체당 query-local weak preload 한 번과 reset 한 번을 쓴다.
4. **관측 가능한 drift와 보이지 않는 오류를 구별했다.** 교체 후 session 전체의 유지율이 κ에서 1−κ로 바뀔 확률 g를 0·0.25·0.5·1로, active 물체의 reference 부호가 실제 모사 impact에서 뒤집힐 확률 e를 0·0.25·0.5로 바꿨다. 교체 전 κ는 0·0.25·0.5·0.75·1을 균등 가중했다. 네 쌍은 weak→reference 관계를 알려주지만, 정책은 direct impact 관측도 e·g·true event도 받지 않는다. Impact 결과는 signed contour를 입력한 PyBullet에서 계산해 둔 binary stick/slip이다. [사전 계획](PLAN_KO.md) · [Calibration과 정책](POLICY_KO.md)

## 2. 같은 예산의 sign vote와 무엇이 다른가?

Likelihood selector는 측정 크기와 부호를 사용하고, vote selector는 같은 dead zone을 넘은 쌍의 부호 일치와 반전 횟수만 사용한다. 두 정책은 동일한 교체 전후 관측, 같은 drift prior 0.25, 같은 reference–impact 오류 가정 0.10, 같은 세 action을 쓴다. 두 정책 모두 교체 전후의 측정으로 교체 후 유지율의 joint posterior를 계산하고 source에서 정한 expected Brier가 최소인 action을 고른다. Posterior drift가 50%를 넘는지 여부는 진단용 cutoff이며 action을 정하는 규칙은 아니다.

**Post-only vote**는 옛 측정을 무시하고 새 측정에서 posterior mean (일치 수+1)/(유효 쌍+2)를 구해 두 방향의 예측을 바로 섞는다. **Continuous joint**는 likelihood selector와 같은 pre/post posterior를 추정하고, 그 posterior의 impact 방향 확률을 직접 혼합계수로 사용한다. 이 비교로 부호의 크기를 사용하는 이득과 최종 출력을 세 action으로 제한하는 효과를 구별했다. [정책 및 source risk](policy52.py) · [Frozen fit](source_risk52.json)

## 3. Drift를 감지한 정도와 선택한 행동

새 측정 0·1·2·4·8쌍에서 likelihood의 drift AUROC는 각각 **0.500, 0.687, 0.742, 0.779, 0.803**이었다. 네 쌍의 같은 조건에서 vote는 0.774였다. 네 쌍의 likelihood는 P(drift|관측)>0.5 기준으로 false-positive rate 2.92%, true-positive rate 33.20%였다. 실제 drift의 다수가 높은 posterior 확률로 검출되지는 않았다.

| 교체 후 쌍 | Keep | Refit | Unsigned |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0% | 100.0% |
| 1 | 18.8% | 7.5% | 73.7% |
| 2 | 22.9% | 19.9% | 57.2% |
| 4 | 17.9% | 43.0% | 39.1% |
| 8 | 6.7% | 64.6% | 28.8% |

각 행은 사전 지정한 κ와 네 g의 균등 mixture에서 likelihood selector의 선택률이다. κ=1에서 교체 뒤 방향 관계가 뒤집힌 경우, 네 쌍으로 keep/refit/unsigned를 고른 비율은 0.03%/70.57%/29.39%였다. 반면 κ=.5이면 유지와 반전의 관측분포가 같아 AUROC는 정확히 .5였다. 이때 unsigned 선택률도 두 event에서 동일했다. [진단 수치와 조건별 선택률](calibration52.json)

## 4. 평균 held-out Brier와 session loss p90

Primary budget은 교체 뒤 네 쌍이다. 각 session은 같은 환경의 서로 다른 물체 128개에 weak preload를 하나씩 적용한다. **Session loss p90은 그 128개 squared error 평균의 90번째 백분위수**다. 물체별 loss의 p90, latency p90과 단위가 다르다. 표의 p90 차이는 두 policy의 marginal p90의 차이다.

| 정책 · 교체 후 K=4 | 평균 Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|
| Likelihood selector · keep/refit/unsigned | **0.158520** | **0.182592** |
| 같은 세 action의 vote selector | 0.158552 | 0.182656 |
| Post-only vote soft | 0.158656 | 0.182722 |
| Always keep · 교체 전 likelihood | 0.161021 | 0.185828 |
| Always refit · 교체 후 likelihood | 0.158623 | 0.182643 |
| Always unsigned | 0.159297 | 0.182935 |
| Continuous joint likelihood | 0.158526 | **0.182367** |

| 사전 지정 대조 · likelihood selector에서 대조를 뺀 값 | 평균 Brier 차이 [simultaneous 95% CI] | Session loss p90 차이 [simultaneous 95% CI] |
|---|---:|---:|
| 같은 세 action의 vote selector | −0.0000326 [−0.0000384, −0.0000268] | −0.0000647 [−0.0001202, −0.0000321] |
| Post-only vote soft | −0.0001365 [−0.0001480, −0.0001249] | −0.0001308 [−0.0002350, −0.0000762] |

12개 catalog 모두에서 두 대조 대비 평균 손실이 낮았다. 평균은 각 환경·물체에서 물리 조건 및 session drift의 조건부 기댓값을 계산하고, 독립적으로 짝지은 256개 calibration 반복을 평균했다. Session p90은 하나의 calibration prefix를 128개 물체가 공유하고 각 물체마다 하나의 binary outcome을 뽑아 계산한 session 평균에서 추정했다. 각 drift 조건의 shared event를 먼저 분리해 구한 뒤 g로 가중했다. 3개 e, 4개 g, 5개 κ를 동일 가중했다. 두 대조 × 두 endpoint를 Bonferroni 보정했다. 평균에는 12개 catalog의 paired t interval, p90에는 같은 catalog를 통째로 뽑는 5,000회 paired bootstrap을 썼다. Calibration 반복을 별개 재질 수로 세지 않았다. [Primary와 전체 조건](score52.json) · [Catalog별 기대 loss](catalog52.npz) · [Session별 rollout](session52.npz)

**연속 혼합은 상위 손실에서 더 강했다.** Likelihood selector − continuous joint의 평균 차이는 −0.0000068 [−0.0000247, +0.0000111], session p90 차이는 **+0.0002244 [+0.0001247, +0.0002936]**이었다. 네 쌍에서 action 선택이 두 vote보다 낫지만 낮은 tail loss를 목표로 할 때에는 보정 방향의 불확실성을 연속적으로 유지하는 대안이 남는다. [강한 대안 검산](audit_score52.json)

## 5. 몇 쌍을 살 것이며 언제 실패하는가?

| 교체 후 쌍 | Selector 평균 Brier | Vote selector 평균 Brier | Post-only vote 평균 Brier | Selector session p90 | Vote selector session p90 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.159297 | 0.159297 | 0.159637 | 0.182935 | 0.182935 |
| 1 | 0.159086 | 0.159106 | 0.159211 | 0.182805 | 0.182831 |
| 2 | 0.158835 | 0.158863 | 0.158938 | 0.182704 | 0.182742 |
| 4 | 0.158520 | 0.158552 | 0.158656 | **0.182592** | 0.182656 |
| 8 | **0.158333** | 0.158356 | 0.158465 | 0.182703 | 0.182711 |

![추가 calibration 쌍에 따른 평균 Brier 및 session loss p90](budget52.png)

- **평균과 tail의 budget은 달랐다.** 네 쌍에서 여덟 쌍으로 늘리면 selector의 평균 Brier는 0.000187 줄었지만 session p90은 0.000111 높아졌다. 이는 시험한 budget의 점추정이며 네 쌍이 보편적인 최적값임을 뜻하지 않는다.
- **현재 surface의 부호 관계가 완전히 달라졌다면 새 측정만 쓸 수 있다.** e=0, g=1에서 selector는 post-only vote보다 평균 Brier가 0.000656 높았다. 모든 표면이 변했는데 source prior는 25%만 변한다고 가정했기 때문이다. e=g=0에서는 기존 측정의 재사용으로 오히려 0.000542 낮았다. 두 값은 사전 지정된 평균 혼합과 별개의 조건별 결과다.
- **Reference 자체가 무작위가 되면 unsigned가 낫다.** e=.5에서 selector는 모든 g에서 unsigned보다 0.001577–0.002008 높은 Brier를 보였다. 마찬가지로 교체 전 κ=.5에서 grid 평균 Brier는 selector 0.160173, unsigned 0.159297이었다. 네 쌍의 weak/reference로는 downstream impact에서 부호가 뒤집히는 정도까지 식별할 수 없다.
- **검산.** 두 surface 모두에서 selector의 평균이 두 vote보다 낮았다. PyBullet의 time step과 mesh를 절반으로 줄이고 다시 채점해도 네 쌍의 평균 Brier는 selector 0.164201, vote selector 0.164237, post-only vote 0.164268이었다. 같은 물체에서 deployment noise와 실제 binary 충돌 draw만 독립적으로 다시 뽑으면 session p90은 각각 0.182690, 0.182737, 0.182811이었다. 이 replay에서도 continuous joint의 p90이 0.182485로 더 낮았다. [독립 deployment replay](tail_repeat52.json)
- **한 번의 배포에서 작은 차이가 재현되는가.** 독립 session 48개씩인 유한 배포를 같은 catalog에서 256회 실현했다. 한 배포의 평균에서 selector가 같은 세 action의 vote보다 나빴던 비율은 39.45%, post-only vote보다 나빴던 비율은 37.11%였다. 각 배포에서 계산한 sample p90 기준으로는 각각 24.22%, 42.58%였다. 이는 같은 환경군에 조건부인 Monte Carlo 결과이며 실제 운영 실패율의 추정값은 아니다. [유한 배포 검산](audit_score52.json)

## 6. GPU latency와 sensing·reset 비용

512개 물체를 네 환경에 나눠 각 환경의 pre/post 네 쌍에서 action을 새로 계산하고, factual acquisition·구매한 sensor·original decoder 1회·선택한 보정까지 같은 RTX PRO 6000 Blackwell에서 실행했다. 12개 catalog에 두 번의 Williams crossover를 적용해 정책당 96개 batch를 쟀다.

| 정책 | GPU p90, ms/512 objects [pointwise 95% CI] |
|---|---:|
| Likelihood selector | 468.12 [467.70, 473.56] |
| 같은 세 action의 vote selector | 467.28 [467.01, 467.75] |
| Post-only vote | 467.49 [466.80, 469.65] |
| Always unsigned | 467.50 [466.85, 469.48] |

Selector−vote selector는 **+0.84 ms [+0.06, +9.55]**, selector−post-only vote는 **+0.63 ms [−2.20, +9.87]**였다(두 개의 paired timing 비교에 대한 simultaneous 95% CI). Likelihood selector가 GPU latency를 줄인다는 근거는 없었다. 네 환경의 estimator만 분리한 K=4 host p90은 likelihood, vote selector, post-only vote가 각각 0.1017, 0.0607, 0.0132 ms였다. 전체 시간은 로봇 움직임을 제외한 synchronized wall time이다. [GPU timing 결과](timing52.json)

기존 네 쌍에 8 contacts·8 resets를 이미 지불했다. 교체 뒤 네 쌍에는 8 contacts·8 resets가 추가되고, 이후 각 deployment 물체에는 query-local weak preload와 reset이 하나씩 든다. **Keep이나 unsigned를 골라도 post-swap 측정을 이미 했다면 비용을 되돌릴 수 없다.** Always-unsigned는 추가 쌍을 실제로 사지 않아도 같은 예측을 내며, post-only vote는 기존 네 쌍 없이도 운용할 수 있다.

접근 100 ms, 측정 50 ms, 후퇴 100 ms, 안정화 40 ms, loading 4 N/s, unloading 8 N/s를 가정하면 weak 접촉은 440 ms, reference 접촉은 1,040 ms다. 교체 뒤 네 쌍은 5.92초/session, 128개 배포 물체에 재사용하면 **추가 46.25 ms/object**다. 물체별 deployment touch까지 합하면 486.25 ms/object다. 교체 전 네 쌍부터 새로 취득하는 전체 cycle이면 532.5 ms/object다. 평균 Brier가 unsigned보다 낮아진 0.000777을 0.04625초로 나누면 추가 1초당 약 0.0168 Brier-equivalent가 손익분기점이다. 로봇 이동, 표면 교체, reset 실패, 잔류 변형은 이 가정에 포함하지 않았고 장비 시간은 실측하지 않았다. [GPU 측정과 물리 비용](GPU_KO.md)

## 7. 논문 기여와 다음에 판별할 것

이 결과는 **표면 교체 후 네 쌍의 관측으로 여러 물체가 공유하는 방향 변화를 일부 탐지하고, 예측이 불안한 session에서는 unsigned로 전환할 수 있다**는 제한된 근거다. 같은 예산의 post-only vote보다 개선됐고, 같은 세 action을 가진 vote로 대조해도 작은 차이가 남았다. 표면·물체가 바뀔 수 있는 데이터에서 평균과 session 상위 손실을 동시에 보고한 점, 소수의 잘못된 보정이 하나의 session에 함께 전파되는 현상을 구분한 점이 논문 기여 후보다.

가장 강한 대안 설명은 표면이 바뀌었는지 정확히 판정한 것보다 **방향의 불확실성을 줄이는 shrinkage와 unsigned fallback이 손실을 낮춘 것**이다. 단순 vote와의 작은 차이, continuous joint의 더 낮은 p90, κ=.5에서 식별 불가능한 drift와 e=.5에서의 실패가 이 설명을 지지한다. 두 합성 surface는 알려진 complement mechanism을 공유하며, 정책은 e=.10과 g=.25를 source prior로 썼다. 완전히 새로운 재료·날짜의 임의 drift나 누적 wear에서 같은 이득을 보장할 수 없다.

다음 확인 실험에서는 교체 전후 0.4 N·2.0 N displacement와 별도의 high-speed impact 영상으로 읽은 변형 방향을 같은 장비에서 측정해야 한다. 입자별 slip 결과와 별도로 session 간 residual strain·reset 시간·관측 순서를 기록하고, surface specimen·날짜 단위 holdout에서 vote selector와 continuous joint를 함께 사전 고정하는 편이 이 연구의 남은 질문을 잘 구분한다. [원논문 비교·지식 지도·연구 공백과 후속 질문](LITERATURE_KO.md)

## 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/postswap_adaptation_v52/`다. 멀티 에이전트가 source 정책과 calibration, independent scoring·session rollout, GPU timing을 분담했다. 정책 학습과 평가에는 Slurm compute node를 사용했고, v50 원본은 읽기 전용 의존성이다. `physics_risk52.py`가 primary·ultra 조건부 risk와 binary rollout을 만들고, `score52.py`가 동결된 결정과 결합해 사전 지정 endpoint를 계산한다. [Risk 생성 코드](physics_risk52.py) · [Paired 채점 코드](score52.py) · [Estimator와 action](policy52.py) · [독립 재실행 코드](tail_repeat52.py)
