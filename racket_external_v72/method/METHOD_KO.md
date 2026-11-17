# v72 · 동일한 4·8회의 calibration으로 고른 두 routing의 비교

## 관측량과 결정

$v_x$와 $v_z$는 racket 좌표계에서 부호가 있는 planar velocity(m/s), $ω_y$는 spin(rad/s)이다. Source model은 입사 상태에서 이탈 상대속도 $s'=v'_x-0.020ω'_y$의 부호를 예측해 두 affine expert 중 하나를 고른다. 두 gate는 모두 이 routing을 통째로 뒤집거나 유지하고 expert 계수는 고정한다. Majority vote는 관측한 $s'$ 부호와 기존 routing의 불일치가 과반수이면 뒤집는다. Continuous-loss gate는 동일한 calibration shots에서 original/flip 각각의 연속 오차 $L=‖v'_{xz}-\hat v'_{xz}‖_2+0.020|ω'_y-\hat ω'_y|$를 평균해 더 낮은 routing을 사용한다. Spin 항에 곱한 ball radius가 두 항의 단위를 m/s로 맞춘다. 두 gate 모두 동률이면 original을 유지한다. [고정한 protocol](PROTOCOL_KO.md)

두 continuous alternative는 동일한 calibration labels로 크기까지 보정한다. Single affine EB는 source 제품별 affine coefficient의 공동 변화를 prior로 받아 표적 제품의 계수를 갱신한다. V71에서 여섯 source-only inner 선택이 일치한 0.1의 prior precision multiplier를 고정했다. Elastic impulse는 thin-shell moment와 tangential impulse로 translation과 rotation을 연결하고 normal restitution 및 tangential response의 intercept·slope를 각각 갱신한다. V71의 실험 코드를 재사용했다. [평가 구현](bench.py)

**분리한 세 가지 sign 지표.** (1) 제품에서 기존 source routing의 부호가 held test에서 과반수로 잘못되는지를 외부 평가자가 정의하고, 두 gate가 그 방향을 골랐는지 본다. (2) 개별 impact에서 실제로 선택한 route의 명목상 sign이 실측 $s'$와 맞는지 본다. (3) 모든 방법에 대해 최종 continuous 출력으로 다시 계산한 $\hat s'=\hat v'_x-0.020\hat ω'_y$의 부호를 실측과 비교한다. Expert는 constrained regression을 쓰지 않아 출력이 명목상의 분기를 벗어날 수 있다. 연속 모델 두 개는 명시적인 product-level flip action을 내지 않는다.

## Retrospective 결과

이 표의 여섯 제품은 v71에 포함됐다. 제품 전체를 source fitting에서 제외하고 해당 제품의 공식 train에서 calibration 4회·8회를 선택했다. 최종 held test는 공식 split의 총 1,623행이다. 제품별 256개 순서에서 동일한 prefix를 네 방법에 제공했다. Draw들은 같은 물리적 impact를 재사용한다.

| K | 방법 | 제품 macro velocity MAE (m/s) | spin MAE (rad/s) | 예측 출력 sign accuracy | 제품 방향 판별률 | held loss가 작은 route 선택률 |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Majority vote | 0.886 | 51.52 | 82.31% | 94.53% | 87.89% |
| 4 | Continuous-loss gate | 0.812 | 44.56 | 73.42% | 76.11% | 92.77% |
| 4 | Single affine EB | 0.373 | 19.53 | 86.97% | 해당 없음 | 해당 없음 |
| 4 | Elastic impulse | 0.352 | 17.93 | 88.07% | 해당 없음 | 해당 없음 |
| 8 | Majority vote | 0.878 | 50.76 | 82.30% | 94.14% | 89.06% |
| 8 | Continuous-loss gate | 0.812 | 44.54 | 74.22% | 77.08% | 93.75% |
| 8 | Single affine EB | 0.348 | 18.15 | 87.35% | 해당 없음 | 해당 없음 |
| 8 | Elastic impulse | 0.300 | 15.06 | 90.03% | 해당 없음 | 해당 없음 |

제품 방향 판별률은 모든 제품이 같은 가중치를 갖고, release 방향의 strict-majority held label과 gate의 flip 행동이 맞는 비율이다. Positive product는 Grass D.TecS 하나이고 네 보통 제품과 A-B-S II soft는 negative로 분류됐다. 특히 A-B-S II soft의 불일치는 76/158이므로 장기 모집단에서 방향이 확정됐다고 볼 수 없다. Held loss 기준으로는 Grass와 A-B-S II soft 모두 반전하는 것이 낫다.

| K | Gate | Grass flip 빈도 | Grass route sign / output sign | A-B-S II soft flip 빈도 | A-B-S II soft route sign / output sign |
|---:|---|---:|---:|---:|---:|
| 4 | Majority vote | 97.27% | 88.41% / 61.74% | 30.08% | 50.76% / 43.52% |
| 4 | Continuous-loss gate | 100.00% | 90.64% / 63.21% | 100.00% | 48.10% / 24.05% |
| 8 | Majority vote | 99.61% | 90.32% / 63.00% | 34.77% | 50.58% / 42.22% |
| 8 | Continuous-loss gate | 100.00% | 90.64% / 63.21% | 100.00% | 48.10% / 24.05% |

Grass D.TecS에서 continuous-loss gate는 두 예산에서 모두 256/256번 flip을 선택했다. 8회에서 release route의 부호는 90.64% 맞았지만 선택된 expert가 실제로 예측한 $\hat s'$ 부호는 63.21%만 맞았다. 같은 조건의 elastic impulse는 90.26%의 output sign accuracy와 0.397 m/s · 21.97 rad/s를 달성했다. Continuous-loss gate의 오차는 0.668 m/s · 40.58 rad/s였다.

A-B-S II soft에서 continuous-loss gate가 8회 모두에 대해 반전을 택한 결과, majority의 1.920 m/s · 95.30 rad/s에서 1.517 m/s · 57.42 rad/s로 연속 오차가 줄었다. 그러나 output sign accuracy도 42.22%에서 24.05%로 줄었다. Diamant에서는 두 routing의 held 차이가 0.033 m/s · 2.25 rad/s에 불과하고 original이 낮다. 8회에서 continuous-loss gate는 37.5%의 draw에서 반전을 선택했다. 이러한 변화는 표적 선택이 동일한 데이터와 예산에서도 관측량에 따라 달라진다는 사실을 보여준다. [제품별 결과](retrospective/summary.json)

## GPU 검산과 external 실행

기존 입력 CSV의 hash, node56의 RTX PRO 6000 Blackwell과 할당 번호, 시작 당시 0 MiB·0% 및 compute process 없음 검사, held reference와 calibration pool 크기를 보존했다. K=4·8의 모든 product×draw에서 v71과 calibration 행·순서가 동일했다. Majority·single affine EB·elastic impulse를 재계산해 9,216개 product·draw·method·budget의 velocity·spin이 각각 4.22×10⁻¹⁵ m/s, 2.35×10⁻¹³ rad/s 이내에서 v71의 기록과 맞았다. [실행 기록](retrospective/run.json) · [검산 코드](audit.py) · [각 draw 결과](retrospective/draws.csv)

External mode는 기존 여섯 제품의 공식 train 전부로 source model을 fit한 뒤 새 target product에는 calibration.csv에서 공개된 labels만 이용한다. test.csv는 평가에 사용한다. 두 파일에 동일한 impact ID가 있거나 calibration이 8행보다 적으면 중단한다. 축·부호·단위·제품 및 실험 session provenance는 원자료에 따라 먼저 검토해야 한다. 각 파일의 필수 열은 다음과 같다.

```csv
impact_id,v_in_x,v_in_z,spin_before_y,v_out_x,v_out_z,spin_after_y
```

4·8회 fixed-budget를 실행하는 명령은 다음과 같다. 빈 GPU를 한 장 할당한 compute node에서 실행한다. 결과 파일에는 각 impact를 원자료 행과 연결할 입력 파일 hash 및 zero-based row indices가 저장된다.

```bash
python bench.py --external /path/to/one_new_product --output external_results
```

External entry point의 구동 여부는 계측값을 흉내 내지 않은 임의적인 유한 수치로 별도 검사했다. All-six source fit, 9행/3행 분리, 두 budget 및 네 방법의 2,048개 결과 row 생성을 확인했다. 생성한 수치와 관련된 성능 추정치는 보고하지 않는다. [기술 검사](interface_check.json)
