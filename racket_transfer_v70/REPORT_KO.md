# v70 · 처음 보는 racket rubber에서 충돌 직후 velocity와 spin을 예측할 수 있는가?

**2026-11-17 · Gossard 공개 실물 impact의 leave-one-rubber-out 평가**

같은 rubber가 다른 blade나 sponge 두께로 쓰인 경우까지 묶어 **6개 제품을 한 번씩 통째로 제외**했다. 1,623개의 공식 test impact에서 v69의 두 모델은 각각 세 제품에서 두 지표 모두 앞섰다. **Two-regime affine는 inverted rubber 2종과 short pips에서, inclined rigid는 long pips·medium pips·anti-spin에서 오차가 작았다.** 모든 impact를 합치면 two-regime affine가 속도와 spin에서 모두 이겼지만, 제품을 같은 비중으로 평가하면 spin에서는 순위가 바뀌었다. 일반적인 새 rubber에 한 모델을 권할 만큼 일관된 우위는 관찰되지 않았다. [시험한 데이터 revision](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) · [실험 원문](https://arxiv.org/html/2604.11349v1)

가장 재현 가치가 높은 실패는 **Grass D.TecS에서 release 방향을 고르는 gate가 299번 중 28번만 맞힌 것**이다. 비슷한 입사 상태가 source에 있던 경우에도 실패했고, 정답 부호로 expert를 골라 주자 spin 오차가 119.10에서 46.44 rad/s로 줄었다. 이 결과는 표면에 따라 gate의 관계가 바뀌는 사례다. 실측 contact patch의 독립적 예측 이득은 대상 데이터의 관측 항목으로 식별되지 않았다. [Impact 단위 검증](analysis/audit.json) · [Patch 자료 감사](patch_schema_audit.json)

## 1. 무엇을 학습하고 무엇을 숨겼나

Gossard가 고정 racket에 ball을 발사해 충돌 전후 운동을 촬영한 실험을 사용했다. 원 연구는 기존 설정 안에서 rebound를 예측하고 일부 조건에서 앞선 target 관측으로 적응했다. v70의 primary question은 **해당 rubber를 단 한 번도 calibration하지 않은 상태에서의 예측**이다. v69 golf에서 fit한 숫자를 옮기는 대신 같은 두 model family와 fitting 원칙을 적용하고, 각 fold의 다른 racket 표면만으로 parameter를 다시 추정했다. [Gossard의 측정·평가 방법](https://arxiv.org/html/2604.11349v1) · [v70 설계](solver/DESIGN_KO.md)

| 평가 단계 | 설계 |
|---|---|
| 제품 단위 분리 | Hexer Duro의 구성 1·3·9, Power 3의 2·4, Grass D.TecS의 5·6을 함께 제외했다. Diamant 7, Blowfish 8, A-B-S II soft 10은 각자 한 fold다. Grass의 sponge 유무가 달라도 같은 제품에 넣었다. |
| Primary split | 다른 5개 제품의 공식 train rows만 fitting, 제외한 제품의 공식 test rows 전부로 평가. 제품명·blade·racket angle code·trajectory ID는 predictor 입력에서 제외했다. 각 target의 train rows도 zero-shot fitting에서 제외했다. |
| 관측 단위 | 물체 중심의 측면 궤적을 중력 보정한 50 ms 구간에서 linear fit한 pre/post velocity, dotted ball로 측정한 spin. 충돌 직후 값은 free-flight로부터 추정한 rebound state다. Patch에 닿은 순간의 국소 속도와 구분한다. |
| 모델 입력·출력 | Incoming $(v_x,v_z,Rω_y)$에서 outgoing 세 성분을 예측. $R=0.020$ m, thin shell의 $I/(mR²)=2/3$. 이 좌표계에서 접촉점의 signed tangential speed는 $v_x-Rω_y$이다. |
| 평가량 | Velocity는 impact별 XZ vector 차이의 Euclidean norm을 평균(m/s), spin은 $ω_y$의 MAE(rad/s). CSV의 모든 행에서 $v_y$는 0이다. 이번 자료로 보고할 수 있는 것은 planar 성능이다. |
| 원자료·중복 | 공개 original 8,184행, train 6,490행, test 1,623행. 원고에는 8,194회로 기재되어 있어 계산에는 배포 파일 수를 사용했다. Train/test와 exact numeric match가 없는 original 71행을 따로 점검했다. Original의 6성분 incoming state가 완전히 중복되는 행은 없었다. |

각 제품의 표본 수가 크게 달라 새로운 **제품** 하나가 단위인 질문에는 product macro를 먼저 사용했다. Impact micro는 공개된 shot 구성에서 기대되는 평균 오차다. Six-product holdout은 새 specimen이나 제조 batch 전체에 대한 무작위 표본이 아니다. 허용 가능한 landing error 또는 robot success threshold는 지정되지 않아 실제 경기 성능은 따로 판정하지 않았다. [데이터 점검과 checksum](data/DATA_KO.md) · [저자 metric](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/python/racket_dynamics/evaluation/metrics.py)

## 2. 같은 shot에서 비교한 모델

- **Inclined rigid.** 유효 접촉면을 회전하고 normal restitution과 Coulomb tangential impulse를 source에서 fit했다. 각도 0–35° 및 signed −35–35°를 검사하고, friction-capped impulse가 rolling에 도달하면 멈추도록 했다. Source fitting 목적은 tangential COM과 rim velocity에 동일 가중치를 준 squared error다.
- **Two-regime affine.** Source release의 $v_x'-Rω_y'$ 부호로 두 affine expert를 fit했다. Incoming state만 받는 logistic gate가 test에서 expert를 고른다. 여기서 regime은 운동학적으로 정의한 두 부호이며, 실제 접촉면의 stick/slip label은 측정되지 않았다.
- **강한 기준선.** Gate가 없는 single affine, constant elastic impulse, 그리고 incoming state에 대한 nonlinear RBF regression을 비교했다. RBF는 bandwidth와 ridge penalty를 outer fold의 다섯 source rubber 안에서 다시 한 제품씩 제외해 골랐다. 원 논문의 각 racket 전용 Gaussian process와 학습 조건이 다르다. Source mean과 specular reflection도 실행했다. [Solver 코드](solver/benchmark.py) · [RBF 설계와 예측](analysis/rbf.py)

## 3. Held-out 속도와 spin

각 칸은 **velocity error (m/s) / spin MAE (rad/s)**다. 두 지표 모두 작을수록 좋다.

| 제외한 rubber | 형태 | Test impacts | Inclined rigid | Two-regime affine | Single affine | Pooled RBF |
|---|---|---:|---:|---:|---:|---:|
| andro Hexer Duro | Inverted · offensive | 506 | 1.394 / 93.06 | **0.710 / 39.49** | 0.985 / 61.08 | 0.961 / 59.13 |
| andro Power 3 | Inverted · allround | 321 | 1.260 / 92.00 | **0.549 / 37.30** | 0.717 / 51.92 | 0.675 / 48.60 |
| Tibhar Grass D.TecS | Long pips | 299 | **0.482 / 34.44** | 1.584 / 119.10 | 1.340 / 102.36 | 1.398 / 106.97 |
| Dr. Neubauer Diamant | Medium pips | 168 | 0.570 / 33.32 | 0.813 / 57.83 | **0.377 / 21.21** | 0.391 / 22.03 |
| andro Blowfish | Short pips | 171 | 1.374 / 92.35 | **0.602 / 33.75** | 0.804 / 48.68 | 0.775 / 45.97 |
| Dr. Neubauer A-B-S II soft | Anti-spin | 158 | **1.445 / 36.23** | 2.134 / 115.49 | 1.958 / 90.10 | 1.962 / 88.92 |
| **Product macro** | 6개 제품 동등 가중 | 1,623 | 1.087 / 63.57 | 1.065 / 67.16 | 1.030 / 62.56 | **1.027 / 61.94** |
| **Impact micro** | 각 impact 동등 가중 | 1,623 | 1.117 / 70.26 | **0.977 / 62.42** | 1.010 / 64.26 | 1.004 / 63.54 |

RBF가 product macro에서 single affine을 앞선 차이는 0.003 m/s, 0.62 rad/s였다. 같은 기준에서 constant elastic impulse는 1.048 m/s와 62.76 rad/s, source mean은 2.897 m/s와 82.90 rad/s였다. 즉, incoming state로부터 평균 이상의 예측 정보는 얻었다. 다만 두 주모델의 spin 오차 63.57–67.16 rad/s는, 제품별로 먼저 평균한 실제 rebound spin 절댓값 116.21 rad/s에 비해 작다고 보기 어렵다. [모든 기준선과 paired 결과](analysis/audit.json) · [속도와 spin의 실측 규모](analysis/gate_margin.json)

![6개 rubber 및 두 종류의 평균에서의 holdout 결과](analysis/rubber_holdout_v70.png)

초록색은 해당 target rubber의 별도 train labels로 학습한 진단 결과다. 다른 세 모델에는 그 labels를 제공하지 않았다. [벡터 그림](analysis/rubber_holdout_v70.pdf)

**평균의 순위가 왜 달라지는가.** Hexer Duro와 Power 3의 impact가 827/1,623회(51.0%)이고, 두 제품에서 two-regime의 이득이 크다. 한 제품씩 빼고 product macro를 다시 계산하면 two-regime에서 inclined를 뺀 차이가 velocity −0.247~+0.128 m/s, spin −12.62~+16.04 rad/s여서 양쪽 모두 부호가 바뀐다. 이 범위는 여섯 제품에 대한 influence diagnostic이다. 수천 회의 impact를 독립적인 새 표면의 수로 세어 작은 confidence interval을 만들지 않았다. [제품별 영향과 계산 코드](analysis/audit.py)

## 4. Long-pips에서 어느 단계가 실패했나

| 제외한 제품 | Gate가 실측 release 부호를 맞힌 비율 | 실제 gate의 spin MAE | 실측 부호로 expert를 골라 준 spin MAE |
|---|---:|---:|---:|
| Hexer Duro | 97.0% | 39.49 | 39.08 |
| Power 3 | 97.5% | 37.30 | 36.89 |
| Grass D.TecS | **9.4%** | **119.10** | **46.44** |
| Diamant | 95.2% | 57.83 | 57.82 |
| Blowfish | 98.8% | 33.75 | 33.77 |
| A-B-S II soft | 51.9% | 115.49 | 76.71 |

**Grass의 실패는 부호 경계의 미세한 측정 오차만으로 설명하기 어렵다.** 실측 release tangential speed의 절댓값이 1 m/s를 넘는 사례만 남기면 gate는 173회 중 **5회(2.9%)**만 맞혔다. 동일 기준에서 두 inverted 제품은 440/440회, 263/263회였다. 정답 부호로 Grass의 expert를 교체해도 spin MAE 46.44 rad/s가 남아 inclined의 34.44보다 컸다. 반대로 Diamant에서는 부호를 거의 다 맞혀도 single affine의 21.21에 크게 뒤졌다. 제품에 따라 잘못된 **expert 선택**과 **expert 자체의 response**가 서로 다른 병목이었다. 정답을 사용한 이 비교는 사후 진단이며 실행 가능한 zero-shot policy는 아니다. [부호 margin별 감사](analysis/gate_margin.json) · [Oracle routing 결과](analysis/audit.json)

**입사 조건도 따로 확인했다.** 다른 source rubber까지의 nearest-neighbor distance로 정한 source 기준 안에 1,577/1,623개 test impact가 있었다. Grass에서는 281/299개였고, 이 subset에서도 two-regime의 spin MAE가 120.98, inclined가 34.75 rad/s였다. 표준화한 세 입력에서 이웃이 있다는 사실은 촬영 session, 내부 변형, 마찰이 같다는 증거는 아니다. 다만 이 실패가 생소한 입사 속도와 spin의 극소수 사례에만 집중됐다는 설명은 자료와 맞지 않는다. [Target-free support 정의](analysis/PLAN_KO.md) · [Incoming-state별 결과](analysis/audit.json)

**같은 표면의 calibration은 얼마나 다른가.** Pooled RBF의 세 입력, basis 수와 source에서 정한 hyperparameter를 유지하고, 해당 target rubber의 공식 train rows로만 fit했다. 분리된 target test에서 6개 제품 모두 개선됐으며 product macro는 **1.027 → 0.213 m/s, 61.94 → 8.41 rad/s**였다. 같은 개수의 다른 source rubber rows를 샘플링한 기준선은 1.022 m/s, 61.39 rad/s였다. Target별로 632–2,022회의 calibration을 쓴 진단이라 few-shot 성능을 말해 주지 않는다. 재질, blade·sponge 구성, session 등 무엇이 차이를 만들었는지는 분리되지 않았다. [Fitting 수와 검증 설계](analysis/rbf_fit.json) · [동일 예산 비교](analysis/audit.json)

## 5. 실측 contact patch가 독립적인 정보를 주는가?

**이 데이터에서 그 이득은 미식별이다.** 저자 저장소의 tracked 파일과 CSV 30개를 검사했다. 14개 열은 racket angle, trajectory ID, 충돌 전후 spin·velocity다. 사진은 장비 구성 사진 한 장이며, 개별 shot과 연결된 patch mask·pressure centroid·local deformation·force 또는 고속 원영상이 배포 파일에 없다. Solver가 계산한 contact speed나 충돌 후 spin으로 역산한 impulse를 실측 patch로 넣으면 새로운 독립 관측의 가치를 시험할 수 없다. [고정 revision의 파일 감사](patch_schema_audit.json) · [저자 저장소](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

서로 보완하는 공개 연구도 확인했다. Rémond et al.은 탁구공이 유리판에 수직 충돌하는 동일 영상에서 shell deformation과 rebound를 읽었다. Zhang et al.의 racket에는 실측 sensor array가 있지만 공개된 회전 hit supplement의 열은 시간과 10개 voltage channel이다. 두 연구 모두 여기서 필요한 **같은 rubber의 oblique impact마다 측정한 patch와 독립적인 post-impact 속도·spin 쌍**을 제공하지 않는다. Rinaldi et al.의 racket 유사 표면 연구에서 시각화한 국소 변형장은 finite-element model의 출력이다. [유리판 고속 촬영](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Racket sensor 원자료](https://doi.org/10.1371/journal.pone.0333735.s003) · [Finite-element 및 촬영 실험](https://www.mdpi.com/2076-3417/9/1/158)

검증할 질문은 두 가지다. **충돌 전에** 예측하려면 이전 calibration impact에서 얻은 patch 특징을 사용하고, **충돌 도중** 예측하려면 현재 impact의 사전에 고정한 시점까지의 patch만 사용해야 한다. 후자의 비교에서는 두 모델에 같은 시간까지의 ball motion을 주고, 하나에만 독립 센서의 patch를 추가한다. Incoming state·blade·rubber를 맞춘 trial 안에서 patch를 섞는 negative control, 같은 calibration 예산의 kinematics-only model, 새 rubber 전체 holdout을 함께 두면 patch 고유의 정보와 target calibration 효과를 구별할 수 있다. [근거와 구체적 후속 질문](LITERATURE_KO.md)

## 6. 다른 설명을 점검한 결과

| 대안 설명 | 추가 비교 | 결과 |
|---|---|---|
| Racket에 기울기가 필요한데 grid가 놓쳤는가? | 같은 source에 ±35°와 ±2°의 0.1° grid를 적용하고 입사 수평 방향에 맞춘 좌표계도 검사 | 1° signed grid의 71개 후보가 모두 허용됐고 여섯 fold 모두 0°를 골랐다. 세분하면 고정 좌표계에서 −0.2°~−0.4°, 입사 방향에 맞추면 −0.1°~−0.3°였다. v69 turf에서 유용했던 큰 effective tilt의 이득은 여기서 재현되지 않았다. [Orientation 결과](solver/orientation.json) |
| 특정 blade 구성 때문에 승패가 뒤집혔는가? | Allrounder blade에 해당하는 구성 5–10으로 source와 target을 모두 한정하고 동일 절차로 5개 rubber 각각 재학습 | 949회에서 product macro는 inclined **1.071 / 58.30**, two-regime **1.253 / 78.18**, single affine **1.132 / 65.30**. 비교 가능한 각 제품의 주모델 승자는 유지됐다. 같은 blade 종류로 제한해도 sponge 두께·rubber 재질·session은 함께 변할 수 있다. [Matched-blade 결과](solver/diagnostics.json) |
| 학습 표본이 많은 source 제품에 치우쳤는가? | Source rubber에 동일 총 가중치를 주어 재학습 | 두 주모델의 승수는 여전히 3 대 3이었다. Macro 속도·spin은 two-regime **1.079 / 66.81**, inclined **1.097 / 63.57**였다. [Balanced fit](solver/diagnostics.json) |
| 배포된 test subset만의 결과인가? | 각 fold의 fitting 결과를 고정해 별도로 공개된 original 8,184행 전체를 재평가 | Product macro는 two-regime **1.058 / 66.37**, inclined **1.092 / 63.61**로 지표별 방향이 유지됐다. 공식 train/test에서 빠진 71회만 보면 micro는 각각 **0.966 / 55.12**, **0.549 / 18.70**이었다. 이 71회 중 Grass가 52회여서 전체 제품을 대표하지 않는다. [Original replay 및 제외분](solver/original_metrics.json) |
| Gate만 완벽하면 Grass의 문제가 풀리는가? | 사후에 두 expert 중 각 shot의 실제 오차가 더 작은 쪽을 고르는 endpoint별 oracle | Grass의 이론상 선택 결과도 **0.664 / 40.17**로 inclined **0.482 / 34.44**에 못 미쳤다. 두 metric에 서로 다른 expert를 선택할 수 있는 유리한 조건이다. 제공된 두 expert의 선택만 바꾸는 데 한계가 있었다. [Oracle audit](solver/gate_audit.json) |

## 7. 논문에 넣을 판단과 재현 자료

**본문에 넣을 만한 결과**는 새 표면에서 전체 평균과 제품별 결론이 달라지고, 동일한 incoming state 범위에서도 특정 rubber에서 release 방향을 고르는 규칙 자체가 역전된다는 실측 사례다. v69의 turf에서는 inclined contact normal이 속도 예측을 개선했다. v70에서는 거의 수평인 normal이 선택됐으며, 표면에 따른 release response를 학습하지 못한 오차가 크게 남았다. 접촉 데이터를 추가할 연구라면 Grass와 A-B-S II soft를 실패 사례로, 비슷한 입사 조건인데도 gate가 잘 작동한 inverted rubber를 대조군으로 먼저 삼을 근거가 생겼다. [v69에서 사용한 공개 turf 연구](https://doi.org/10.1007/s12283-023-00442-4) · [v70 제품별 결과](analysis/audit.json)

**기여의 한계**도 분명하다. Gossard가 이미 rubber-dependent dynamics와 calibration의 이점을 보고했다. v70이 더하는 것은 동일 제품 전체를 숨긴 조건의 비교와 gate·expert 실패를 나눈 진단이다. 독창적인 새 contact solver, 독립 patch의 효과, pips가 실패의 물리적 원인이라는 주장, 로봇의 성공률 개선까지 입증한 단계는 아니다. 다음 확인 실험에서 새로운 제조사·물체·session을 통째로 분리하고, 같은 calibration 비용으로 surface response의 방향을 알아낼 수 있는지 검증하면 현재의 사후 진단을 배포 가능한 정책으로 발전시킬 수 있다. [원 연구와 비교](https://arxiv.org/html/2604.11349v1)

**실행 환경.** `datacenter3`의 Slurm allocation 2744471, node56의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. 매 GPU 실행 전 compute PID, 사용 메모리, utilization을 확인했고 primary 시작 기록은 0 MiB·0%이다. PyTorch 2.7.1+cu128의 float64로 fitting과 예측을 실행했다. Original 자료에 대한 별도 CPU replay와 GPU의 공통 8,113개 shot 출력 차이는 rim-speed 환산 최대 2.31×10⁻¹⁴ m/s였다. [GPU 시작 기록](solver/gpu_before.txt) · [Primary 실행](solver/run.log) · [RBF fold 기록](analysis/rbf_fit.json) · [CPU replay](solver/original_metrics.json)

서버 재현 디렉터리는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/racket_transfer_v70`이다. [Impact별 주모델 예측](solver/predictions.csv), [RBF 예측](analysis/rbf_predictions.csv), [8,184행 replay](solver/original_predictions.csv.gz), [데이터·단위 감사](data/DATA_KO.md), [출판용 그림](analysis/rubber_holdout_v70.pdf)을 함께 보관했다. 아래 부록에는 독립적으로 검토한 원문별 표, 근거의 강도, 상충 결과, 후속 질문 10개와 논문에 대한 검토가 이어진다.

재실행은 배포된 CSV와 CUDA 지원 PyTorch, NumPy를 준비한 뒤 이 디렉터리에서 아래 순서로 할 수 있다. 서버 실행 기록의 compute job 환경을 사용하고, 원본 결과 보존이 필요하면 재현 bundle을 별도 디렉터리에 풀어서 실행한다. 그림 생성에는 matplotlib가 필요하다. [상세 solver 기록](solver/SOLVER_KO.md)

```sh
python solver/benchmark.py
python solver/diagnostics.py
python solver/replay.py
python solver/orientation.py
python solver/gate_audit.py
python analysis/rbf.py
python analysis/audit.py
python analysis/gate_margin.py
python analysis/figure.py
```

## 부록 · 9가지 기준으로 본 문헌과 contact patch 검증

### 1. 원문별 질문, 측정, 결과와 한계

| 연구와 질문 | 방법·표본 | 주요 결과와 결론 | v70에서 확인 가능한 범위 |
|---|---|---|---|
| Gossard (2026, preprint). 서로 다른 rubber의 rebound를 incoming state로 예측할 수 있는가? | 6개 rubber 제품, 3개 blade 종류, 10개 구성. 370 fps planar tracking, 50 ms local regression, dotted ball spin 측정. 공개 original CSV에 8,184개 impact. | 저자 평가에서 상태에 따라 달라지는 impulse parameter가 constant model보다 정확했다. 저자 Gaussian process는 각 racket의 학습자료로 적합했다. | 기존 80/20 split에는 모든 구성이 양쪽에 있다. 새 rubber를 처음 접하는 v70의 비교에는 target 자료를 사용하는 fit과 공개 aggregate score를 그대로 가져올 수 없다. [원문 §III–V](https://arxiv.org/html/2604.11349v1) · [저자 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Nakashima 외. 표면 종류에 따라 다른 tangential impulse를 적용해야 하는가? | 고정 table과 Butterfly racket에 공을 발사하고 900 fps 카메라로 중심과 회전을 추적했다. | Racket rubber가 접선 방향의 탄성 에너지를 돌려준다는 가정에서 입사 접선 속도에 비례하는 impulse를 유도하고 영상 자료와 비교했다. | 단순한 rigid/Coulomb의 물리적 대안이다. 다양한 상용 rubber의 product-level transfer나 patch 판독을 검증한 연구는 아니다. [원문 §II–IV](https://nagoya.repo.nii.ac.jp/record/12580/files/1031.pdf) |
| Rinaldi 외. Foam과 상부 compact layer가 회전하는 공의 충돌을 어떻게 바꾸는가? | 14 m/s, 3개 입사각, 3개 spin 조건, foam 및 foam+compact 표적의 촬영 실험과 3D finite-element model. | 실험에서 restitution, 반사 방향, spin ratio를 구했다. 모형으로 두 층의 국소 변형을 계산했다. | Fig. 9는 **모사한** displacement field다. Fig. 4–5의 광학 측정값은 공 궤적과 표식이다. Fig. 9를 외부 실측 patch로 쓰면 simulator 가정을 다시 평가하는 셈이다. [원문 §2–4 및 Fig. 9](https://www.mdpi.com/2076-3417/9/1/158) |
| Rémond 외 (2022). 얇은 shell의 buckling을 impact 중에 직접 볼 수 있는가? | 지름 40 mm ABS ball, 무회전 normal impact, 고정 유리, 약 1–12 m/s. 20,000 fps로 같은 영상에서 shell 단면과 전후 속도를 판독. | 약 5.5±0.5 m/s에서 접촉 중심이 들리기 시작했고 restitution 변화율도 달라졌다. 외부 직경이 같아도 중심부 접촉 여부가 달라질 수 있다. | 같은 trial에서 geometry와 rebound를 측정하는 방법이 실현 가능하다. 유리·정면 충돌에서 입사 속도를 바꾼 결과이므로 rubber의 oblique impact에서 patch의 독립적 예측 이득을 확인하려면 새 비교가 필요하다. [원문 §II, Fig. 2–6](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| Zhang·Fan·Kang (2025). Racket에 부착한 sensor로 hit 위치와 회전을 구분할 수 있는가? | 5×5 row-column PVDF array, 10개 전압 channel. S2는 서로 다른 위치의 4 hits, S3는 회전 조건의 3 hits. | 행·열 신호로 타격 위치를 찾고 회전 방식에 따른 신호 차이를 보고했다. 압력 변화에 반응하는 실물 데이터다. | S3의 11열은 시간과 channel voltage이다. 고속 영상의 독립 rebound velocity·angular velocity 또는 실제 접촉 경계가 주어지지 않는다. [원문과 supplement 목록](https://doi.org/10.1371/journal.pone.0333735) · [직접 검사한 S3](https://doi.org/10.1371/journal.pone.0333735.s003) |
| Biber 외 (2023). 두 turf에서 golf ball의 전후 운동은 어떤 모델로 설명되는가? | 인조·천연 turf 2종, 1,023회, 입출력 translation과 spin의 영상 측정. | 저자는 inclined surface와 piecewise affine를 포함한 모델들을 비교했고, 간단한 데이터 기반 fit이 당시 검증 조건에서 물리 model을 앞설 수 있었다. | 골프공의 잔디 충돌과 얇은 탁구공의 rubber 충돌은 inertia·표면 구조·실험 장비가 다르다. v70은 모델 계열의 전이를 검증한다. [측정·모델링 원문](https://doi.org/10.1007/s12283-023-00442-4) |

### 2. 문헌 종합

충돌 전후의 **전체 운동**은 공의 중심과 회전 표식을 추적해 검증할 수 있다. Gossard는 서로 다른 rubber에서 이 측정을 반복했고, Nakashima와 Rinaldi는 에너지를 저장하는 rubber의 접선 응답이 rebound와 spin에 미치는 영향을 모델링했다. 이들 연구가 주는 공통 제약은 잘 맞은 endpoint만으로 힘이 어느 위치에 어떻게 분포했는지 결정할 수 없다는 점이다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Nakashima 외](https://nagoya.repo.nii.ac.jp/record/12580/files/1031.pdf) · [Rinaldi 외](https://www.mdpi.com/2076-3417/9/1/158)

**접촉 경계**는 별도의 관측이다. 투명 유리 아래에서 본 shell 단면에서는 접촉 중심의 이탈을 프레임마다 판독할 수 있었다. 불투명 racket에 붙인 전압 sensor는 전달된 응력에 반응하므로, 시간별 실제 접촉 경계의 바깥에서도 신호가 측정될 수 있다. 두 관측을 같은 변수로 사용하려면 먼저 같은 충돌에서 광학적으로 경계를 판독해 sensor의 spatial response를 보정해야 한다. [Rémond 외](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang 외](https://doi.org/10.1371/journal.pone.0333735)

### 3. Knowledge map

입사 상태와 장착 조건 → shell·rubber·sponge 변형 → 접촉 경계와 국소 traction → 힘과 토크의 시간 적분 → 이탈 velocity와 spin으로 이어진다. Gossard와 Nakashima의 관측은 주로 양 끝에 있다. Rémond는 중간 단면을 광학적으로 관측했고, Zhang은 지지부에 전달된 stress signal을 기록했다. Rinaldi는 실측 양 끝과 그 사이를 연결하는 FE 해석을 제공한다. **가운데의 서로 다른 변형 history가 같은 endpoint로 합쳐질 수 있다.** 따라서 endpoint residual에서 geometry의 존재 가능성을 제안할 수 있으나 공간 분포를 유일하게 복원할 수는 없다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735) · [Rinaldi](https://www.mdpi.com/2076-3417/9/1/158)

### 4. 연구 공백과 실제 파일 감사

- **같은 trial에서 추가 feature를 분리할 수 있는가.** 저자 repository의 revision 159c6f04에서 전체 파일을 조사했다. CSV 30개의 14열은 racket angle, trajectory directory, 전후 spin 세 축, 전후 velocity 세 축이다. 유일한 이미지 파일은 장비 사진 `rackets.jpeg`이다. `traj_dir`은 원 acquisition의 이름이며 그 행에 대응하는 patch 영상이 공개 tree에 없다. 따로 조사한 Zhang의 S3 workbook에도 시간과 전압 10개만 있다. 공개 자료 사이에 동일 impact를 잇는 키가 없다. [원 repository](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) · [Zhang S3](https://doi.org/10.1371/journal.pone.0333735.s003) · [파일 검사 결과](patch_schema_audit.json)
- **계측과 계산의 구분.** 위치 한 점 $(x,y)$는 racket의 어디를 맞혔는지 나타낸다. `v_s`는 rigid ball의 회전·translation에서 계산한 접점 상대 속도이다. 시간별 면적·경계·압력 분포는 공간과 시간에 걸친 별도 관측이다. Incoming state에서 생성한 simulated patch는 학습 prior의 가치를 시험할 수 있다. Outbound로부터 역산한 patch 또는 impulse coefficient를 같은 행의 입력으로 사용하면 label leakage가 생긴다. [Gossard §II, §V](https://arxiv.org/html/2604.11349v1) · [Rinaldi Fig. 9](https://www.mdpi.com/2076-3417/9/1/158)
- **독립 표면·반복 단위.** 원 논문의 10개 racket 구성에는 동일 rubber 제품을 다른 blade 또는 sponge thickness로 장착한 사례가 있다. 1·3·9, 2·4, 5·6은 각각 같이 holdout해야 새 rubber 제품이 된다. 제품 수가 6개이고 일부 조합만 존재하므로 제품별 성능 차이를 pimple·blade·sponge 중 하나의 인과 효과로 배분할 수 없다. [Gossard Tables II–IV](https://arxiv.org/html/2604.11349v1)

### 5. 서로 달라 보이는 연구 결과

Gossard에서 state-dependent GP가 알려진 racket의 성능을 높였다는 것과 새 rubber에서 flexible model이 실패할 수 있다는 것은 다른 평가 조건이다. 전자에는 같은 racket의 training impacts가 있고, 후자는 target product에 대한 계수 보정 기회가 없다. 또한 rigid table model의 rolling clamp와 rubber 탄성을 담은 impulse model은 회전 반전이 가능한 범위가 다르다. 서로 다른 오차를 설명하려면 같은 held-out shot의 velocity와 spin을 따로 비교하고 두 평가에서 사용할 수 있는 정보를 명시해야 한다. [Gossard §III–V](https://arxiv.org/html/2604.11349v1) · [Nakashima 외](https://nagoya.repo.nii.ac.jp/record/12580/files/1031.pdf)

### 6. 방법 간 비교

| 방법 | 장점 | 주요 취약점과 검정 |
|---|---|---|
| Inclined rigid | 소수의 parameter로 effective normal과 tangential impulse를 함께 제한한다. | 식별된 기울기는 잔차를 줄이는 유효값이다. 카메라에서 측정한 실제 patch normal과 비교하려면 독립 관측이 필요하다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| Two-regime affine | 방향 및 회전 결합의 서로 다른 두 mapping을 구분해 학습할 수 있다. | Gate·expert fit을 source로 제한한다. 소수 regime의 target에서 outlier가 결과를 주도하는지 product별 오차를 본다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| GP 또는 state-dependent impulse | $(v_s, v_z)$에 따라 연속적으로 바뀌는 restitution과 impulse를 학습한다. | 원문 GP의 hyperparameter는 individual racket에서 적합됐다. Source-only로 합친 모델이나 몇 회의 target calibration을 사용한 모델은 별도 비교군이다. [Gossard §IV–V](https://arxiv.org/html/2604.11349v1) |
| Optical profile 및 PVDF array | 전자는 geometry의 시계열, 후자는 backing에 전달되는 stress의 시계열을 읽는다. | 유리의 정상 충돌이나 소수 회전 hit 결과를 6개 rubber에 행 단위로 결합할 수 없다. 시간 해상도·시야·공간 응답·측정 위치가 서로 다르다. [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735) |

### 7. Evidence synthesis

**강한 근거.** 공개 racket 자료는 동일 shot에서 운동을 측정했으므로 planar rebound velocity와 spin 오차를 직접 채점할 수 있다. 채점 단위는 실제 impact이며 서로 다른 holdout으로 평가하는 단위는 6개 rubber 제품이다. **조건부 근거.** Shell buckling, rubber의 탄성, stress propagation이 contact 과정에 관여한다는 실험 결과가 있지만 운동과 접촉 경계를 동시에 읽은 가장 가까운 연구는 유리의 normal impact이다. **미식별.** 같은 입사 정보와 같은 prediction 시점에 측정 patch를 추가할 때 새 rubber에서의 loss가 얼마나 줄어드는지는 Gossard 자료에서 계산할 수 없다. 정보가 없다는 판정에 수치상 gain=0을 대입하면 안 된다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735)

### 8. 근거에서 나온 후속 질문 10개

| 질문 | 왜 필요한가·변수 | 평가 설계와 근거 |
|---|---|---|
| 1. 같은 rubber에서 blade가 바뀐 효과가 제품 교체보다 큰가? | Blade와 rubber의 교란, velocity·spin error | Inverted 제품의 반복 blade 조합에서만 두 holdout을 별도로 평가한다. [Gossard Table IV](https://arxiv.org/html/2604.11349v1) |
| 2. Sponge가 없는 long pips에서 spin을 가장 크게 놓치는가? | 동일 제품 안의 sponge thickness 차이 | 두 long-pips 구성을 동시에 holdout한 예측을 configuration별로 나누고 incoming support를 맞춘다. [Gossard §V-C](https://arxiv.org/html/2604.11349v1) |
| 3. Incoming spin이 비슷한 shot에서도 새 rubber의 uncertainty가 커지는가? | state shift와 material shift 구분 | 공통 입사 범위에서 제품별 paired residual을 비교한다. [Gossard §III–V](https://arxiv.org/html/2604.11349v1) |
| 4. 소수의 target shots로 zero-shot 실패를 회복하는가? | zero-shot과 few-shot의 경계 | 무작위 calibration k회를 분리하고 한 번도 fit에 쓰지 않은 나머지 shot을 평가한다. [Gossard §IV](https://arxiv.org/html/2604.11349v1) |
| 5. 고정 기울기의 이득이 독립 측정한 표면 방향과 일치하는가? | effective parameter의 물리적 의미 | 실제 surface profile을 blind reconstruction해 fitted inclination과 paired 비교한다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| 6. 같은 outer diameter에서 annular contact가 spin 정보를 더 주는가? | 면적 추정이 놓치는 중심 이탈 | Pre-impact state·외경을 맞춘 뒤 중앙 들뜸과 local flow를 측정하고 held-out spin 오차를 비교한다. [Rémond Fig. 2–5](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| 7. 격자 센서에서 같은 hit 위치라도 서로 다른 patch를 구별할 수 있는가? | stress propagation과 경계 측정의 혼동 | 동일 landing cell에서 time-aligned optical boundary로 10개 channel readout을 검증한다. [Zhang S2·S3](https://doi.org/10.1371/journal.pone.0333735) |
| 8. Area, centroid path, contact duration 중 어떤 측정이 추가 정보를 주는가? | 더 적은 비용으로 찍을 수 있는 feature 선택 | 같은 shot, 같은 model capacity에서 feature를 하나씩 추가하고 lag와 cost를 함께 기록한다. [Rinaldi](https://www.mdpi.com/2076-3417/9/1/158) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| 9. 입사속도만으로 생성한 FE proxy가 실측 patch 효과를 대체하는가? | inductive bias와 독립 관측의 가치 구분 | kinematics only, kinematics+FE, kinematics+measured patch를 같은 source-only fit과 target에 비교한다. [Rinaldi Fig. 9](https://www.mdpi.com/2076-3417/9/1/158) |
| 10. 같은 shot의 접촉 정보가 실제 decision 시점에 도착하는가? | 사전 예측과 충돌 중 nowcast의 시간 차이 | Pre-impact prediction, pre-liftoff prefix, 과거 calibration impact의 patch를 분리하고 sensor latency를 센다. [Gossard §I](https://arxiv.org/html/2604.11349v1) · [Rémond §II](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |

### 9. 논문 심사 관점에서 보강할 주장

v70에서 주장할 수 있는 단위는 **측정된 표면 제품을 통째로 제외한 rebound prediction**이다. Shot 개수를 독립적인 surface 개수처럼 보고하면 transfer의 확실성을 과대평가한다. 특정 제품에서 모델이 틀렸다면 target에 알려지지 않은 restitution, sponge 구조, 불충분한 입사 범위, spin 추정 잡음 등도 같은 패턴을 만들 수 있다. 두 모델의 순위만으로 실패 원인이 patch라고 결론내릴 수 없다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi](https://www.mdpi.com/2076-3417/9/1/158)

독립적인 patch 이득의 주 비교는 동일 held-out impact의 **입사 상태만 사용했을 때의 loss에서 같은 상태와 실제로 측정한 patch를 사용했을 때의 loss를 뺀 값**이다. 두 방법에 같은 source, decoder capacity, 시간 cutoff를 준다. Outer fold에서 rubber 제품을 분리하고 같은 제품·비슷한 입사 조건 안에서 patch를 shuffle한 negative control을 두면, patch가 단지 rubber ID나 입사 속도를 알려준 효과를 점검할 수 있다. 같은 impact의 접촉 정보는 충돌 시작 이후에만 읽을 수 있으므로 pre-impact deployment를 주장하려면 별도 calibration shot에서 얻은 patch를 쓰고 감지·reset 비용을 포함해야 한다. [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735) · [Gossard](https://arxiv.org/html/2604.11349v1)
