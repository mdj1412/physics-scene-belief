# v70 · held-out rubber의 velocity와 spin

한 제품을 통째로 가리는 6-fold 비교에서, v69의 two-regime affine는 inclined rigid보다 3개 제품에서 속도와 spin을 모두 더 정확하게, 다른 3개 제품에서는 모두 부정확하게 예측했다. 제품을 같은 비중으로 평가하면 속도 오차는 소폭 감소했지만 spin 오차는 증가했다. 이번 비교에서 기울기 효과는 작았다. 두 방향의 입사를 허용한 signed 1° grid에서도 모든 fold의 source가 선택한 effective tilt는 0°였다. [수치 및 fit 계수](metrics.json)

**대상과 단위.** Racket configuration을 rubber 제품에 따라 (1,3,9), (2,4), (5,6), (7), (8), (10)으로 묶었다. 같은 long-pips rubber의 1.2 mm와 OX를 함께 제외했다. Official train 중 다른 제품들의 충돌만 사용해 fit하고, 제외한 제품의 official test를 평가했다. 저자가 측정한 planar velocity (vx,vz), transverse spin ωy로 원 저자의 metric을 재현했다. 표의 속도는 shot별 2D Euclidean error의 평균(m/s), spin은 ωy absolute error의 평균(rad/s)이다. 상이한 단위의 지표를 하나의 점수로 합치지 않았다. [저자 benchmark](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) · [설계](DESIGN_KO.md)

**모델 이전.** Golf 실험의 coefficient를 재사용하면 다른 반지름·inertia와 고무 접촉 조건이 섞이므로, v69의 model class 및 fitting 규칙을 이전하고 각 fold의 source에서 parameter를 다시 구했다. Sphere는 R=0.020 m, k=I/(mR²)=2/3인 thin shell이다. Racket 측 좌표에서 tangential surface speed는 s=vx−Rωy이다. Release sign은 ground truth를 직접 읽은 local stick/slip label이 아니며, 여섯 rubber 모두 같은 식을 사용했다. Two-regime의 source에서 s_out 부호별 두 affine를 fit하고, source의 incoming 세 변수로 학습한 logistic gate만 test expert selection에 썼다. Inclined rigid는 v69와 같이 rotated normal restitution e를 least squares로 구한 뒤, vx와 Rωy의 squared error를 같은 비중으로 사용해 source의 tilt와 Coulomb μ를 선택했다. Flat rigid, signed tilt, single affine, constant elastic tangential impulse, source mean, specular controls를 함께 계산했다. 표에는 저자의 benchmark와 일치하는 ωy를 제시하며 다른 두 회전축의 정확도는 미검증이다. [구현](benchmark.py)

| Holdout 제품 | Primary test | Inclined rigid 속도 / spin | Two-regime affine 속도 / spin | Single affine 속도 / spin |
|---|---:|---:|---:|---:|
| Hexer Duro (inverted offensive) | 506 | 1.394 / 93.06 | 0.710 / 39.49 | 0.985 / 61.08 |
| Power 3 (inverted allround) | 321 | 1.260 / 92.00 | 0.549 / 37.30 | 0.717 / 51.92 |
| Grass D.TecS (long pips) | 299 | 0.482 / 34.44 | 1.584 / 119.10 | 1.340 / 102.36 |
| Diamant (medium pips) | 168 | 0.570 / 33.32 | 0.813 / 57.83 | 0.377 / 21.21 |
| Blowfish (short pips) | 171 | 1.374 / 92.35 | 0.602 / 33.75 | 0.804 / 48.68 |
| A-B-S II soft (anti-spin) | 158 | 1.445 / 36.22 | 2.134 / 115.49 | 1.958 / 90.10 |
| Product macro | 6 제품 | 1.087 / 63.57 | 1.065 / 67.16 | 1.030 / 62.56 |
| Shot micro | 1,623회 | 1.117 / 70.26 | 0.977 / 62.42 | 1.010 / 64.26 |

**Secondary controls.**

- 10개 configuration 중 allrounder blade의 5개 제품(IDs 5–10)에만 fit과 evaluation을 제한했다. 949회에서 macro velocity/spin은 inclined **1.071/58.30**, single affine 1.132/65.30, two-regime 1.253/78.18이다. Hexer·Blowfish와 Grass·Diamant·A-B-S에서 두 주요 모델의 승패 방향은 전체 실험과 같았다. 한 blade에서 재현했더라도 각 제품에 연결된 sponge는 달라질 수 있다. 6개 제품 전체로 돌아가 source의 각 제품 총 학습 가중치를 같게 해도 macro는 inclined 1.097/63.57, two-regime 1.079/66.81였다. [공통 blade와 가중치 sensitivity](diagnostics.json)
- 모든 fold에서 ±35°의 정수 후보 71개 모두 접근 조건을 만족해 실제로 평가했다. ±2°의 0.1° grid에서는 −0.4~−0.2°로 작은 각도가 선택됐다. 입사 vx 방향으로 tilt의 기준축을 바꾸는 exploratory model에서도 −0.3~−0.1°였다. 이 범위에서 0°라는 coarse-grid 선택을 물리적인 surface normal의 통계적 추정치로 해석할 수 없다. [Orientation audit](orientation.json)
- Target의 published train 행까지 독립적인 target test로 포함하고, 분할에 없는 71행까지 포함한 8,184회에 대해 source의 frozen coefficients를 별도로 실행했다. Micro는 inclined 1.121/70.08, two-regime 0.973/61.89, macro는 1.092/63.61 및 1.058/66.37이었다. 제외된 71행만 따로 보면 각각 0.549/18.70과 0.966/55.12였다. 8,113개 공개 split 행의 CPU replay와 원래 GPU 예측의 최대 차이는 rim 속도 기준 2.31×10⁻¹⁴ m/s였다. [전체 original 결과](original_metrics.json)
- Two-regime의 target sign 예측 정확도는 Hexer Duro 97.0%, Power 3 97.5%, Grass D.TecS **9.4%**, Diamant 95.2%, Blowfish 98.8%, A-B-S II soft **51.9%**였다. Grass에서 |s_out|>0.5 m/s인 238건만 남겨도 10건(4.2%)만 맞혔다. Target의 true sign을 알려주는 privileged switch는 Grass의 오차를 0.739/46.44로 줄였다. 각 endpoint에 유리한 expert를 target 정답을 보고 따로 고르는 oracle도 0.664/40.17, inclined는 0.482/34.44였다. 두 제품에서 sign 결정에 실패한 이유, 또는 실측 patch가 그 정보를 전달하는지는 이 실험만으로 구별할 수 없다. [Gate audit](gate_audit.json)
- 위 성능 기재는 zero-shot target surface에서 각 모델이 남기는 오차다. 특정 robot의 landing tolerance나 성공률을 이 수치만으로 판정할 수 없다.

**서버 기록.** 2026-11-17 datacenter3의 node56, Slurm allocation 2744471, NVIDIA RTX PRO 6000 Blackwell Server Edition에서 실행했다. Primary와 추가 GPU run은 각각 시작 전 0 MiB·0%와 빈 compute-process 목록을 확인했다. 모델 fitting·추론은 torch CUDA float64로, independent replay와 source-only angle sensitivity는 같은 allocation의 CPU로 실행했다. [Primary GPU 기록](gpu_before.txt) · [실행 결과](run.log) · [impact별 output](predictions.csv) · [전체 원본 row별 replay](original_predictions.csv.gz)
