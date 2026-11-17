# v52 · 교체 후 관측으로 posterior를 갱신하고 세 action을 고르는 방법

교체 전 실제 유지율을 κ, 교체 후를 η라 하자. 이 실험의 latent drift event에서 η=1−κ이고, 유지되면 η=κ다. 추정 단계에서는 κ~Uniform(0,1), P(drift)=0.25를 미리 고정했다. Reference와 impact의 부호가 다를 확률에도 설계값 0.10을 부여했다. 이 상수들은 실제 배포 표면에서 관찰한 빈도가 아니다. 배포 정책에는 표면 이름, 실제 η·κ, drift event, impact label이 들어가지 않는다. [동결 계획](POLICY_PLAN_KO.md)

Calibration용 물체 4개의 weak/full displacement를 교체 전 표면에서 취했다. 교체 후에는 같은 contact law의 반대 표면에서 K=0·1·2·4·8쌍을 취했다. 교체 전후 specimen은 서로 다른 environment 행을 사용했다. 교체 후 repetition 순서에는 catalog별 독립 순열을 고정했다. 배포와 calibration의 physical parameter support는 v50에서 분리되어 있고, 한 calibration prefix 안에서 specimen은 중복되지 않는다. v50 calibration bank를 재사용하므로 이 반복 실험이 물리적으로 새로운 surface를 추가하지는 않는다. 양쪽 정책이 정확히 같은 raw prefix를 읽었다.

Likelihood 방식은 두 displacement의 부호와 크기를 v50 sensor mixture likelihood로 읽는다. Vote 방식은 같은 dead zone(weak 0.018 mm, full 0.120 mm)을 통과한 쌍 가운데 부호 일치와 반전의 수만 센다. 단일 prefix에서 vote의 posterior mean은 (일치 수+1)/(유효 쌍+2)다. Joint posterior의 경우, 교체 전 관측 우도를 L₀(κ), 새 우도를 L₁(η)로 두면 η에 대한 미정규화 밀도는 [0.75 L₀(η)+0.25 L₀(1−η)]L₁(η)다. 두 정책 모두 이 밀도를 16-point Gauss–Legendre quadrature로 적분한다. P(drift|측정)은 두 항 중 두 번째의 정규화된 적분값이다. Joint posterior의 η 평균에서 q=0.1+0.8 E[η|측정]을 얻어 weak-to-impact 방향 유지율을 예측한다. [정책 코드](policy52.py)

세 후보 중 keep은 교체 전 관측만 사용한 standalone posterior mean으로 p(d)와 p(−d)를 섞고, refit은 교체 후 관측만 같은 방식으로 사용한다. Unsigned는 |d|를 사용하는 기존 source-trained predictor다. 세 action의 값은 v47 source 12개 catalog에서만 계산했다. True weak-to-impact retention을 0과 1로 둔 source conditional Brier를 fork해, signed blend 계수 a의 Brier를 C₀+C₁a+C₂a²로 적산했다. Unsigned는 별도 risk 한 개다. 새 session에서는 두 endpoint의 coefficients를 q로 선형 보간한 후 세 후보 중 최소 expected source Brier를 고른다. K=0이면 refit 선택지를 제외한다. Likelihood와 vote는 같은 prior, risk table, keep/refit/unsigned 선택지를 갖는다. 독립 post-only vote soft는 세 후보 중 선택하는 절차도 생략하는 대조다. Nominal prior와 source risk의 적합성은 학습한 물리적 보장이 아니다. v47에서 predictor 자체도 학습했으므로 source risk의 과적합 위험은 별도로 남는다. [Source risk](source_risk52.json) · [계수 계산](source52.py)

| 교체 후 K | Likelihood drift AUROC | Vote drift AUROC | Likelihood false positive | Likelihood true positive |
|---:|---:|---:|---:|---:|
| 0 | 0.500 | 0.500 | 0.00% | 0.00% |
| 1 | 0.687 | 0.683 | 1.08% | 8.41% |
| 2 | 0.742 | 0.737 | 1.83% | 17.64% |
| 4 | 0.779 | 0.774 | 2.92% | 33.20% |
| 8 | 0.803 | 0.800 | 3.54% | 47.32% |

AUROC는 catalog와 교체 전 κ를 같게 둔 채 4개 환경×256개의 보존 prefix와 같은 수의 반전 prefix를 비교한 뒤 동일 가중 평균했다. True/false positive는 posterior drift 확률이 0.5를 넘는지로 결정했다. 세 action을 고르는 정책은 이 진단 cutoff 대신 source risk를 사용한다. K=4에서 κ=0, .25, .5, .75, 1일 때 likelihood AUROC는 각각 .992, .709, .500, .701, .992였다. κ=.5이면 보존과 반전이 같은 관측분포를 생성한다. 이때 unsigned 선택률은 두 event에서 모두 61.53%다. κ=1이면서 뒤집힌 session에서 keep/refit/unsigned 선택률은 0.03%/70.57%/29.39%였다. 교체 후 새 증거가 약해서 다시 추정하는 것보다 방향 자체를 포기하는 경우도 있었다. [전체 action 비율과 탐지 결과](calibration52.json)

Frozen fit은 node55 allocation 2744468의 Slurm CPU에서 실행했다. 모든 K에서 standalone mean은 기존 likelihood 또는 명시적인 Beta vote 계산과 2×10⁻¹² 이내로 일치했다. K=0의 P(drift)는 모든 session에서 prior였고 refit 선택은 0건이었다. [Source·정책 freeze](policy_freeze52.sha256) · [12개 action bank checksum](policy_results52.sha256)
