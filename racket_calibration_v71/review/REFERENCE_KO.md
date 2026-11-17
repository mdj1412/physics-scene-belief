# v71 독립 검산과 finite-population detection 확률

## 계산과 정보 접근

각 outer fold에서 v70에 동결된 source-only gate, 두 expert와 표준화 계수를 읽었다. 원자료에서 `s_out = v_out_x − 0.020 × spin_after_y`를 다시 계산하고, 각 calibration 후보의 `s_out > 0`과 source gate 판독의 불일치를 집계했다. Calibration은 held rubber의 공식 train CSV에서 무복원 추출한다. 공식 test CSV는 동결된 두 가지 예측의 최종 평가에만 사용한다. 이 절의 전수 집계는 실험 후 평가다. Deployment detector가 보는 정보는 해당 draw에서 지불한 prefix로 제한된다.

Calibration pool의 크기를 N, source gate와의 불일치 수를 M, 지불한 impact 수를 K라고 두면 불일치 개수 D는 `Hypergeometric(N, M, K)`를 따른다. `P(D > K/2)`를 조합의 합으로 정확히 계산했다. 동률이면 기존 routing을 유지한다. 두 routing에서의 held loss를 L₀, L₁로 두면 해당 K의 gate-only loss 기대값은 `(1−p)L₀ + pL₁`이다. 이 식은 각 draw가 같은 held set에 동일한 두 예측 중 하나를 전달하기 때문에 성립한다. Expert 계수를 새로 학습하는 방법에는 해당하지 않는다. [원자료 및 공개 protocol](https://arxiv.org/html/2604.11349v1) · [코드](finite_reference.py)

추가 evidence 기준은 사전에 정한 K에서 `P(Binomial(K, 0.5) ≥ D) ≤ 0.05`이다. 이 기준을 넘을 확률 역시 실제 pool의 `Hypergeometric` 분포로 계산했다. 한 번의 고정 K에 대한 binomial 기준과, 지정한 유한 pool에서 detector가 행동할 확률의 차이를 유지한다. 여러 K를 보다가 첫 통과 때 멈추면 별도의 sequential error control이 필요하다. 제품 간 독립성, 동일 제품의 다른 날·시편에서의 재현성은 이 전수 계산의 가정 밖에 있다.

## Rubber별 결과

`mismatch`의 분모는 각각 해당 제품을 구성하는 모든 configuration의 공식 train/test impact다. 마지막 두 열은 K=8에서 calibration sample을 무작위로 다시 뽑을 때 각 기준을 넘을 정확한 확률이다. [전체 결과](finite_reference.json)

| Target rubber | Pool mismatch | Held mismatch | Majority flip 확률 | 0.05 evidence 확률 |
|---|---:|---:|---:|---:|
| Hexer Duro | 41/2,022 | 15/506 | 0.0000143% | <0.000001% |
| Power 3 | 27/1,282 | 8/321 | 0.0000151% | <0.000001% |
| Grass D.TecS | 1,111/1,195 | 271/299 | 99.8718% | 89.6335% |
| Diamant | 30/674 | 8/168 | 0.000636% | <0.000001% |
| Blowfish | 18/685 | 2/171 | 0.0000369% | <0.000001% |
| ABS II Soft | 310/632 | 76/158 | 34.1749% | 3.0390% |

## Grass D.TecS의 예산별 행동과 오차

| K | Majority flip | 0.05 evidence | Gate-only expected velocity error (m/s) | Gate-only expected spin error (rad/s) |
|---:|---:|---:|---:|---:|
| 0 | 0% | 0% | 1.5841 | 119.104 |
| 1 | 92.9707% | 0% | 0.7328 | 46.098 |
| 2 | 86.4301% | 0% | 0.7927 | 51.234 |
| 4 | 97.3300% | 0% | 0.6929 | 42.675 |
| 8 | 99.8718% | 89.6335% | 0.6696 | 40.679 |
| 16 | 99.9997% | 99.6407% | 0.6684 | 40.579 |
| 32 | >99.99999% | 99.9996% | 0.6684 | 40.578 |

K=2에서는 두 번 모두 불일치해야 majority flip이 일어난다. K=1에서 K=2로 갈 때 생긴 하락을 이 동률 규칙이 설명한다. K=4라면 전부 불일치해도 binomial tail은 1/16이다. K=8에서 관측된 256 draw 중 255회에 flip, 227회에 0.05 evidence가 들어왔다. 표의 확률은 모든 가능한 동일 크기 표본에 대한 값이다. 두 configuration의 pool mismatch는 531/603, 580/592이고 held mismatch는 각각 125/151, 146/148이다. 이 수치는 해당 두 configuration에서 같은 방향이 나타났음을 지지한다. 이 제품은 v70에서 이미 반전 사례로 지목됐다.

## 연속 예측 최적화와 부호 검출의 차이

ABS II Soft의 held 표본에서는 source gate와 76/158가 다르다. 여전히 원래 routing을 따른 평균 velocity·spin 오차는 2.1342 m/s·115.492 rad/s이고, 전부 뒤집은 routing은 1.5167 m/s·57.418 rad/s이다. 거의 반반인 mismatch frequency가 전문가 선택으로 인한 **오차의 크기**를 표현할 수 없음을 보여준다. 특히 잘못 고른 expert의 손실은 impact마다 다르다. 이 비교는 held label을 전부 본 평가자의 고정 정책 진단이다. 새 제품에서 미리 얻을 수 있는 calibration 성능으로 해석하려면 paid prefix만으로 어느 쪽의 예상 손실이 작은지 별도 검증해야 한다.

## 재현성 검산

서버 `node56`, Slurm job `2744471`의 step `290`에서 표의 정확한 확률·기대값을 계산했다. 다른 step `300`에서 공식 calibration 행과 독립 구현이 생성한 256개 무복원 순열을 사용해 GPU 결과를 재현했다. 여섯 제품·일곱 K에서 10,752개 prefix의 불일치 수, majority flip, 0.05 evidence와 held sign accuracy가 모두 맞았다. 각 draw에서 gate-only velocity 및 spin 오차의 최대 절대 차이는 각각 1.33×10⁻¹⁵ m/s, 7.11×10⁻¹⁴ rad/s였다. 검산은 CPU에서 실행했으며 GPU latency 측정에 포함되지 않는다. [검산 코드](check_draws.py) · [검산 결과](draw_audit.json)
