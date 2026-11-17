# v63 영상 판독과 독립 검증 센서

## 새 board와 반사 표면

새로 생성한 24개 catalog에 두 표면씩 두고, 48개 표면 각각에서 8개 새 board를 모두 평가했다. 표면당 256 pilot, board당 32 pilot이다. 한 pilot의 8개 물체 각각에 2.0 N reference와 impact를 한 번씩 가했다. 따라서 물리적으로 별개인 impact slot은 98,304개다. 원인과 환경을 교차한 네 counterfactual을 같은 난수로 만들었고, 네 tracking 조건에서 같은 scene·noise를 다시 사용했다. 이 반복을 독립적인 물리 표본으로 세지 않았다. 기존 여덟 board와 parameter vector가 일치하는 새 board는 없었다. [사전 명세](VISION_PLAN_KO.md) · [구현](optics63.py) · [동결 기록](optics_freeze63.utc)

1,000 fps의 8개 frame, 0.80 ms LED pulse, 0.25 ms shutter exposure에서 표면별 highlight 밝기·위치와 ghost 비중을 고정했다. 단발성 tracking error는 한 impact frame에서 위쪽 marker를 3 pixel 옮겼다. 지속적인 오인식에서는 표면의 반사가 fixture를 기준으로 원래 marker의 반대 방향으로 움직이게 하고 해당 상부 영상에서 ghost 비중을 표면별로 86–99%로 설정했다. 두 사건이 겹치면 ghost에 한 frame 교란을 더했다. 참 부호는 교란할 marker나 방향을 고르는 데 쓰지 않았다. 이는 특정 반사 교란의 stress test이며, 실제 표면의 반사 확률·광학 계수는 실측하지 않았다. [모델과 표면별 설정](optical_results63.json)

각 사건의 부여 확률은 독립적으로 0·35%였다. 두 확률이 모두 35%인 조건에서 실제 transient 사건은 34.967%, coherent 사건은 34.943%, 동시 사건은 12.054%였으며 두 이진 지표의 상관계수는 −0.0072였다. 가설·optical condition별 mirror는 이 집계에서 제외했다. LED와 camera의 지연 보정과 confidence cutoff는 v62와 같았다. [측정값](optical_results63.json)

## 독립 센서

표면 뒤의 differential strain pickup과 별도 ADC를 모사했다. 영상과 gauge는 같은 물리 방향과 변형 크기에 반응하지만, gauge는 화면의 marker·timing 또는 교란 상태를 읽지 않는다. 두 bridge leg에 반대 방향 strain이 입력되며, 표면별 기계적 coupling, 별도 board gain·양자화·noise·dropout을 적용했다. Pre/post 4개 sample 평균 차이가 5 ADC count 이상일 때만 부호를 읽었다. Gauge 파형은 독립 검산에만 사용했다. 보류 정책과 deployment는 이 파형, 참 부호 또는 crossing cell 이름에 접근하지 않았다. 계측의 독립성은 이 측정모델의 설계 속성이며 실물에서 두 센서가 환경에 함께 영향을 받는 정도는 측정하지 않았다. [센서 구현](optics63.py)

- 활성 접촉을 네 paired counterfactual로 집계하면 395,584건 중 339,622건(85.853%)에서 gauge가 부호를 읽었다. 읽은 부호 중 1,374건(0.405%)은 참 부호와 달랐다.
- 활성 reference–impact 쌍 197,792건 중 146,429건(74.032%)에서 gauge가 두 부호를 모두 읽었다. 그중 1,134쌍(0.774%)의 mismatch bit가 틀렸다. 비활성 물체 쌍 195,424건 중에도 noise로 5,064쌍(2.591%)에서 양쪽 부호가 출력됐다. 따라서 independent gauge도 오차와 결측이 함께 있는 검증 수단이다. [Sensor counts](optical_results63.json)

## 판독 수를 맞춘 광학 비교

같은 pilot에서 기존 v61 reader가 읽은 활성 쌍이 n개, frozen confidence가 남긴 쌍이 k개일 때, 균등 무작위 선택에서 각 쌍의 포함 확률은 k/n이다. 표의 random error는 이 확률로 모든 유효 판독을 적분한 기대값이다. Policy는 electronics가 비활성으로 판독한 쌍을 쓰지 않으며, 이 생성기에서는 electronics의 active mask와 실제 변형 active mask가 일치한다. Error의 분모는 해당 방법이 양쪽 부호를 읽은 활성 쌍이다. Gauge disagreement는 camera와 gauge 양쪽이 읽은 쌍의 mismatch bit가 다른 비율이다. [Audit 코드](audit_optics63.py)

| Transient / coherent 발생률 | V61 → confidence 판독률 | V61 error | Confidence error | 같은 수 random error | Confidence − random, pp [95% CI] | Gauge disagreement, confidence / random |
|---|---:|---:|---:|---:|---:|---:|
| 0% / 0% | 9.258% → 5.263% | 0.016% | 0.000% | 0.000% | 0.000 [0.000, 0.000] | 0.501% / 0.501% |
| 35% / 0% | 9.371% → 5.069% | 4.942% | 0.269% | 1.404% | −1.135 [−1.657, −0.569] | 0.749% / 1.603% |
| 0% / 35% | 9.091% → 5.136% | 33.178% | 33.931% | 33.541% | +0.390 [−0.456, +1.261] | 34.252% / 33.916% |
| 35% / 35% | 9.206% → 4.940% | 34.296% | 33.640% | 33.690% | −0.050 [−0.871, +0.760] | 33.896% / 33.896% |

CI는 24개 catalog와 8개 board를 각각 재추출한 5,000회 paired crossed bootstrap의 cell별 pointwise 구간이다. Clean 조건에서 기존 v61의 오판 3건은 셋 다 해당 pilot에서 k가 0인 경우였다. Confidence와 same-count random 모두 이 판독을 사용하지 않았다. Transient-only 조건에서 confidence는 10,026쌍 중 27개, 무작위 정책은 같은 수에서 기대 140.78개를 잘못 읽었다. Sensor로 함께 검증할 수 있는 쌍에서도 confidence의 오판독을 시사하는 불일치 비율이 낮아졌다. 반면 coherent error가 포함된 조건에서는 camera 판독의 약 3분의 1이 틀렸고 gauge도 비슷한 크기의 불일치를 냈다. [셀별 건수와 CI](optical_results63.json)

두 사건이 모두 가능한 cell을 실제 발생 사건으로 다시 나누면, coherent만 발생하여 판독된 쌍은 세 방법에서 모두 오판이었다. 같은 impact에서 두 사건이 겹쳤을 때는 판독된 쌍 중 기존 v61 83.38%, same-count random 94.48%, confidence 99.09%가 틀렸다. Temporal consistency를 강하게 요구하면 단발성 교란으로 깨진 ghost는 거르지만, 반대로 일관되게 움직이는 ghost는 계속 통과한다. 이 값은 판독된 부분집합의 광학 지표다. Held-out Brier loss는 별도 물체에서 후속 정책과 decoder를 실행하여 판단해야 한다. [사건별 집계](optical_counts63.npz)

## 재현

Datacenter3의 Slurm allocation 2744471에서 두 CPU worker로 simulation을 수행하고 CUDA를 숨겼다. 24개 fixture 전체에서 4개 cell × 3개 reader × paired 가설의 raw clip 판독과 저장된 code 입력이 같았고, raw gauge waveform의 재판독도 모두 일치했다. 자료는 전체 signed acquisition audit, catalog당 한 개의 raw pilot fixture 및 surface·board별 집계로 구성된다. [실행 명령](run_optics63.sh) · [Checksum](optics_bundle63.sha256)
