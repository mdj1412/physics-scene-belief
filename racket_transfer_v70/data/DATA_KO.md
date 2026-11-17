# v70 · Racket impact 원자료와 holdout 감사

## 어느 rubber를 독립 단위로 삼았나

Thomas Gossard의 공개 Git revision `159c6f040dd7832570b537b3b5fd00a97fc28c15`(2026-06-07)을 고정했다. 상업적 rubber 제품 하나의 모든 configuration을 한 fold에 배정했다. 동일 제품이 blade나 sponge 두께만 달리해 양쪽에 걸치는 일을 막는다. 원 논문은 각 configuration 내부에서 80/20으로 나누어 학습·평가했으므로 그 수치를 이 protocol의 test score로 사용할 수 없다. [원문 §III, Tables II–IV](https://arxiv.org/html/2604.11349v1) · [고정한 저장소](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

| Held-out rubber | Surface | Configuration (blade, sponge mm) | 원본 impact | 저자 train+test에 포함 |
|---|---|---|---:|---:|
| andro Hexer Duro | Inverted, offensive | 01 offensive 2.1, 03 defensive 2.1, 09 allround 2.1 | 2,534 | 2,528 |
| andro Power 3 | Inverted, allround | 02 offensive 1.8, 04 defensive 1.8 | 1,609 | 1,603 |
| Tibhar Grass D.TecS | Long pips | 05 allround 1.2, 06 allround 0(OX) | 1,546 | 1,494 |
| Dr. Neubauer Diamant | Medium pips | 07 allround 1.2 | 845 | 842 |
| andro Blowfish | Short pips | 08 allround 2.0 | 859 | 856 |
| Dr. Neubauer A-B-S II soft | Anti-spin | 10 allround 2.1 | 791 | 790 |

세 blade는 offensive Donic Holz Original Carbospeed, defensive Tibhar Holz Defense Plus, allround DONIC Appelgren Allplay Senso V1이다. 이 설계에서 Hexer Duro를 holdout해도 다른 종류의 inverted rubber인 Power 3는 source에 남는다. 각 pips 길이·anti-spin은 한 제품뿐이며, six-product holdout 결과로 제조사·rubber 계열 전체의 분산을 추정할 수 없다. Rubber, sponge, blade, 제조 편차를 독립적으로 cross한 실험이 아니므로 모델별 차이를 특정 성분의 효과로 귀속할 수 없다. [공식 configuration 표](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/README.md)

## 원래 split과 중복

공개 CSV 총 30개의 header·numeric row를 확인했다. `original`은 8,184건이며 이 중 6,490건이 고정 `train`, 1,623건이 고정 `test`에 정확히 대응한다. 이 두 파일 집합은 configuration 내에서 겹치지 않는다. 원 논문에 적힌 취득 건수는 8,194이다. 원본 CSV의 71건은 공식 train·test 양쪽에서 빠졌고 ID 06에서 42건, ID 05에서 10건이다. 두 폴더의 membership을 통해 좁힌 차이이며 개별 제외 사유는 문서에 없다. 71건을 invalid 측정으로 가정하면 특정 long-pips configuration이 과도하게 영향을 받는다. [원본 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15/data) · [행별 공식 membership](official_membership.csv)

8,184개 모두 14개 수치가 finite이다. Numeric 14열 전체, 운동학 12열, 연속형 incoming 6열의 exact duplicate는 각각 0건이다. `(racket_angle, traj_dir)`는 한 configuration 안에서 고유하다. 그러나 전체에서 서로 다른 키 1,314개 중 1,141개가 여러 configuration에서 반복된다. 다른 rubber가 같은 번호를 사용했다는 사실을 동일한 물리 trial의 증거로 보기는 어렵다. 독립 session·물리적 ball serial·wear 순서는 CSV에서 검증할 수 없다. [재현 코드](audit.py) · [원본별 SHA-256과 범위](audit.json)

## 속도·spin의 측정과 좌표계

논문 §III에 따르면 고정된 racket으로 공을 발사해 370 fps 단일 high-speed camera로 2D trajectory를 촬영하고 중력을 보정한 50 ms 구간의 linear regression으로 impact 전후 속도를 추정했다. Spin은 표식이 있는 공에 SpinDOE를 적용해 얻었다. 따라서 충돌 직후라는 표현은 이탈 부근의 추정 속도를 뜻한다. CSV의 강체 (x,z) 이동과 (y)축 spin이 공식 benchmark의 endpoint이다. 저자 구현은 각 impact의 2D velocity error norm을 평균하고, y축 spin의 absolute error를 따로 평균한다. Racket angle 값은 0, 10, 20의 세 설정이고 CSV 전 행에서 (v_y^-=v_y^+=0), (v_z^-<0<v_z^+)이다. Spin의 x·z 성분에는 영이 아닌 값들이 있다. [측정 절차](https://arxiv.org/html/2604.11349v1#S3) · [저자 metric 구현](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/python/racket_dynamics/evaluation/metrics.py)

저자 코드에서 (R=0.020) m, (m=0.0027) kg, (I=2mR^2/3)이고 (v)는 m/s, (ω)는 rad/s로 사용한다. Racket-local (z)를 normal로 사용하는 코드 convention에서 구의 최하점 위치는 (-R e_z), 접촉점 속도는 (v+ω×(-R e_z))다. 두 tangential components는 (v_x-Rω_y, v_y+Rω_x)이다. v69 golf 영상의 spin 부호와 다르다. 원 논문의 Eq. 3과 실행 코드에서 첫 항 부호가 다르게 표기된다. Eq. 9와 저자의 `get_surface_v` 및 `elastic_bounce`가 서로 일치하므로 코드의 부호를 따랐다. Metadata인 `racket_angle`을 이용해 명목 normal을 한 번 더 돌리는 것은 공식 loader의 처리와 다르다. [상수](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/python/racket_dynamics/physics/constants.py) · [Impulse와 contact speed](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/python/racket_dynamics/physics/bounce.py) · [Loader](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/python/racket_dynamics/data/dataset.py)

## 새 rubber의 입사조건은 source 범위와 얼마나 겹쳤나

각 fold에서 incoming ((v_x,v_z,ω_y))만 사용하고, source 평균·표준편차로 세 변수를 표준화했다. Source convex hull 밖에 있는 target 수와 source 중 가장 가까운 다섯 번째 shot까지의 Euclidean distance를 계산했다. Distance 기준값은 source 내부에서도 자기 제품을 빼고 구한 5-nearest-neighbor distance의 95% quantile이다. Inner reference는 한 제품이 더 빠져 후보 수가 줄어든다. 이 진단은 target overlap의 기술통계이며 동일 분포에 대한 통계검정은 아니다. [서버 코드](support.py) · [모든 행의 판정](support_rows.csv)

| Held-out rubber | Target 수 | Source convex hull 밖 | Source cross-rubber distance q95 초과 |
|---|---:|---:|---:|
| Hexer Duro | 2,534 | 39 | 75 |
| Power 3 | 1,609 | 19 | 46 |
| Grass D.TecS | 1,546 | 20 | 89 |
| Diamant | 845 | 13 | 17 |
| Blowfish | 859 | 6 | 10 |
| A-B-S II soft | 791 | 2 | 9 |
| 전체 | 8,184 | 99 (1.21%) | 246 (3.01%) |

대부분의 shot에서 측정한 세 입사 변수의 source coverage는 충분했다. Source 안에서도 샘플 밀도와 spin–속도 결합은 다를 수 있고 racket mounting, temperature, ball wear 같은 숨은 조건은 점검할 수 없다. 이 계산은 datacenter3의 node56, Slurm allocation 2744471에서 GPU를 비활성화한 CPU step으로 수행했다. [계산 결과](support.json)

## 독립 contact patch와 재현 자료

원본 열은 설정 2개, 전후 spin 6개, 전후 COM velocity 6개다. 코드 저장소의 공개 자료에는 impact별 patch mask·pressure map·contact frame·patch 내 local slip label이 없다. 제공된 racket 제품 사진은 고정 configuration의 외관이다. 입사 state만으로 계산한 patch size는 같은 입력의 함수이므로 독립 실측 정보가 되지 않는다. 이탈 속도에서 역추정한 patch를 같은 trial의 이탈 예측에 넣으면 outcome leakage가 생긴다. 소수의 센서로 실제 patch를 취득할 때는 언제 사용 가능한지를 고정해 prediction과 post-impact reconstruction의 질문을 구별해야 한다. [공개 파일 목록](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

저장소 최상위에는 Thomas Gossard의 2026년 MIT License가 있으므로 공표된 license notice를 원문과 함께 보존했다. 논문 파일에는 별도의 CC BY-SA 4.0 표기가 있다. 원본 checkout에서 복사한 자료·라이선스는 `upstream/` 아래에 두었다. SHA-256은 30개 입력 파일별로 기록했으며 `python audit.py`와 `python support.py`로 감사 산출물을 다시 생성할 수 있다. [저장소 license](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/LICENSE) · [원문](https://arxiv.org/abs/2604.11349)
