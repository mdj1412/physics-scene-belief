# v39 target catalog와 geometry의 사전 명세

두 shape를 query height와 마지막 probe height에서의 contour slope로 v38 shape에 대응시켰다. 이 절의 숫자는 설계한 곡선의 도함수 값이다. v39 물체의 contact outcome이나 noisy sensor를 생성·열람하기 전에 계산했다.

## 새로운 두 형상의 정의

높이 y를 m로 나타내고 u=y/0.025로 둔다. 기본 profile의 위치·기울기·곡률에 아래 profile 변위를 더한다. Local contact normal을 만들 때 같은 변위의 도함수를 쓴다.

| Shape | 추가 profile 변위 | 추가 slope d(profile)/dy | y=23 mm에서의 추가 slope | y=25 mm에서의 추가 slope |
|---|---|---|---:|---:|
| v38 shoulder | 기존 rational shoulder | 0.072u²/(1+u²) | +0.03301 | +0.03600 |
| **v39 shelf** | 0.025·0.043·(u−sin(ku)/k), k=π/2 | 0.043·(1−cos(ku)) | **+0.03761** | **+0.04300** |
| v38 bead | 기존 Gaussian contour | −0.078u·exp(−u²/(2·0.80²)) | −0.03704 | −0.03571 |
| **v39 ridge** | −0.025·0.153·u²/(2·(1+u²)) | −0.153u/(1+u²)² | **−0.04129** | **−0.03825** |

두 높이에서 추가 slope로 이차원 descriptor를 만들고 Euclidean distance가 작은 v38 shape를 대응시켰다. Shelf에서 shoulder와 bead까지의 거리는 각각 **0.00838, 0.10848**이며 ridge에서 shoulder와 bead까지는 **0.10504, 0.00495**다. 따라서 shelf는 shoulder에, ridge는 bead에 대응된다. 이 대응은 추가 slope가 local tangent와 normal impulse에 들어간다는 물리적 동기를 갖는다. 전체 profile, sensor에서 관측되는 geometry, 방향별 slip 신호, decoder error의 동일성을 보장하지는 않는다. v38의 shape별 기본 curvature 배율 0.97·1.03도 v39에서는 공통 1로 고정했다. 두 물체 형상이 가깝다는 이유로 optimizer 성능도 가까울 것이라는 부분이 검증할 가설이다.

## Contact law와 counterfactual pairing

Weakening/strengthening의 static threshold와 kinetic coefficient, near-field 강도의 공간적 변화, noise는 v38 함수를 그대로 사용한다. 네 cell은 두 형상 × 두 법칙이다. v39에서 새롭게 일반화시키는 대상은 catalog와 형상이다. Contact-law family까지 새로 일반화했다는 주장은 이 설계에서 검증하지 않는다. 첫 세 접촉은 0 mm에서 부호를 교대하고, 마지막 접촉은 23 mm·+1.10, query 충돌은 25 mm·+0.425다. 같은 cell·catalog·block에서는 두 방향과 세 강도에 동일 prototype draw와 sensor noise를 사용한다. 여섯 조건을 독립 표면 여섯 개로 계산하면 불확실성을 과소평가한다.

## 독립 catalog와 시험 단위

네 cell마다 catalog replicate 12개를 독립 생성한다. 매 replicate 안에서도 고정된 여덟 개 α와 calibration/deployment split별로 2,048개 latent prototype을 서로 다른 seed에서 생성한다. 총 **48개 독립 cell–catalog replicate**와 **768개 catalog–α–split support**, 1,572,864개의 예정 prototype row가 있다. 각 catalog 안에서 물체는 복원추출한다. 두 block의 다른 첫 접촉 방향을 짝지어 catalog 내에서 평균한 후 independent catalog를 추론 단위로 삼는다. 시간과 정확도의 반복 횟수는 구분한다.

Catalog seed는 1,939,100,000 + 1,000,000 × cell + 10,000 × catalog_id + 100 × alpha_index + lane (calibration 10, deployment 20)이다. v38 support seed 1,838,210,000와 다른 namespace다. 이 공식은 cell 자체도 다르면 독립적으로 catalog를 다시 생성한다. 같은 catalog·환경 안에서 observation seed와 prototype index 추출 seed는 별도이며, 실제 latent state, hidden sign, tier, 뒤에 읽을 test label은 정책에 전달하지 않는다. [구현](physics39.py)
