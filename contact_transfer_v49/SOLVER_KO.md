# v49 독립 contact solver 검산

PyBullet 3.2.7의 DIRECT mode를 사용했다. 물체의 full-load contour를 3차원에 돌출시켜 triangle mesh를 만들고, 회전이 구속된 sphere를 해당 표면으로 발사했다. Bullet의 collision detection이 접촉점을 찾고 sequential impulse solver가 restitution과 Coulomb friction을 적용했다. 첫 positive normal impulse에서 구한 local tangent로 outgoing velocity를 투영하여 0.020 m/s 이하를 stick으로 정의했다. 회전이 가능한 자유 물체에서는 이 값이 바뀔 수 있다. [Solver](solver49.py) · [Freeze](solver49.sha256) · [실험 전 protocol](SOLVER49_PROTOCOL.md)

V48은 contact patch에 pressure impulse를 지정하고 남은 tangential speed를 한 식으로 계산했다. V49는 같은 contour와 물체 parameter를 받아 contact를 다시 풀었다. 따라서 동일한 0.25/2 mm가 v48에서는 pressure footprint radius, v49에서는 충돌하는 sphere의 외부 반경이다. Surface와 onset rate에 따른 마찰 법칙도 새로 정의했다. Weak-load 및 full-load의 변형 모드 자체는 지정된 입력이다. Bullet은 preload와 full-load 사이에서 변형 방향이 어떻게 바뀌는지를 탄성 방정식으로 예측하지 않는다.

12개 catalog에 884,736개 binary label을 저장했다. 네 family의 inert contour가 접촉 범위 전체에서 정확히 같아 동일한 결과를 재사용한 뒤 primary solver를 실제 호출한 횟수는 663,552회다. 모든 case에서 접촉 impulse 뒤 분리 방향으로 움직였다. 세 상태의 실제 weight로 계산한 stick 비율은 51.78%였다. Negative state에서는 68.13%, positive에서는 32.36%였다. 같은 물체에서 변형 방향을 반전시키자 35.83% [95% CI 34.50, 37.16]에서 label이 달라졌다. PyBullet과 v48 pressure solver는 3.91% [3.71, 4.11]에서 서로 다른 label을 냈다. [결과](solver49_audit.json)

| Resolution | Time step (ms) | Mesh edge (mm) | Solver iterations | 첫 impulse 뒤 관찰 시간 | 전체 weighted stick |
|---|---:|---:|---:|---:|---:|
| Primary | 0.025 | 0.050 | 40 | 0.050 ms | 51.78% |
| Fine | 0.0125 | 0.025 | 80 | 0.025 ms | 50.07% |
| Ultra | 0.00625 | 0.0125 | 160 | 0.0125 ms | 46.55% |

평판에서는 두 tip radius, 네 friction 값, 세 기울기와 두 restitution 값의 48개 조합을 두 resolution에서 반복했다. 평판 analytic tangential·normal 결과와의 최대 차이는 각각 1.11×10⁻¹⁶ m/s였다. 곡면에서는 resolution sensitivity가 확인됐다. 사전 지정한 1/16 subset에서 primary와 fine의 label 차이가 2.22%여서, fine과 ultra도 전체 물체에 확장했다. 전체 label의 primary↔fine 차이는 2.13%, fine↔ultra는 3.61%, primary↔ultra는 5.56%였다. 이렇게 구한 세 결과만으로 curved-contact label의 numerical convergence를 주장할 수 없다. 숫자는 서로 짝지은 support의 실제 state prevalence로 가중했다. [Flat 검산](solver49_flat.json) · [Fine 검산](fullref_audit49.json) · [Ultra 검산](clock_audit49.json)

최초 impulse 뒤 출력 시간을 맞추려고 사전 선택 subset에서 ultra를 0.050 ms까지 더 적분했다. 시간을 맞춘 초세분 결과와 primary의 label 차이는 5.67%였다. 원래 초세분 시점과 primary에서는 5.75%였다. 초세분 설정에서 출력 시간만 바꾸었을 때 label은 0.076%에서 바뀌었다. 이 대조는 출력 시점만으로 resolution sensitivity를 설명하기 어렵다는 근거다. 실제 curved contact를 어느 설정이 더 잘 근사하는지는 독립 실측이 필요하다.

서버 work directory의 `labels49`, `labels49_fine_all`, `labels49_ultra_all`에 같은 12개 catalog의 full outcomes를, `labels49_fine`, `labels49_ultra`, `labels49_ultra_clock`에 subset outcomes를 기록했다. `object_slots`로 대조했을 때 fine/ultra full run과 각 subset replay의 저장 속도가 정확히 일치했다. [Source 파일](solver49_fullref.py) · [Fine/ultra 비교](fullref_audit49.py) · [관찰 시간 비교](clock_audit49.py)
