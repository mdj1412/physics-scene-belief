# v60 · 광학 동기화 표식이 복원한 정보

Impact clip의 20%를 2 ms 늦게 시작했을 때, 같은 8회 impact에서 읽힌 reference–impact mismatch의 오류율은 보정 전 **14.39%**, optical marker 보정 후 **0%**였다. 두 방법 모두 읽을 수 있었던 관측만 남겨도 보정 전 **32,163/223,712건(14.38%)**에서 틀렸고 보정 후에는 같은 관측에서 틀린 것이 없었다. Optical marker가 보이지 않으면 보정기가 판독을 포기했고, 2 ms pre-trigger buffer 범위에서만 원래 구간을 되찾았다. [광학 실험 원자료](vision60.json) · [공통 판독 집합 분석](vision_controls60.json)

24개 fresh catalog마다 2개의 material과 각 material당 512개의 독립 4×2 pilot을 만들었다. 각 pilot에서 서로 다른 8개 물체에 reference와 impact를 하나씩 가하고, 같은 paid contact로 sham과 reset, 3개 trigger 지연 비율, 보정 전후를 평가했다. 물체 쌍의 순서와 active 물체(확률 50%)를 독립적으로 뽑았으며 순서가 역전된 비율은 50.23%, 두 번째 전자센서가 reset target으로 뽑힌 비율은 49.95%였다. 실제 phase가 2 ms 늦게 들어간 impact는 20% 조건에서 19.99%였다. [생성기](vision60.py) · [동결 시각](vision_plan60.utc)

광학 채널은 1,000 fps, 15×27 pixel, 8-frame ring buffer다. 물리 이벤트에 맞춰 0 ms에서 한 frame 동안 독립 LED가 켜지도록 했다. 상단 13행에는 변형 표식과 fixture 표식을 두었고, 하단 2행에는 동기화용 LED와 인접 background ROI를 두었다. ROI 둘의 평균 밝기 차가 38 grayscale level보다 큰 단일 frame을 anchor로 쓴다. 그 시점부터 여섯 장이 확보되어야 하며 bright marker 두 개의 상대적 변형이 0.5 pixel 이상이어야 부호를 읽는다. 보정 전 reader는 trigger가 정시에 왔다고 가정해 buffer의 세 번째 frame에 고정한다. 두 reader 모두 pixel에만 접근한다. Camera 장면에는 subpixel 진동과 pixel noise, 7.5% tip marker 가림, 약한 변형, 4% LED 가림이 있다. Reference와 impact의 광학 결측은 각각 일어날 수 있다. [사전 계획](VISION_PLAN_KO.md) · [Reader 및 Renderer](vision60.py)

| 2 ms 늦은 impact 비중 | 판독 방식 | Active 중 부호 두 개를 읽음 | 판독한 mismatch 중 오판 |
|---|---|---:|---:|
| 0% | 고정 창 | 260,725 / 393,768 (66.21%) | 0 / 260,725 |
| 0% | LED 보정 | 239,841 / 393,768 (60.91%) | 0 / 239,841 |
| 20% | 고정 창 | 243,438 / 393,768 (61.82%) | 35,033 / 243,438 (14.39%) |
| 20% | LED 보정 | 239,841 / 393,768 (60.91%) | 0 / 239,841 |
| 100% | 고정 창 | 172,610 / 393,768 (43.84%) | 172,610 / 172,610 (100%) |
| 100% | LED 보정 | 239,841 / 393,768 (60.91%) | 0 / 239,841 |

각 분모에는 같은 specimen과 noise를 대응시킨 물리 가설(P/C)·환경(A/B)의 네 counterfactual이 포함된다. 독립 surface는 48개, bootstrap 단위는 24개 catalog다. 물리적으로 inactive인 표본이 false mismatch로 통과한 경우는 0건이었다. 정시 조건에서 LED 보정은 기존 판독이 맞았던 20,884건을 marker 결측으로 포기했다. 20% 지연에서 같은 판독 집합으로 제한한 오류율의 보정 후 차이는 **−14.38 pp** [24-catalog paired bootstrap 95% CI −14.69, −14.06]였다. 정시와 2 ms 지연에서 두 buffer는 모두 절대시간 0~5 ms를 포함하고 같은 pixel과 noise를 재사용한다. 이에 따라 보정 결과가 3개 지연 조건에서 동일했다. [공통 집합과 케이스 수](vision_controls60.json)

정시와 2 ms 조건에서 서로 다른 reference·impact 393,216개 중 377,192개(95.92%)에서 marker가 읽혔고, 검출된 위치의 오류는 0개였다. Marker 패치를 실제 background pixel로 덮은 raw fixture에서는 보정된 부호가 전부 결측이 됐다. Gate가 보지 못한 frame을 recovery에 사용하지 않는지 확인하려고 4 ms 늦은 조건을 추가했다. 이때 이벤트는 ring buffer에 들어오기 전에 끝났고, corrected reader가 양쪽 contact를 모두 읽은 쌍은 0이었다. 이 검증은 camera clock correction이 buffer에 담기지 않은 시점을 생성할 수 없음을 보여준다. [Buffer·marker 음성 대조](vision60.json)

마지막으로 2 ms 늦은 raw timing fixture에서 물리 event에 대한 LED의 발광 자체를 추가로 0, 1, 2 ms 늦췄다. 이 사후 대조의 784개 active 관측 중 양쪽 contact를 판독한 건수는 456, 256, 300이었고 잘못 읽은 건수는 각각 0, 0, 300이었다. 1 ms lag에서는 약한 변형이 대거 결측으로 들어갔다. 2 ms lag에서는 body가 제자리로 돌아오는 방향을 읽었다. 이 값은 설정한 pulse, 단순한 marker 장면과 estimator에 국한되며 실제 camera 허용 clock 오차나 장비별 정확도의 추정치가 아니다. [스푸리어스 동기화 대조](vision_controls60.py) · [결과](vision_controls60.json)

Primary에서는 local latch와 common fault의 발생, camera trigger의 지연을 다른 random draw로 생성했다. A/B의 평균 true mismatch q는 0.29다. 물리적 지속 가설은 v59의 A/B별 second moment를 주고, common-mode 가설은 실제 물체의 mismatch를 독립으로 유지한 채 A/B 각각에 29.58% 또는 4.08%의 upstream fault를 준다. 동일 전자 기록을 만들 수 있는 P·C counterfactual을 짝지어 측정 오차와 원인 차이를 분리했다. 같은 pilot을 실제 배포 손익과 어떤 관계로 조합하는지에 따라 aligned, independent, reversed, constant high/low를 평가할 수 있다. 고정되지 않는 값은 새 물체·surface 자료에서 결정할 수 없으므로 이를 표면별 fault 빈도 자체의 실측으로 해석할 수 없다. [Pilot 입력과 인코딩](vision60.py)

32개의 유료 광학 판독(두 surface ×8개 물체 ×reference·impact)을 실제 pixel부터 다시 읽는 GPU timing을 위해 catalog별 첫 episode의 raw frame을 보관했다. 나머지 모든 episode는 유료 sensor sign, audit truth, camera 판독, selector 입력 code, 지연 배정과 generator seed를 보관해 pixel을 재현할 수 있다. Raw fixture로 전체 delay·sync arm의 camera sign을 다시 계산하고 archive와 일치함을 확인했다. 입력 데이터·코드·통계·사전 계획의 digest는 서버와 local mirror에서 확인했다. [재현 checksum](vision_bundle60.sha256)
