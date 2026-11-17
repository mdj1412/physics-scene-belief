# v68 · 외부 PyBullet의 calibration부터 terminal까지의 보조 검정

독립 compliant 및 prescribed primary의 판정과 별도로 외부 엔진을 처음부터 적용한다. 원고가 이미 알고 있던 v67의 두 법칙, 두 형상, 4개 source axis와 head, v38 four-pair 보정을 유지한다. 12개 기존 catalog 모두, 4개 환경, 환경별 128개 후보로 구매를 결정한 뒤, 순서상 첫 8개 prototype만 평가한다. 이 subset은 외부 terminal label에 사전 지정한 것과 같다. 48개 조합 전부, 두 반지름 및 세 위치, 네 measurement repeat를 균등 평가한다. Primary forecast는 이미 실행 중이다. External subset의 terminal 수치를 이 보조 검정 설계자는 아직 열지 않았다. 다른 프로세스가 terminal label을 미리 생성할 수 있어 이 검정의 독립성은 별도 prediction code·관측 산출물에 국한된다.

외부 solver는 v44의 PyBullet rigid sphere/mesh (회전 고정, 0.10mm mesh, 20μs physics step, 0.80ms horizon)이다. Calibration 및 deployment에서 모든 실제 접촉 속도는 두 새 law 각각으로 계산한다. v68과 동일한 미리 고정한 Gaussian noise를 더한다. 움직임을 관측하지 않는 동일 geometry-only channel은 재사용한다. 구매 이전에 후속 물체 128개의 factual response를 구한다. Query를 고른 경우 profile 이외의 sensor는 NaN으로 가린다. Terminal label은 기존 외부 engine의 같은 prototype 8개를 사후 채점한다. Calibration optimizer와 GPU checkpoint를 고정하고 추가 학습하지 않는다.

사전 방향은 law마다 Wendland soft−full<0, Dispersive soft−full>0, 두 형상 평균 restored−dropped<−0.002이다. 여섯 catalog paired mean에 대응하는 비교(표본 단위 12개 catalog)에 Bonferroni 동시 95% 구간을 낸다. 새 scope는 secondary다. 기존 compliant-history·PyBullet-terminal와 같은 표본에서 Brier contrast도 비교한다. 실패·종료 contact 수, valid 분모 및 code/input/forecast 해시를 보존한다.
