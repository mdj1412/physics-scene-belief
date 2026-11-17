## 어느 contour 정보가 실제 영상에서 측정됐나

접촉 전 96×32 grayscale frame 세 장을 읽었다. 비교하는 모든 정책이 v45의 rail fiducial·노출·노이즈·접근 궤적을 공유한다. 과거 v45의 source clip 221,184개를 다시 생성해 512개 clip 묶음의 raw pixel SHA-256, 추정 x를 v45 archive와 대조했다. 입력을 바꾸지 않고 source와 target에 같은 추정기를 적용했다. Target에는 이전에 쓰지 않은 terrace·dimple family를 렌더링했다. [입력 일치 검산 코드](capture46.py)

먼저 v45 방식으로 추정한 x와, 첫 frame에서 복원한 surface boundary를 사용한다. 카메라에서 읽은 배율로 x±2 mm의 9개 subpixel 높이를 추출해, 길이 1 mm인 일곱 secant의 기울기를 구한다. Surface tangent와 normal 각도는 상수만큼 차이 나므로 두 각도의 분산이 같다. 위치마다 arctan을 취한 뒤 일곱 각도의 population variance를 계산한다. 이 feature의 단위는 rad²다. 비교용 analytic reference는 진짜 x 주변의 같은 일곱 현을 사용한다. [추정기 및 geometry reference](camera46.py) · [측정·학습 사전 계획](VISION_PLAN_KO.md)

| 측정 | v45 source | 새 terrace | 새 dimple |
|---|---:|---:|---:|
| Clip 수 | 221,184 | 110,592 | 110,592 |
| x MAE (mm) | 0.09535 | 0.09314 | 0.09352 |
| 분산 MAE (rad²) | 0.04708 | 0.04825 | 0.04949 |
| 영상 분산과 analytic reference의 Spearman ρ | 0.665 | 0.672 | 0.645 |
| 영상 분산과 analytic derivative 기반 분산의 Spearman ρ | 0.659 | 0.670 | 0.645 |

Target의 2 family × 6 exact x 조건을 따로 계산해도 ρ는 0.467~0.679였다. 전체 target에서 측정 절대오차의 median은 0.00160 rad²였으나 p90은 0.17765 rad², 0.1 rad² 초과는 11.84%였다. 이에 비해 두 analytic 정의의 target ρ는 0.997이었다. 유한 구간 기울기와 미분값의 정의 차이만으로 설명하기 어려운 영상 측정 오차가 있다. 추정기는 추정 x와 pixel만 받으며 oracle 모형에만 실제 x와 geometry를 공급했다. [검산 자료](vision_audit46.json) · [구현](vision_audit46.py)

## Source에서 정한 선택이 새 형태에서 얼마나 달라졌나

같은 v45 finite-pressure target, source-axis decoder, 표본 가중치로 두 정책을 학습했다. 위치 정책은 가장 가까운 여섯 x의 평균 projection 손익을 계산했다. Joint 정책은 각 x에서 source variance의 quartile로 나누고, 해당 그룹의 평균 손익을 75%, x 전체 평균을 25% 반영했다. 어느 쪽도 target에서 튜닝하지 않았다. Source 전체의 refit x-only는 30.5와 31.0 mm에서만 projection을 선택해 기존 v45 규칙과 정확히 일치했다. Joint에서 달라진 것은 30.5 mm의 Q2와 31.0 mm의 Q1·Q2에서 original을 택하는 세 가지 경우였다. [Source lookup 및 각 bin 표본 수](router46.json) · [동결 SHA-256](router_freeze46.sha256)

각 위치에서 source 데이터의 Q1~Q4 비율은 약 25%씩이지만 target 분포는 달랐다. 추정 좌표가 30.5 mm에 속한 target에서 Q2 비율은 1.58%였다. 31.0 mm에 속한 경우 Q1+Q2 비율은 0.90%였다. 모든 여섯 x에서 source 1~99 percentile 범위를 벗어나는 target은 각 0.88~1.15%에 불과했다. 범위 밖인 물체는 적으나, 새 feature를 추가했을 때 결정을 바꿀 수 있는 물체도 적었다. [Source 구간 안의 target 비중](vision_audit46.json)

Leave-one-catalog-out으로 threshold와 표를 11개 source catalog에서 매번 다시 추정한 결과 joint−refit x Brier는 −0.000220이었다(12개 test catalog 평균의 95% CI [−0.000288, −0.000152]). Training split들이 겹치므로 이 구간은 source 성능의 기술적 요약이다. 한 source family 전체를 제외했을 때는 두 방향 모두에서 joint와 refit x가 정확히 같은 결정을 냈다. Ripple로만 학습하면 held-out fold에서 둘 다 original을 골랐고, fold로만 학습하면 held-out ripple에서 둘 다 Brier가 original보다 +0.000531 높았다. 범위가 겹치는 위치와 분산을 측정했다는 사실은 그 feature와 손익의 관계가 새 family로 옮겨간다는 충분조건이 아니다. [Source 내부 교차검증](source_cv46.json)

## 같은 물체를 다시 촬영했을 때

사후 검산으로 각 환경의 처음 32개 물체를 택해, 55,296개 target clip을 독립적인 배율·이동·노이즈·부분 가림으로 다시 렌더링했다. 두 촬영 간 분산의 Spearman ρ는 terrace 0.485, dimple 0.443이었다. Joint의 route 일치율은 98.33%·97.84%였고, refit x-only는 99.04%·98.43%였다. 정책이 자주 같은 route를 사용하는 점을 감안하면 route 일치율을 분산 측정의 정확도로 해석하기 어렵다. 두 촬영의 joint−x Brier 효과 차이는 −0.000014 [catalog 95% CI −0.000049, +0.000022]였다. 동일 시뮬레이터 안에서 camera nuisance를 바꾼 결과이며 실물 camera의 재현성은 별도 측정이 필요하다. [반복 측정](repeat_camera46.json) · [동일 물체 Brier 대조](repeat_score46.json)
