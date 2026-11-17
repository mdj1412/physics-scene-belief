# Target label을 열기 전 projection 손익 예측

예측 대상 D는 보정 없는 privileged projection의 기대 Brier에서 동일 자료를 받은 original의 기대 Brier를 뺀 값이다. 주 표본은 base·hollow·bulge를 동일 가중한 all, 보조 표본은 hollow·bulge만 남긴 deformed다. 절댓값에 threshold를 두지 않고 양수는 손해, 음수는 이득, 정확한 0은 tie로 처리한다. 이 예측은 simulator 기하를 아는 설계자용 진단이다.

### 예측식 및 비교 기준

1. 완료된 v43의 12개 catalog에서 과거 물체의 원래 두 예측과 latent를 읽는다. 새 v44 물체, 해당 GPU 예측, PyBullet label은 읽지 않는다. 물체의 7점 센서가 주는 두 확률은 과거 값 그대로 유지한다.
2. Tip 반경 R, 곧은 sole의 길이 L, 명목 높이+offset을 정한다. 수평 u축의 간격 0.025 mm에서 정지한 footprint의 지지 높이 z(u)+sqrt(R²−max(|u|−L/2,0)²)가 최대인 지점의 실제 미분 기울기를 구한다. 최대값이 수치상 같은 지점은 평균한다. Material field와 유효 static friction도 그 지점에 적용한다. 실제 충돌의 미끄러지는 동안 나타나는 접촉점 이동·압력 분포·속도 변화를 이 근사가 표현하지는 않는다.
3. V43의 단일 impulse 법칙에 앞 단계의 기울기와 마찰을 넣어 대리 사건 Y*를 얻고, 같은 과거 물체에서 p_priv²−p_orig²−2(p_priv−p_orig)Y*를 직접 평균한다. R=L=0인 검산에서 이 대리식은 과거의 세 offset label을 모두 재현했다. PyBullet solver를 호출하지 않는다.
4. 강한 비교는 과거 v43의 같은 형상·높이·offset에서 관측한 Brier 차이를 그대로 반복하는 carry다. 단순 기준은 모든 조건에서 projection이 손해라고 답한다. 입력 family를 알고 비교하는 평가이므로 실시간 blind selector의 성능을 나타내지 않는다.

### 동결된 방향과 크기

아래는 all 표본의 예상 D×1,000이다. 각 칸의 세 값은 순서대로 실제 접촉점을 −0.5, 0, +0.5 mm 이동한 경우다. R·L은 mm다.

| Shape / 명목 높이 | R=0.5, L=0 | R=0.5, L=2 | R=2, L=0 | R=2, L=2 |
|---|---:|---:|---:|---:|
| fold / 19 mm | +8.56 / -0.20 / +17.48 | +16.52 / +15.33 / +13.51 | +9.47 / +2.87 / +13.98 | +16.48 / +16.42 / +11.75 |
| fold / 31 mm | -7.33 / -1.43 / +3.61 | -8.71 / -8.26 / -3.65 | -7.55 / -2.44 / +3.23 | -8.70 / -8.37 / -3.50 |
| ripple / 19 mm | +6.66 / +2.80 / +9.22 | +6.28 / +5.85 / +5.67 | +6.64 / +4.49 / +7.08 | +5.86 / +5.52 / +6.01 |
| ripple / 31 mm | +3.69 / +1.12 / +3.14 | +2.98 / +3.96 / +3.77 | +4.13 / +2.81 / +2.89 | +2.75 / +3.35 / +4.20 |

### 채점 단위와 불확실성

각 scope에서 2 shape × 2 높이 × 2 R × 2 L × 3 offset의 48개 칸을 평가한다. 먼저 각 칸의 12개 독립 catalog 평균 D의 부호와 비교한다. 두 scope의 48개를 합쳐 n=96 독립 시행으로 계산하지 않는다. 각 방법의 부호 일치율, matched carry 대비 차이, 평균 절대 예측 오차(MAE)를 기록한다. 12개 catalog를 대응한 상태에서 재표집(20,000회, seed 440517)하여 sign agreement와 MAE 차이의 95% percentile interval을 구한다. 같은 설계의 칸들은 재표집할 독립 단위가 아니다.
각 scope별 48개 paired t 구간을 Bonferroni 보정해 0과 구별되는 칸 수와 그 안의 적중 수도 보조 보고한다. 같은 R·L의 두 offset 평균 및 nominal look에서 형상별 pooled 부호도 기록한다. 설계자가 재사용할 수 있는 평가 신호와 실제 배포 정보는 구분한다.

[동결된 예측 데이터](forecast44.json) · [서버 동결 checksum](forecast44.sha256) · [UTC 시각](forecast44_utc.txt) · [예측 코드](forecast44.py)
