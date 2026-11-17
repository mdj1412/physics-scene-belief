# v47 · 관측만으로 projection을 선택하는 규칙과 식별력

## 배포 입력과 source fit

정책 입력은 접촉 전 영상에서 계산한 위치와 normal variance, 명령한 높이, 이미 구매한 seven-channel profile이다. 한 물체의 한 번의 paid sensor draw에서만 feature를 뽑는다. Deployment probe를 산 경우 query 입력의 일곱 값은 모두 NaN이고 정책은 original을 선택한다. 같은 물체의 다른 variant, alignment, mechanical coupling 및 미래 outcome은 보지 않는다.

Source axis와 measured profile을 동일한 quadratic residual space에 투영한다. 일곱 측정 좌표에 대해 상수항·기울기·곡률을 least squares로 제거하므로 기존 bank의 quadratic baseline이 사라진다. 남은 profile과 residualized source axis의 squared cosine, 그리고 residual profile RMS(mm)를 계산한다. 이 값은 sensor noise, 영상에 보이는 원래 형상, 변형을 모두 포함하는 single-observation proxy다. 실제 두 변형의 차이는 사후 검산에만 쓴다.

각 source 표본에서 projected와 original의 Brier 차이를 계산했다. v45와 v46의 두 pressure radius를 같은 비중으로 합쳤고 v46의 실제 camera gate 결정을 보존했다. 위치 6 bin, 기존 gate 결정 2가지, squared cosine 4 bin, RMS 3 bin을 조합했다. Cosine 경계는 0.125·0.375·0.625, RMS 경계는 0.30·0.60 mm다. 각 구간의 상대 loss를 같은 위치·기존 결정의 평균 쪽으로 25% shrink했다. 128개 미만의 source observation만 있으면 parent 평균을 썼다. 빈 parent는 위치 평균을 따른다. Source 학습에는 GPU inference를 추가하지 않고 node55의 할당된 CPU 한 개를 썼다. 새 target을 평가하기 전에 [policy](policy47.json)와 [hash](router_freeze47.sha256)를 고정했다.

**학습 후 projection 허용 범위가 매우 좁았다.** 144개 bin 중 89개에 source observation이 있었고 projection을 선택한 bin은 하나였다. 31.0 mm에서 기존 v46이 original을 선택하고, cosine ≤0.125, RMS가 (0.30, 0.60] mm인 경우다. 해당 bin의 source 예측 Brier 차이는 −0.00001576이었다. Source 전체 query observation의 0.2622%가 이 조건에 속했다. 따라서 높은 alignment를 관측하면 projection이 좋다는 단조 규칙으로 해석할 수 없다. 독립 source family를 차례로 제외하고 학습한 뒤 그 family에서 평가한 query-only 평균 차이는 fold 0, ripple +0.0001265, terrace +0.005067, dimple +0.0000020이었다. 어느 family에서도 source 외 개선을 보여주지 못했다.

## 같은 source로 camera만 다시 학습하면 어떻게 되는가

[Camera-only ablation](camera_ablation47.json)은 primary source와 동일한 paired label과 paid query를 사용해 residual feature를 적분하고, 위치와 v46 gate 결정만으로 Brier 차이를 학습한다. 표본 수를 primary의 해당 bin에 대해 합산한 값과 완전히 일치시켰다. 12개 camera cell의 상대 risk가 모두 양수였다(0.001296~0.019092). 모든 입력에서 original을 선택하므로 이 정책의 target Brier는 분석적으로 always-original과 동일하다. 이 대조는 추가 source family를 이용한 보수적 재학습과 paid profile을 이용한 개별 선택을 구분한다. 별도로 GPU latency를 재지 않았으므로 Brier 동일성이 latency 동일성을 뜻하지 않는다.

## 새 target에서 proxy는 alignment를 실제로 읽었는가

미래 label에 접근하지 않는 [검산](proxy_audit47.py)은 12개 catalog의 이미 구매한 profile 706,560개를 사용했다. Probe arm의 profile 79,872개는 모두 제외했다. 아래 stratified AUC는 각 catalog·family·높이·variant 안에서 high를 low보다 높은 점수로 판별하는 unpaired AUC를 구한 후 같은 비중으로 평균했다. Paired rank는 **같은** object·환경·noise를 고정하고 alignment만 바꾼 두 관측의 순위다. 구간은 catalog를 단위로 한 descriptive 95% t interval이다.

| 측정량 | 결과 |
|---|---:|
| Squared-cosine proxy의 stratified AUC | 0.756290 [0.755501, 0.757079] |
| Matched pair에서 high proxy가 더 큰 비율 (동률은 0.5) | 0.815917 [0.814895, 0.816938] |
| Squared-cosine proxy 평균, low / high | 0.272151 / 0.493293 |
| Residual RMS 평균, low / high | 0.485872 / 0.358999 mm |
| RMS의 stratified AUC, 큰 RMS가 high라는 방향 | 0.204552 [0.203382, 0.205721] |
| 공통 noise인 hollow/bulge pair로 raw axis의 실제 squared cosine을 검산한 최대 오차 | 3.68×10⁻⁷ |
| v46 및 신규 gate replay 불일치 | 모두 0건 (신규 gate 선택 2,359,296건) |

고정한 family·height·variant에 따라 proxy AUC는 0.505(channel·19 mm·bulge)부터 0.945(keel·19 mm·bulge)까지였다. Raw deformation 전체의 norm을 1.20 mm로 같게 해도 quadratic component를 뺀 뒤에는 남는 진폭이 달라졌다. RMS로 low와 high가 일부 구별되는 것은 이 관측 경로에 맞는 결과다. 반면 null과 strong loaded coupling은 동일한 배포 입력을 공유하므로, balanced prior에서 해당 coupling 자체를 판별할 정보는 없다. 총 Brier에 대한 이득은 별도로 target outcome을 사용해 평가해야 한다. [전체 수치](proxy_audit47.json)
