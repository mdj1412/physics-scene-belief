# v72 · 처음 보는 racket rubber에서 4회와 8회의 calibration은 무엇을 구별하는가?

**Butterfly Feint OX에서 제품 전체의 release 방향을 뒤집는다는 가설은 재현되지 않았다.** 이 long-pips 제품에서는 입사 spin을 바꾸면 같은 rubber에서도 이탈 상대속도의 부호가 달라졌다. 같은 5개 impact를 보류한 8회 비교에서 가장 작은 접선속도·spin 오차는 **single affine empirical Bayes(EB)의 0.245 m/s·19.52 rad/s**였다. Continuous-loss gate는 부호를 판독할 수 있는 4개를 모두 맞혔지만 spin MAE는 50.84 rad/s였다. [Feint OX 실험 기록](method/feint/result.json) · [Araki, Sato & Yamazaki 원문](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf)

**Butterfly Super Anti에서는 영상으로 복원한 속도와 spin이 서로 다른 방법을 지지했다.** 재생 시간축을 보존했다고 가정한 무차원 비교에서 elastic impulse가 속도, continuous-loss gate가 spin에서 가장 낮은 오차를 보였다. 그러나 공통 test 두 충돌 중 이탈 부호를 안정적으로 읽을 수 있는 것은 하나뿐이었다. 두 제품 모두 같은 기준으로 실측 가능한 항목과 추가 가정이 필요한 항목을 구분해 아래에 제시했다. [Super Anti GPU 결과](method/anti_audited/interval/result.json) · [원 영상](https://www.youtube.com/watch?v=43WWRf5yW6M)

## 1. 새 제품의 어떤 측정값을 확보했는가?

| 제품·자료 | 확인한 사실 | 평가에 사용할 수 있는 범위 |
|---|---|---|
| Butterfly Feint OX, long-pips | 1996년 독립 연구팀이 38 mm ball을 고정한 woodblock의 rubber에 충돌시켰다. Fig. 12(a,b)에 실측 이탈 접선속도와 spin의 개별 점이 있고 Fig. 11에는 normal restitution이 있다. | 세 panel에서 유일하게 연결되는 13점으로 외부 평가. 이탈 접선속도·spin은 직접 판독했다. 입사 선속도는 실험 설정값이고 이탈 normal velocity는 보조 분석에서 환산했다. [원문 pp. 24–30, Figs. 11–13](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| TSP Curl P-3 Soft, Butterfly Feint LONG II, long-pips | JTTA 실험은 이 두 제품과 v71의 Grass D.TecS를 포함한다. 입·출사 속도·각도·spin, topspin/backspin과 10구 반복을 명시한다. | 공개 자료는 grouped friction 결과를 제시한다. 같은 ball의 입·출사 상태를 제품별로 연결해 calibration과 test로 나눌 수 없었다. [JTTA 원자료](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf) |
| Butterfly Super Anti, anti-spin | Preston의 고속영상은 표시한 공과 라켓을 보여 준다. Araki의 제품 지도에 나온 Super Anti 위치는 저자가 경험으로 추정한 값이다. | 후보 24개에서 10개 impact의 translation·spin을 재구성했다. 촬영 clock이 확인되지 않아 오차를 입사 normal speed로 나눈 값으로 비교했다. [제작자 영상](https://www.youtube.com/watch?v=43WWRf5yW6M) · [Araki Fig. 3 설명](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/8_1_5_3.pdf) |
| Dr. Neubauer ABS 3 Pro, Der Materialspezialist Scandal, anti-spin | SpinSight 사용자가 제품과 sponge를 명시하고 같은 blade에서 비교했다. | 공개 기록에는 회전 범위와 요약 통계가 있다. 개인별 타구 동작과 결합된 수치에서 shot별 signed spin과 planar velocity를 복원할 수 없었다. v71의 A-B-S II soft는 별도 기존 제품이다. [측정자 원 게시물](https://www.reddit.com/r/tabletennis/comments/1sxutxf/my_experience_with_rubber_comparison_using/) |

논문 참고문헌, ITTF·JTTA archive, 출판사 supplement, 영상 제작자 페이지와 공개 repository를 조사했다. DataCite의 넓은 제목 검색에서 확인한 138개 dataset record에는 중복 버전, 경기 영상, 선수 생체신호, racket kinematics도 포함된다. 좁은 키워드 검색만으로 원자료 유무를 판정하기 어려워 원 논문의 그림까지 추적했다. [검색 후보·배제 근거](data/SEARCH_KO.md) · [검색식·반환 제목·URL](audit/repositories.json)

## 2. Feint OX의 13개 impact와 측정 한계

같은 4월 20일 run에서 Fig. 12(a)의 식별 가능한 17개 중심과 Fig. 12(b)의 16개 중심을 각각 판독했다. 공통 가로축인 incident spin을 맞춰, 두 방향에서 유일하고 독립 판독한 Fig. 11의 사각 marker까지 최대 2 rev/s 이내인 조합만 채택했다. 채택한 점들의 최대 가로축 차이는 1.164 rev/s였다. 약 21–22, 160–161 rev/s의 겹치는 점을 제외했다. 따라서 13점의 연결은 같은 run과 고유한 입사 spin에 근거하며 원본 trial ID는 공개되지 않았다. 복원한 두 panel의 원점 통과 회귀기울기는 원문 인쇄값과 각각 0.3% 이내로 일치했다. [판독 감사](data/ARAKI_AUDIT_KO.md) · [복원 CSV](data/feint_ox_digitized.csv)

원문의 clockwise spin을 $ω=2πn$으로 변환하고, 반지름 $R=0.019$ m에서 이탈 상대속도를 $s'=u'-Rω'$로 계산했다. 공이 접선으로 이동하는 속도와 회전 때문에 접촉점이 이동하는 속도의 차이다. 입사 접선속도 0, 입사 normal speed 30/3.6 m/s는 원 저자가 설정한 **명목 입사값**이다. 주 평가에서는 Fig. 12에서 직접 읽은 이탈 접선속도와 spin만 사용했다. EB와 elastic impulse의 observation covariance도 이 두 성분으로 marginalize했다. [측정 출처와 단위](data/araki_source.json) · [외부 평가 구현](method/feint.py)

![Feint OX의 incident spin에 따른 접선속도, 회전에 의한 표면 속도, signed release](analysis/feint_release.png)

두 점은 그림 해상도에서 $s'=0$이다. 나머지 중 입사 spin 30.3–56.2 rev/s의 네 점에서 음수, 78.1–180.4 rev/s의 일곱 점에서 양수였다. 56.2 rev/s의 한 점은 −0.14 m/s로 0에 가깝다. $|s'|≤0.20$ m/s를 보류해도 음수 3점과 양수 7점이 남는다. 두 구간 사이의 정확한 transition threshold는 관측하지 않았다. Araki 등도 입사 spin에 따른 tangential restitution 변화를 이미 보고했다. [원문 Fig. 13](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [점별 수치](analysis/external_support.json)

## 3. 동일 예산의 네 방법과 지표

- **Majority vote**는 v71의 두 frozen expert를 그대로 쓴다. Calibration에서 source gate가 실측 release 부호와 어긋난 횟수가 판독 가능 표본의 절반을 넘으면 routing을 뒤집고, 동률이면 유지한다.
- **Continuous-loss gate**는 같은 두 frozen expert로 두 routing을 모두 계산한다. Calibration에서 $|Δu'|+R|Δω'|$ 평균이 작은 routing을 선택한다. 두 항의 단위는 m/s다.
- **Single affine EB**는 여섯 source 제품 사이에서 affine coefficient가 함께 달라지는 양상을 prior로 학습한다. 새 제품의 같은 calibration 결과로 계수를 갱신한다. Prior 강도는 v71의 source 검증값으로 고정했다.
- **Elastic impulse**는 접선 impulse가 translation과 rotation을 함께 바꾸도록 제약한다. v71에서 고정한 네 계수의 parameterization과 source prior 추정법을 사용한다. 가설별 target 관측 예산은 같다. [고정 protocol](method/FEINT_PROTOCOL_KO.md)

Source는 v71의 여섯 제품 공식 train 6,490회다. 보류할 점은 **입사 spin 순위만으로** 정했다. 13점에 고르게 걸치도록 8개를 calibration 후보로 정하고, 나머지 5개를 모든 정책·예산의 공통 test로 사용했다. K=4는 8개 중 4개를 고르는 70개 조합의 평균이고, K=8은 한 번의 분할이다. 같은 장치·session 안에서의 성능이며 새 시편이나 날짜에 대한 반복 실험은 아니다. [실제 분할과 결과](method/feint/primary_rows.csv)

**Release 방향 감지**에는 세 숫자가 관련된다. 제품 전체의 source routing을 뒤집을지 고르는 것은 두 gate의 행동이다. 선택된 expert에 지정한 명목상 부호의 적중률과, 네 방법이 실제로 출력한 접선속도·spin으로 계산한 부호의 적중률도 분리했다. Expert에는 자기 분기의 부호를 항상 내도록 하는 제약이 없다. 0으로 판독된 행도 calibration 비용과 velocity·spin 오차에는 포함했다. 공통 test 5개에서 부호를 판독할 수 있는 것은 4개다.

## 4. Feint OX 외부 평가

주 평가의 velocity error는 **직접 판독한 접선속도의 MAE (m/s)**, spin error는 **angular velocity의 MAE (rad/s)**다. 백분율은 최종 연속 출력의 release sign accuracy다.

| 방법 | K=4 접선속도 / spin | K=4 부호 | K=8 접선속도 / spin | K=8 부호 |
|---|---:|---:|---:|---:|
| Majority vote | 1.232 / 47.89 | 81.1% | 1.543 / 46.95 | 3/4 |
| Continuous-loss gate | **0.264** / 50.84 | **100%** | 0.264 / 50.84 | **4/4** |
| Single affine EB | 0.326 / **22.37** | 77.9% | **0.245 / 19.52** | 3/4 |
| Elastic impulse | 0.503 / 37.91 | 83.2% | 0.387 / 26.00 | 3/4 |

**제품 단위 감지 결과.** Source gate는 13점에서 모두 양수를 예측했다. 공통 test에서 부호를 판독한 4개는 양수 3, 음수 1이므로 제품 과반 기준의 정답은 원래 방향 유지다. Majority vote는 K=4에서 53/70회(75.7%), K=8에서 1/1회 이 방향을 선택했다. Continuous-loss gate는 K=4의 70/70회와 K=8의 1/1회에 routing을 뒤집었다. K=8에서 이 gate가 고른 nominal branch 부호는 1/4만 맞았으나 실제 expert 출력의 부호는 4/4였다. 큰 연속 오차를 피하려고 expert를 바꾸는 행동을 제품의 물리적 방향 반전으로 해석하면 이 사례를 잘못 설명하게 된다. [Impact별 실측값과 예측값](method/feint_k8_held_predictions.csv)

**분할에 따른 순위.** Calibration과 test가 겹치지 않도록 13개에서 가능한 모든 조합도 평가했다. K=4에는 715개, K=8에는 1,287개다. 같은 13개에서 남은 점들을 반복 평가한 sensitivity다.

| 방법 | K=4 접선속도 / spin | K=8 접선속도 / spin | K=8 출력 부호 적중률 |
|---|---:|---:|---:|
| Majority vote | 1.235 / 57.55 | 1.351 / 59.30 | 63.6% |
| Continuous-loss gate | **0.340** / 43.58 | **0.339** / 43.53 | **100%** |
| Single affine EB | 0.419 / **32.49** | 0.352 / **27.51** | 70.4% |
| Elastic impulse | 0.622 / 41.62 | 0.544 / 37.56 | 71.5% |

8회 공통 test에서 EB가 continuous-loss gate보다 접선속도 MAE가 0.019 m/s 낮았지만, 모든 분할에서는 continuous-loss gate가 0.013 m/s 낮았다. 명목 입사 접선속도를 −0.1 m/s로 바꿔도 공통 test에서 순서가 바뀌었다(0.218 대 0.241 m/s, continuous-loss gate가 앞섰다). EB의 spin 우위는 분할을 바꾸거나 입사 normal 속도를 27·33 km/h로 둔 sensitivity에서도 유지됐다. 이 범위는 연구자가 정한 stress test다. 원 실험의 측정오차 분포는 알 수 없다. [전체 sensitivity](method/feint/result.json)

**Normal 성분을 더한 보조 평가.** Fig. 11의 normal restitution에 명목 입사속도를 곱해 이탈 normal speed를 근사했다. 같은 13개 행과 분할에서 네 방법 모두 이 근사 성분도 calibration에 사용했다. 각 칸은 근사 2D vector error (m/s) / spin MAE (rad/s)다.

| 방법 | K=4 | K=8 |
|---|---:|---:|
| Majority vote | 1.478 / 47.89 | 1.817 / 46.95 |
| Continuous-loss gate | 0.422 / 50.84 | 0.422 / 50.84 |
| Single affine EB | **0.414 / 22.25** | **0.333 / 19.52** |
| Elastic impulse | 0.549 / 36.79 | 0.441 / 26.12 |

위 표의 normal 오차에는 모든 shot에서 입사속도가 nominal setting과 같다는 가정이 들어간다. 가장 직접적인 외부 증거는 앞의 접선속도·spin 결과다. [보조 평가](method/feint_derived/result.json)

## 5. v71과 순서가 달라진 이유를 어디까지 알 수 있는가?

같은 GPU에서 기존 6제품·1,623 held-out impact에 네 방법을 다시 적용했다. 제품을 하나씩 source에서 통째로 제외하고 동일한 256개 무복원 prefix를 썼다. 여기서 velocity는 **동시에 실측된 XZ vector error**다. 제품을 동등하게 가중했고 신규 Feint OX는 이 표에 포함하지 않았다.

| 방법 | K=4 velocity / spin | K=4 출력 부호 | K=8 velocity / spin | K=8 출력 부호 |
|---|---:|---:|---:|---:|
| Majority vote | 0.886 / 51.52 | 82.31% | 0.878 / 50.76 | 82.30% |
| Continuous-loss gate | 0.812 / 44.56 | 73.42% | 0.812 / 44.54 | 74.22% |
| Single affine EB | 0.373 / 19.53 | 86.97% | 0.348 / 18.15 | 87.35% |
| Elastic impulse | **0.352 / 17.93** | **88.07%** | **0.300 / 15.06** | **90.03%** |

K=8에서 기존 제품의 elastic impulse는 EB보다 −0.048 m/s·−3.09 rad/s였고 제품을 하나씩 제외해도 두 차이 모두 음수였다. 반대로 외부 Feint OX에서는 공통 분할과 전수 분할 모두 EB가 elastic impulse보다 두 오차를 낮췄다. 기존 제품에서 continuous-loss gate가 majority 대비 얻은 −0.066 m/s·−6.22 rad/s는 A-B-S II soft의 영향이 컸다. 그 제품을 제외하면 평균 차이는 +0.002 m/s·+0.11 rad/s였다. 같은 A-B-S II soft의 held mismatch는 76/158인데 두 routing의 연속 손실 차이는 컸다. [Paired·leave-one-product-out 분석](analysis/paired.json) · [기존 제품별 방향 감지](method/METHOD_KO.md)

**입력 조건의 차이.** Feint OX 13점 중 2점은 공의 rim speed가 기존 6제품 source의 최대보다 컸다. 세 입력축을 source 표준편차로 나눈 뒤 최근접 거리를 구하면, 13점 전부가 source끼리의 최근접 거리 95 percentile보다 컸다. 이는 표본 배치를 기술하는 진단이다. Rubber, ball diameter, woodblock, 입사 조건, 측정 장치가 함께 바뀌어 모델 순위 변화의 원인을 이 중 하나로 특정할 수 없다. [Source support 계산](analysis/external_support.json)

**Elastic impulse가 동시에 맞출 수 있는 오차의 하한.** V71의 thin-shell 제약은 $Δu+κΔ(Rω)=0$, $κ=2/3$이다. 명목 입사 $u=0$에서 이 제약을 지키는 모든 예측의 접선속도 오차와 rim speed 오차 합은 $|u'+κR(ω'-ω_{in})|$ 이상이다. 이는 두 항의 부호가 어떻든 성립하는 triangle inequality다. 공통 test 5개에서 이 하한의 평균은 **0.691 m/s**였다. K=8의 실제 두 오차 합은 elastic impulse **0.881**, EB **0.616 m/s**였다. 즉 이 표와 명목 입력에서는 v71의 제약 안에서 계수만 바꿔 EB의 오차 합까지 도달할 수 없다. 다만 Araki가 제시한 해당 ball의 inertia ratio 0.653을 대입하면 하한이 **0.600 m/s**로 내려간다. 이 민감도와 입사속도 미측정 때문에 실제 공의 impulse 법칙이 깨졌다고 결론낼 수는 없다. 이 하한은 target label을 본 뒤의 진단이며 정책 입력으로 쓰지 않았다. [하한·점별 잔차](analysis/impulse_bound.json) · [원문](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf)

## 6. Super Anti에서 어느 관측을 믿을 수 있는가?

Preston의 원 영상은 수직으로 고정한 blade에 손으로 공을 보내고, 같은 조건으로 분류한 여러 rubber의 장면을 나란히 편집했다. 왼쪽 위 Super Anti 패널에서 입사 topspin/backspin, average/faster feed, 선행 table bounce 유무, blade의 상·중·하단을 교차한 **24개 고유 장면**을 확인했다. 각 장면의 첫 연속 재생에서만 ball center와 stripe를 추적했다. 2차원 회전을 가정하기 어려운 sidespin과 replay는 표본에서 제외했다. [영상 제작자의 실험 설명](https://www.youtube.com/watch?v=43WWRf5yW6M&t=70s) · [24개 장면 목록](audit/preston/validation/title_inventory.csv)

**판독 가능성 자체가 속도에 의존했다.** 궤적과 stripe의 품질 기준을 함께 통과한 것은 10/24개였다. Average feed는 9/12개, faster feed는 1/12개였고, faster topspin은 0/6개였다. 따라서 아래 평균은 영상에서 추적할 수 있었던 충돌에 대한 값이다. 10개 충돌의 입·출사 구간에서 435개 인접 frame을 검사했으며 복제되거나 거의 정지한 쌍은 발견되지 않았다. 이 검사는 누락된 acquisition clock이나 시간 재매핑을 복원하지는 못한다. [판독 기준](audit/preston/quality_protocol.md) · [조건별 coverage](analysis/anti_coverage.json) · [Frame 점검](audit/preston/validation/cadence.csv)

**측정 단위.** Playback은 29.97 fps이며 이 영상 자체의 capture rate와 metric scale은 확인되지 않았다. 각 impact의 입사 normal speed 절댓값으로 translation과 rim speed를 모두 나눈 후, 같은 변환을 거친 기존 source 6제품에서 네 모델을 다시 적합했다. Velocity error는 정규화한 이탈 평면 속도 벡터 간 거리이고 spin error는 정규화한 rim speed의 절대차다. 동일 clip의 충돌 전후에 공간·시간 scale이 유지될 때 비율에서 이 scale이 상쇄된다. 아래 수치는 영상 point estimate와의 차이이며, Feint OX의 m/s·rad/s와 합산할 수 없다. [원본 식별 정보](audit/preston/source.json) · [추정식·split](method/ANTI_PROTOCOL_KO.md)

입사 상태만으로 정렬한 10개에서 8개를 calibration 후보로 정하고 Ref. 24와 1을 공통 test로 고정했다. K=4는 70개 가능한 calibration 조합의 평균이고 K=8은 하나의 조합이다. 아래 마지막 열의 분모는 판독된 **고유 held impact 한 개**다.

| 방법 | K=4 velocity / spin, 무차원 | K=8 velocity / spin, 무차원 | K=8 release 방향 |
|---|---:|---:|---:|
| Majority vote | 0.1722 / 0.3550 | 0.1722 / 0.3550 | 1/1 |
| Continuous-loss gate | 0.2208 / **0.0362** | 0.2208 / **0.0362** | 0/1 |
| Single affine EB | 0.2056 / 0.2377 | 0.2190 / 0.2244 | 1/1 |
| Elastic impulse | **0.1063** / 0.1803 | **0.1013** / 0.1781 | 1/1 |

K=4에서 같은 Ref. 1을 맞힌 calibration 조합 수는 위 순서대로 70/70, 0/70, 59/70, 66/70이었다. Majority vote에 사용할 수 있는 calibration label은 8개 중 Ref. 17 하나였다. 70개 조합 중 35개에는 유효 vote가 0개여서 기본 routing을 유지했다. 나머지 35개에도 vote가 하나뿐이었다. 따라서 이를 새 anti-spin 제품에서 reversal을 감지한 비율로 해석할 수 없다. [GPU 결과와 모든 split](method/anti_audited/interval/result.json)

**왜 1개만 채점했는가?** 추정 반지름, blade tangent, 중력에 의한 vertical acceleration, 각속도·위치 판독을 바꾼 sensitivity에서 이탈 상대속도의 부호가 남은 것은 Ref. 1과 17뿐이었다. Ref. 1의 범위는 [−2.260, −0.025], Ref. 17은 [−8.25, −4.77] px/playback frame이었다. Ref. 1에 0.1 px/frame의 최소 여유를 요구하면 held-out 분모는 0이다. Ref. 17의 이탈 상대속도는 독립 판독에서도 약 −6.3이었지만, 입사 부호는 자동 판독과 독립 판독에서 달랐다. 이탈 부호 하나가 명확해도 입사 대비 물리적 reversal까지 판정할 수 없는 실제 사례다. [점별 sensitivity](audit/preston/sign_sensitivity.csv) · [엄격한 margin 재평가](method/anti_audited/margin_010/result.json)

**Test composition에 대한 민감도.** 같은 10개 관측을 256개 공통 재분할로 반복하면 K=8에서 가장 낮은 velocity error는 elastic impulse 0.309, 가장 낮은 spin error는 continuous-loss gate 0.184였다. 반면 궤적 전·후반의 normal speed 비율에 0.8–1.25 조건을 더한 9개로 공통 split을 다시 만들면 held-out이 하나로 줄었고, K=8에서 elastic impulse가 두 오차 모두 가장 작았다(0.035 / 0.067). 재분할 간 일치는 새 물체·새 session에서의 반복 검증과 구별해야 한다. [10개 재분할](method/anti_audited/interval/result.json) · [9개 sensitivity](method/anti_audited/steady_normal/result.json)

미확인 편집 배속의 영향을 별도로 살피기 위해 이탈 translation과 rotation을 함께 0.5배와 2배로 바꿨다. 전자에서는 K=8 velocity가 EB 0.092, elastic impulse 0.118로 순위가 뒤집혔고, 후자에서는 elastic impulse 0.159가 가장 낮았다. Continuous-loss gate는 두 경우 모두 spin error가 가장 작았다. 배속을 추정한 실험이 아니라, 일관된 시간 변환에 의존하는 결론이 무엇인지 드러내는 검사다. [민감도 요약](method/SUPER_ANTI_KO.md) · [독립 수동 판독](audit/preston/validation/AUDIT_KO.md)

## 7. 논문에 남길 주장과 다음 확인 실험

**지금 방어할 수 있는 기여는 few-shot calibration에서 ‘방향을 맞혔다’는 말의 의미를 측정으로 구분한 것이다.** Feint OX에서 8회 continuous-loss gate는 명목상 잘못된 branch를 3/4회 선택하면서 최종 release 부호를 4/4회 맞혔다. 반대로 EB는 부호 하나를 놓쳤지만 spin MAE가 50.84에서 19.52 rad/s로 줄었다. Super Anti에서는 continuous-loss gate가 작은 spin 오차를 보였어도 평가 가능한 이탈 부호 하나를 놓쳤다. 따라서 gate를 무엇으로 선택했는지, 그 expert가 어떤 부호를 예측했는지, 연속 운동을 얼마나 정확히 재현했는지를 세 가지 별도 결과로 보고할 실증적 근거가 있다. [Feint 사례별 예측](method/feint_k8_held_predictions.csv) · [Super Anti 사례별 예측](method/anti_audited/interval/k8_held.csv)

**방법론 논문의 외부 검증으로는 아직 약하다.** 한 long-pips 시편에서 EB의 spin 우위는 비교적 안정적이다. Anti-spin에서 확인된 것은 추적 가능한 단일 영상 내 상대 오차이며, 빠른 공이 선별적으로 누락되고 명확한 held-out sign은 하나다. 기존 6제품의 elastic impulse 우위와 외부 제품의 순위 차이는 contact law, 공, 입사 범위, 실험실과 계측 방법이 함께 변한 결과다. Pimple deformation의 기전이나 제품군 전체의 reversal detection rate는 이 자료로 식별할 수 없다. 선행연구가 이미 측정한 state-dependent restitution 자체를 신규 기여로 삼기도 어렵다. [Araki 등](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard](https://arxiv.org/html/2604.11349v1)

**다음 확인 실험의 우선순위는 두 번째 독립 anti-spin 시편과 공통 기준 시편을 같은 장치에서 측정하는 것이다.** Grass D.TecS를 두 실험실 간 공통 제품으로 삼고, Feint 계열과 Super Anti 또는 추가 anti-spin을 포함한다. 구매 가능한 새 시편의 정확한 모델·sponge·사용 이력을 기록한다. 입사 spin의 두 방향과 낮고 높은 normal speed를 교차하고, 새로운 날짜와 시편을 통째로 보류한다. Synchronized high-speed camera와 metric scale로 각 impact의 입·출사 translation과 spin을 측정한다. 보류할 입사 조건, 4/8회 allocation, sign margin 및 속도·spin별 오차를 target outcome을 열기 전에 고정한다. 판독 불가율도 속도별로 보고하면 물리적 모호성과 tracking 실패를 구분할 수 있다. 기존 자료를 추가 확보할 수 있다면 Feint LONG II와 Curl P-3 Soft를 측정한 JTTA 실험의 shot-level 기록이 가장 직접적인 후보다. [JTTA 측정 조건](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf)

## 8. 서버 실행과 재현

- Datacenter3의 node56에서 실행 직전 사용량을 확인한 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장과 PyTorch 2.7.1+cu128을 사용했다. Slurm job 2744471 내 기존 제품 비교는 step 305, Feint OX 주 평가는 310, Super Anti의 판독 구간 적용 평가는 317이다. 네 방법은 각 split에서 같은 calibration impact와 같은 test를 사용했다. [서버 실행 기록](method/retrospective/run.json) · [Feint OX](method/feint/result.json) · [Super Anti](method/anti_audited/interval/result.json)
- Feint OX의 두 endpoint에서 각각 284개 공통 분할·방법 조합을 별도의 CPU 구현으로 검산했다. GPU와 CPU의 최대 절대차는 velocity에서 1.1×10⁻¹⁴ m/s, spin에서 3.8×10⁻¹³ rad/s 이하였다. Super Anti의 판독·배속 조건 다섯 개도 각각 284개를 검산했다. 독립 판독자는 원 영상의 세 장면을 다시 추적했고, 전체 광학 복원도 같은 영상에서 재실행해 세 CSV가 byte 단위로 일치했다. 계산 재현성과 물리 시편의 재현성은 구분된다. [Feint 검산](method/feint_audit.json) · [Super Anti 검산](method/anti_audited/interval/audit.json) · [원영상 복원 절차](audit/preston/README_KO.md)
- 재현 ZIP을 풀고 racket_external_v72 디렉터리에서 아래 명령을 실행한다. 이 명령은 포함된 측정값부터 모델 평가를 재현한다. 별도로 figure에서 좌표를 다시 읽거나 영상에서 측정하는 절차, 원본 URL·checksum은 각 audit에 있다. [Araki 그래프 복원](data/ARAKI_AUDIT_KO.md) · [Preston 영상 복원](audit/preston/README_KO.md) · [기존 source data와 license](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

~~~sh
python method/bench.py --output method/retrospective
python method/feint.py data/feint_fig12_paired.csv --output method/feint
python method/feint.py data/feint_ox_digitized.csv --derived-normal --output method/feint_derived
python method/anti_sensitivity.py
python analysis/anti_coverage.py audit/preston/anti_eligible.csv
~~~

## 부록. 선행연구 비교와 연구 질문

### 1. 개별 연구를 같은 질문으로 비교한 결과

아래에서 impact는 한 공이 한 번 racket에 닿은 사건을 뜻한다. 제품명, 공의 입사 상태, 같은 공의 이탈 상태를 연결할 수 있어야 제품별 calibration과 독립 평가를 만들 수 있다. 제품명만 확인되는 자료와 모든 측정값이 같은 사건으로 연결되는 자료의 차이가 컸다.

| 원 연구 | 연구 질문과 방법 | 표본과 발견 | 한계와 v72에서 내릴 결론 |
|---|---|---|---|
| Araki, Sato & Yamazaki (1996) | 회전하는 공에서 normal·tangential restitution은 입사 상태에 따라 어떻게 달라지는가? 정지한 woodblock에 rubber를 붙이고 공의 운동을 고속 촬영했다. | Butterfly Feint OX의 Model I은 수직 입사, nominal 30 km/h, 입사 spin 0–200 rps다. 저자는 low spin에서 상대적으로 미끄럽고 spin을 높이면 tangential restitution이 음수 쪽으로 바뀌는 경향을 보고했다. Fig. 11은 normal restitution, Fig. 12a·b는 tangential velocity와 spin의 개별 marker를 보인다. | 새 long-pips 제품의 실제 충돌 결과를 읽을 수 있다. 다만 일부 입사 spin이 겹치고 공개 trial ID는 없다. Fig. 3에 물음표와 함께 표시한 Super Anti는 저자가 경험에 따른 추정이라고 명시했다. [본문 pp. 24–30, Fig. 3, 11–13](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| Preston Table Tennis Association (2010), 저자 공개 실험 영상 | 수직 고정한 동일한 blade에서 입사 spin, 속도, impact 위치, 표면 종류에 따라 실제 rebound가 어떻게 달라지는가? | Butterfly Super Anti의 오래된 시편을 포함한다. Image Ref. 1–24는 top/back × 두 feed speed × 선행 bounce 유무 × blade third 세 곳이다. 24개 조건의 상단 왼쪽 패널에 실제 공과 marker가 보인다. | Hand feed이며 편집자가 비교에 적합한 장면을 골랐다. 원본 capture rate를 확인하지 못해 playback 기준 운동과 같은 구간 내 비율·부호를 평가한다. 24개 중 10개가 입·출사 tracking 품질 기준을 통과했다. 435개 연속 frame 쌍에서 편집 hold는 발견되지 않았다. 동일 shot replay는 제외했다. [시험 영상 1:10–2:16 및 2:22–8:03](https://www.youtube.com/watch?v=43WWRf5yW6M&t=70s) |
| Gossard (2026) | 입사 속도와 spin으로 restitution 및 tangential impulse를 추정하고 새 관측으로 갱신할 수 있는가? 정지 racket에서 여러 state를 만들고 고속 촬영 후 spin과 궤적을 추정했다. | 10개 configuration의 state-dependent model이 상수 계수 baseline 대비 성능 개선을 보였다. 공개 배포에는 제품별 반복 impact와 분리된 train/test가 있다. | Grass D.TecS와 A-B-S II soft는 이미 v71에서 보았다. 이 기록은 source prior 및 기존 제품 재분석에 사용할 수 있다. [논문 §III·V·VI](https://arxiv.org/html/2604.11349v1) · [공식 배포본](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Ushiyama 등 (2012), Niigata University·JTTA 발표 | 정적 drag와 실제 bounce로 측정한 마찰 특성이 load·spin·제품에 따라 달라지는가? | 19개 제품 중 long-pips 세 개는 TSP Curl P-3 Soft, Butterfly Feint LONG II, Tibhar Grass D.TecS다. Dynamic experiment에는 topspin/backspin, 약 2,000–6,000 rpm, 10구 조건 및 입·출사 speed·angle·spin 측정 변수가 명시되어 있다. 발표는 마찰을 조건·종류별로 요약했다. | 앞의 두 제품은 유망한 후속 target이고 Grass는 실험실 사이 연결점이다. 발표에는 제품별 impact마다 결합된 원 측정행이 없다. Feint LONG II와 Araki의 Feint OX는 원문 명칭을 유지하고 합치지 않는다. [원 발표 pp. 1, 3–5](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf) · [연구실 업적 목록](https://www.niit.ac.jp/kotozukuri/resource_db/labo-34/achievement/) |
| Varenberg & Varenberg (2014) | 미끄러지기 직전의 preliminary displacement로 rubber의 grip을 기술할 수 있는가? | short-pips 4개와 long-pips 4개의 sliding onset을 다루며 비교 시 환경 조건의 중요성을 보고했다. | 검토 가능한 abstract의 측정 지표는 재료와 공의 sliding이다. Free ball의 paired velocity·spin 및 4/8-shot 성능을 주지 않는다. [원 논문 DOI](https://doi.org/10.1007/s11249-013-0247-2) · [저자 abstract](https://www.researchgate.net/publication/258165666_Table_Tennis_Preliminary_Displacement_in_Pimples-Out_Rubber) |
| Miyazawa, Hadano & Tanaka (2020) | Pimple height가 rebound에 미치는 영향을 FE에서 예측할 수 있는가? | 실측 검증은 pimples가 없는 rubber plate에서 60 km/h와 두 각도로 수행했다. Pimple height 0.50–1.00 mm인 세 조건은 simulation이다. Spin의 순서가 수평 impulse의 순서를 따랐다. | Impulse를 고려할 물리적 근거를 주며, 세 조건을 실측 신규 제품으로 쓸 수는 없다. [§3–4, Tables 3–4](https://www.mdpi.com/2504-3900/49/1/55) |
| Kreiser 등 (2026), *1000 Rallies* | Event camera의 위치·속도 추정으로 실경기 trajectory prediction이 나아지는가? | 1,000개 이상 rally에서 racket contact를 사람이 표시하고 dotted-ball orientation으로 spin을 산출했다. 200 Hz multi-camera 관측을 사용하는 배열과 다른 실험실의 배열을 별도로 기술했다. | 이 데이터의 ball-state label 생성은 유력하다. 공개 원문에서 각 hit의 rubber 제품·사용 면까지 연결되는 metadata를 확인하지 못했다. [§III, Table I](https://arxiv.org/html/2606.25620v1) |
| Rahmanian 등 (2026), *TT4D* | Monocular broadcast에서 가려진 ball·player·racket trajectory를 복원할 수 있는가? | 140시간 이상 경기의 spin을 lifting network로 추정하고 racket pose와 velocity도 복원했다. | Appendix F의 racket 복원은 inverted rubber용 고정 impact parameter를 사용한다. 이 pose로 같은 impact law를 평가하면 공통 가정이 결과에 들어간다. 방송 경기를 사용하려면 별도의 face·racket-motion·spin 검증이 필요하다. [§3–5, Appendix F](https://arxiv.org/html/2605.01234v1) |

### 2. 주제별 문헌 종합

**제품별 응답.** Araki가 같은 Feint OX에서도 입사 spin에 따라 보고한 변화는 제품 전체에 이산적인 방향 하나만 부여하는 접근의 한계를 제시한다. Gossard도 modern rubbers에서 입사 상태와 impulse의 관계를 측정했다. 두 실험은 ball diameter, 입사 상태의 범위, rubber와 mounting이 달라 직접 합산하기 어렵다. [Araki pp. 24–30](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard §III·V](https://arxiv.org/html/2604.11349v1)

**계측과 상태 추정.** JTTA 연구진은 product identity와 pre/post kinematics를 함께 기록할 수 있는 설계를 공개했다. 최신 multi-camera 및 monocular 연구는 훨씬 많은 자연스러운 rally를 담지만 spin·racket pose의 유래와 각 hit의 제품 identity를 따로 검토해야 한다. 공개 회귀선이나 clip 수만으로 제품별 matched impact 수를 정할 수 없다. [Ushiyama 등](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf) · [1000 Rallies](https://arxiv.org/html/2606.25620v1) · [TT4D](https://arxiv.org/html/2605.01234v1)

### 3. Knowledge map

**연결되는 변수**는 입사 translation·rotation → racket에 상대적인 contact-point velocity → tangential impulse → 이탈 translation·rotation → release sign과 연속 오차다. Araki는 이 연결을 effective restitution으로, Gossard는 state-dependent impulse로 기술한다. Varenberg는 sliding onset을, Miyazawa는 FE 내 contact force의 시간적분을 다룬다. Release sign에서 개별 pimple의 굽힘이나 마찰계수를 역추론하려면 추가 측정이 필요하다. [Araki Eqs. 27–31](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard](https://arxiv.org/html/2604.11349v1) · [Varenberg](https://doi.org/10.1007/s11249-013-0247-2) · [Miyazawa](https://www.mdpi.com/2504-3900/49/1/55)

### 4. 근거가 가리키는 연구 공백

- **하나의 새 제품 안에서 입사조건 효과와 제품 효과가 겹친다.** Feint OX의 공개 개별점은 거의 한 normal speed에서 spin을 바꾼 표본이다. 반대로 Gossard의 다양한 상태는 다른 제품에서 나왔다. 새 제품에서 spin의 부호와 normal speed를 교차해야 두 효과를 구분할 수 있다. [Araki p. 30](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard §III](https://arxiv.org/html/2604.11349v1)
- **물리 단위로 검증된 독립 anti-spin target의 공백이 크다.** Araki의 Super Anti는 Fig. 3의 경험적 배치이고 Gossard의 A-B-S II soft는 기존 target이다. Butterfly의 공개 제품 지표는 자체 시험으로 매긴 상대값이다. Preston의 영상은 이와 별개의 실제 Super Anti 시편을 제공한다. 동일 clip 안에서 translation과 rotation을 비교할 수 있지만 SI 단위의 held-out velocity·spin 오차를 내려면 검증된 capture clock과 metric scale이 필요하다. [Preston 0:20 및 1:21–2:16](https://www.youtube.com/watch?v=43WWRf5yW6M&t=20s) · [Araki Fig. 3](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [공식 데이터](https://github.com/gossardt/tt_rackets) · [Butterfly 지표 설명](https://butterflyonline.com/butterfly-rubber-performance/)
- **모형의 가정이 label에 유입될 수 있다.** Racket frame을 측정하지 않은 경기에서는 pose 복원에 쓴 고정 impact law가 어느 solver가 유리한지를 바꿀 수 있다. Dotted-ball spin과 직접 관측한 racket motion으로 검산할 필요가 있다. [TT4D Appendix F](https://arxiv.org/html/2605.01234v1) · [1000 Rallies §III-B](https://arxiv.org/html/2606.25620v1)

### 5. 불일치의 가능한 이유

Feint OX의 낮은 spin에서 나타난 slippery response와 높은 spin에서 보고된 tangential restitution 변화는 한 제품에 대해 서로 다른 조건에서 관측됐다. Varenberg의 sliding-onset 지표, JTTA의 coefficient 비교, Miyazawa의 FE에서 얻은 impulse는 서로 다른 observable이다. 어느 연구도 다른 연구의 제품별 gate accuracy를 직접 반증하지 않는다. 물리적으로 비교하려면 같은 표면·공·입사 state에서 sliding 지표, rebound sign, 연속 손실을 함께 측정해야 한다. [Araki Fig. 13](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Varenberg](https://doi.org/10.1007/s11249-013-0247-2) · [JTTA](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf) · [Miyazawa](https://www.mdpi.com/2504-3900/49/1/55)

### 6. 방법과 분석 단위가 결과에 미치는 영향

Fig. 11–12의 1차 분석 단위는 incoming spin으로 대응시킨 개별 plotted impact다. Araki의 공 지름은 3.8 cm, dimensionless moment of inertia는 0.653이다. v71은 radius 0.020 m, thin-shell approximation 2/3을 사용했다. 한 번의 impact에서 y축 수치와 x축 수치를 곱하거나 빼기 전에 축 방향과 단위, historical radius를 맞춰야 한다. 특히 Araki의 Fig. 12a는 cm/s, Fig. 12b는 rps다. Normal component는 Fig. 11의 restitution에 nominal 30 km/h를 곱한 복원값이어서 속도 오차에 포함된 근사를 별도로 보고한다. 같은 x에 marker가 겹치면 어느 trial의 spin과 velocity인지 알 수 없어 제외한다. 여러 calibration split에서 같은 shot을 재사용해도 독립 product나 session이 늘어나지 않는다. [Araki pp. 24–25, 30, 44–45](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard 데이터](https://github.com/gossardt/tt_rackets)

### 7. 증거 수준

**직접 확인**할 수 있는 것은 Feint OX가 원문의 실측 대상이라는 점, source에 없는 제품이라는 점, 개별 marker가 나타내는 범위다. **제한된 근거**로는 한 실험 장치와 한 입사축에서 calibration 후 velocity·spin 오차의 순위를 비교할 수 있다. 새 Super Anti 영상에서는 상대 척도의 translation·spin과 부호를 점검할 수 있다. **미확인**인 것은 SI 단위의 Super Anti 오차, 독립적인 다른 anti-spin 시편·날짜와 실제 active stroke에서의 반복성이다. [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M) · 원문의 hardware, setup, plotting resolution 차이까지 흡수한 모형의 성능이 좋아도 원인이 rubber 물성 하나였다고 결론 내릴 수 없다. [Araki §§6–7](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard §§III·VII](https://arxiv.org/html/2604.11349v1)

### 8. 근거에서 도출한 10개 후속 질문

| 질문 | 필요한 변수와 가치 | 연구 접근과 출발 문헌 |
|---|---|---|
| 1. 한 제품에서도 global flip이 바뀌는가? | Incoming spin, normal speed, sign mismatch. 단일 product vote의 한계를 정한다. | 같은 시편에서 spin × speed factorial design으로 source-frozen predictions를 비교한다. [Araki Fig. 13](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| 2. Curl P-3 Soft와 Feint LONG II에서도 ranking이 유지되는가? | 모델·제품·session, paired error. 독립 long-pips 수를 늘린다. | JTTA 원기록을 확보해 4/8-shot protocol을 동결해 적용한다. [Ushiyama 등](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf) |
| 3. Super Anti의 measured response는 경험적 분류와 일치하는가? | Product identity, sign, velocity, spin. anti-spin 검증 공백을 채운다. | Preston의 4/8-shot 상대척도 평가 결과를 별도 시편·calibrated high-speed camera에서 대조한다. [실물 영상](https://www.youtube.com/watch?v=43WWRf5yW6M) · [Araki Fig. 3](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| 4. 장비와 데이터셋 차이가 model ranking을 설명하는가? | 같은 Grass, lab, ball, blade. 제품 shift와 계측 shift를 구별한다. | JTTA와 Gossard의 중복 제품에서 공통 support 내 비교를 한다. [JTTA](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf) · [공식 데이터](https://github.com/gossardt/tt_rackets) |
| 5. 38 mm와 40 mm 공의 차이는 radius 변환으로 얼마나 흡수되는가? | Ball radius·mass·inertia, spin, reversal. System shift를 분리한다. | 동일 rubber에서 공을 무작위 교대하고 dimensionless state로도 평가한다. [Araki pp. 24–25](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| 6. 같은 총예산에서 서로 다른 spin을 골라 관측하면 gain이 커지는가? | Calibration composition과 action gap. 정보가 중복되는 경우를 찾는다. | 결과 공개 전에 spin-stratified policy를 고정하고 uniform draw와 비교한다. [Araki Fig. 12](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| 7. Sliding onset 측정이 few-shot prior를 개선하는가? | Preliminary displacement, 온습도, held-out risk. 재료 특성에 독립 예측력이 있는지 본다. | 동일 시편에서 tribometry와 impact를 독립 취득하고 source-only ablation을 수행한다. [Varenberg](https://doi.org/10.1007/s11249-013-0247-2) |
| 8. Figure digitization이 실험 결론을 바꿀 수 있는가? | Axis error·marker overlap·sign margin. 작은 효과의 측정 한계를 정한다. | 같은 figure를 독립적으로 읽고 가능한 좌표 범위 안에서 ranking 변화량을 계산한다. [Araki Figs. 11–12](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) |
| 9. Broadcast reconstruction이 특수 surface의 spin을 편향시키는가? | Monocular spin·dotted-ball spin·독립 racket frame. Pseudo-label의 편향을 계량한다. | 동일 impact를 두 방식으로 측정하고 product별 residual을 비교한다. [TT4D Appendix F](https://arxiv.org/html/2605.01234v1) · [1000 Rallies §III-B](https://arxiv.org/html/2606.25620v1) |
| 10. Pimple 변형 지표가 남은 연속 오차를 설명하는가? | Lateral deflection, tangential impulse, held-out loss. 단순한 부호와 기전을 구별한다. | 동일 입사 state에서 high-speed 영상과 두 상태량을 측정한 뒤 independent product에서 residual prediction을 검증한다. [Miyazawa §§3–4](https://www.mdpi.com/2504-3900/49/1/55) |

### 9. 논문 주장에 대한 검토

두 외부 시편은 검증하는 양이 다르다. Feint OX의 수치화한 그래프는 nominal speed를 동반한 SI velocity와 spin을, Super Anti의 영상은 상대 척도에서의 운동과 방향을 제공한다. 서로 다른 단위의 오차는 별도 표에 제시해야 한다. [Feint OX 원문](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Super Anti 원 영상](https://www.youtube.com/watch?v=43WWRf5yW6M) 선행연구는 이미 state-dependent tangential restitution과 온라인 adaptation을 제시했다. 새로운 기여 후보는 동일한 4/8개 outcome에서 방향 다수결과 연속 오차 기준을 비교하고, 어떤 조건에서 physical response estimation이 더 나은지 평가하는 것이다. 이를 일반적인 anti-spin·long-pips 적응법으로 주장하려면 공개 영상과 다른 anti-spin 제품·시편, 독립 session, 시간축을 보존한 shot-level 기록이 필요하다. Ball radius, nominal normal speed 및 digitization에 대한 sensitivity가 없으면 작은 성능 차이는 미결로 남겨야 한다. [Araki](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf) · [Gossard](https://arxiv.org/html/2604.11349v1)
