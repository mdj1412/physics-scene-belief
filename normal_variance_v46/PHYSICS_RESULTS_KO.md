# V46 물리 조건과 projection axis가 전이된 정도

## 처음 보는 두 family에서 실제로 지운 신호

V45와 같은 source axis가 새로운 profile의 변형 방향에도 맞는지, target outcome을 사용하지 않고 검사했다. 같은 prototype·sensor noise·위치에서 실제 query를 양쪽 모두 구매한 hollow와 bulge의 7점 profile을 뺐다. 차이 벡터와 해당 nominal height의 frozen unit axis 사이의 squared cosine은 **이 변형 차이에서 projection이 제거하는 energy의 비율**이다. 각 좌표의 noise scale은 동일하다. 무차원 비율이며, Brier나 stick 확률 감소율로 해석하지 않는다. Target의 analytic 차이와 paired paid sensor에서 복원한 차이는 절대 오차 1e−12 이내에서 일치했다. Base→bulge, hollow→base의 별도 값도 보관했다. [계산](axis_transfer46.py) · [모든 variant pair](axis_transfer46.json)

| Family | 19 mm의 제거 비율 | 31 mm의 제거 비율 | 높이당 paired observation |
|---|---:|---:|---:|
| Source fold | 67.93% | 58.28% | 5,520 |
| Source ripple | 67.93% | 58.28% | 5,520 |
| Target terrace | 3.39% | 11.23% | 5,520 |
| Target dimple | 19.19% | **0.036%** | 5,431 |

Dimple 31 mm에서는 원래 하려던 hollow↔bulge 변화가 거의 그대로 남는다. Source에서 유용했던 축이 새 family에서 어떤 정보를 지우는지가 달라졌다는 직접 증거다. 이 측정은 같은 물체의 variant를 만들 수 있는 평가용 진단이며 배포 selector의 입력이 아니다. Axis 불일치가 모든 손실을 일으켰는지 판정하려면 독립 자료에서 축을 재추정하고 동일한 route에 적용하는 별도의 교차 실험이 필요하다.

## 고정한 접촉 위치와 면적의 영향

두 family의 pressure field에서 peak effective area의 평균은 terrace 0.18317→11.68865 mm², dimple 0.18085→11.58431 mm²였다. 양쪽 면적의 최초 위치와 총 scalar normal impulse는 같은 입력을 사용했다. D는 모든 물체에 source-axis projection을 적용했을 때의 Brier에서 original의 Brier를 뺀 값이다. 같은 물체를 각 셀에서 짝지어 12개 catalog를 독립 단위로 구간을 계산했다. 아래 broad−narrow 두 차이의 구간에는 동시 95% 보정을 적용했다. [물리 명세](PHYSICS_PLAN_KO.md) · [결과와 구간](mechanics_audit46.json)

| Shape | Narrow stick 비율 | Broad stick 비율 | 면적에 따라 outcome이 달라진 비율 | Narrow D | Broad D | Broad−narrow D [동시 95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| Terrace | 50.70% | 49.77% | 3.57% | +0.026873 | +0.025127 | −0.001746 [−0.002072, −0.001420] |
| Dimple | 55.76% | 54.79% | 1.67% | +0.009773 | +0.008649 | −0.001124 [−0.001259, −0.000990] |

V45 fold에서는 area를 넓힐 때 D가 +0.003738 커졌다. 이번 두 family에서는 D가 줄었고, 좁은 면적과 넓은 면적 모두 original 쪽의 Brier가 낮았다. 같은 원형 pressure law를 쓰더라도 source에서 얻은 면적 효과의 **방향**을 새 contour로 옮길 때는 이 차이를 고려해야 한다. [V45 면적 대조](../image_pressure_v45/PRESSURE_RESULTS_KO.md)

## 수치와 contact model에 대한 검산

전체 pressure 조건 2,654,208개에서 quadrature를 12×6에서 24×12로 바꾸자 point·narrow·broad의 gross stick label이 각각 0·0·11개 바뀌었다. 최대 terminal speed 변화는 0.000143 m/s였다. Primary 0.020 m/s threshold에서 joint−original은 +0.007589이고, 0.010에서 +0.007583, 0.040에서 +0.007535였다. 같은 비교에서 joint−위치 규칙은 각각 −0.0000164, −0.0000164, −0.0000159였으며 세 threshold 모두 동시 95% 구간에서 0을 제외하지 못했다. [정밀도 및 threshold 검산](mechanics_audit46.json)

독립 R=0.5 mm sphere의 PyBullet 충돌 884,736개 모두에서 양의 normal force가 검출됐고 horizon 0.4 ms에는 하중이 남아 있지 않았다. 요청 위치와 최초 positive-force 위치의 절대 오차는 평균 0.00204, p90 0.00664, 최댓값 0.01012 mm였다. 어떤 물체도 0.25 mm의 위치 bin 경계를 넘지 않았다. Prescribed point control과 rigid label이 달라진 비율은 terrace 1.49%, dimple 0.56%였다. 두 solver는 동일한 재질·geometry·factual sensor를 쓰므로 두 개의 독립 실물 환경으로 셀 수 없다. Pressure와 독립 rigid 원자료 총 1,728개 archive에 대해 서버와 로컬에서 SHA-256을 대조했다. [Raw archive manifest](mechanics_outputs46.sha256) · [Neutral pressure 검산](pressure_neutral46.json) · [Neutral rigid 검산](bullet_neutral46.json)
