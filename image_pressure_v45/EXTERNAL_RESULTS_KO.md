## 5. 독립 PyBullet에서도 영상의 이득이 남았나

Rigid sphere(반지름 0.5 mm)가 같은 요청 위치에 처음 닿도록 초기 상태를 조정하고 PyBullet의 manifold에서 첫 positive-force 좌표를 직접 읽었다. Oracle에는 이 실측 좌표를 주었다. 전체에서 지정 위치와의 MAE는 0.005105 mm, 절대 오차 p90은 0.010893 mm였다. 여섯 bin 사이의 oracle 선택이 바뀐 일은 없었다. Primary의 point model과 rigid solver 간 stick/slip은 2.29% 달랐다. 이 실험은 독립 query dynamics에 대한 대조다. 같은 pre-impact pin 영상을 rigid 접근의 대리 관측으로 사용했으며, sphere를 촬영한 실물 영상은 없다. 분포 면적의 효과는 앞 절의 pressure 실험에서 측정했다. [Rigid setup과 입력 정보](BULLET_PROTOCOL_KO.md) · [전체 검산](external_full_score45.json)

사전 지정한 첫 8개 물체/환경만 보면 visual−original은 **+0.000532**로 이득이 확인되지 않았다. 동일 물체의 prescribed point와 finite pressure에서도 각각 +0.000659와 +0.000791이었다. 표본 구성과 solver 변경을 분리하기 위해 규칙을 그대로 두고 나머지 120개/환경까지 rigid 충돌을 사후 확장했다. 추가 표본은 같은 12개 catalog에서 추출됐고, 첫 8개의 연속 속도·접촉점은 재실행에서 일치했다.

| Visual − original, 같은 물체끼리 비교 | Prescribed finite 평균 | Prescribed point | Rigid PyBullet [3개 정책 비교 동시 95% CI] |
|---|---:|---:|---:|
| 처음 8개/환경 | +0.000791 | +0.000659 | +0.000532 [−0.000823, +0.001887] |
| 추가 120개/환경 | −0.000445 | −0.000649 | **−0.000793** [−0.001139, −0.000448] |
| 전체 128개/환경 | −0.000368 | −0.000567 | **−0.000711** [−0.001005, −0.000416] |

전체 884,736개 rigid impact에서 Brier는 always-original **0.150661**, nominal-only **0.150038**, oracle **0.149899**, visual **0.149950**이었다. Visual−nominal은 −0.000088 [−0.000187, +0.000010]으로, rigid solver에서 세밀한 위치가 명목 높이 위에 더하는 이득은 primary만큼 확실하지 않았다. Visual−oracle은 **+0.000051** [+0.000008, +0.000095]로 작지만 exact location의 이득을 구별할 수 있었다. Fold의 visual−original은 −0.001549 [−0.001927, −0.001170], ripple은 +0.000127 [−0.000120, +0.000375]였다. 방향이 다른 소표본 결과를 solver 교체 하나로 설명할 수 없고, 전체 표본의 일치도는 사후 확장에 해당한다. [초기 소표본](external_score45.json) · [동일 물체 전체와 추가 표본](external_full_score45.json)

## 6. 논문 기여 가능성과 남은 대안 설명

**논문의 중심으로 삼을 결과는 두 가지다.** 첫째, 이미 고정된 shape-blind decoder의 선택에 pre-impact contact coordinate가 어느 정도 가치가 있는지, 그리고 그중 명목 높이만으로 얻을 수 있는 양이 얼마인지 분리했다. Primary에서는 visual이 두 기준을 모두 이겼고, 독립 rigid solver 전체에서도 always-original 대비 이득이 남았다. 둘째, 최초 위치와 scalar impulse를 고정했는데도 pressure 폭이 fold의 projection 손해를 키웠다. 같은 비율로 event가 바뀐 ripple에서 두 방향이 상쇄된다는 paired 분해는 shape별 차이가 왜 생기는지 설명한다. 기존 연구에는 영상 contact localization과 pressure integration 각각의 선례가 충분하다. 이 연구의 후보 기여는 이 둘을 **미래 충돌 예측에 어떤 contour 정보를 남길지**라는 하나의 결정으로 연결하고, 기대 이득과 실패 조건을 함께 측정한 것이다. [V44 선행 실험](../contact_patch_v44/REPORT_KO.md) · [Im2Contact](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Elandt et al.](https://arxiv.org/pdf/1904.11433)

**현재 근거로 설정할 주장 범위.** 같은 위치에서 두 형상이 반대 손익을 보이므로 x 하나로 모든 물체의 최적 decoder를 고를 수 없다. 양의 평균 개선은 이 실험의 균등한 shape 혼합에 의존한다. V44의 손익과 v45의 손익은 projection axis, source rule, 물체와 target dynamics가 함께 달라진 결과여서 버전 간 크기 차이를 camera 오차로 취급할 수 없다. Optical calibration·등속 접근·보이는 fiducial·훈련 때의 paired variant 구성이 필요하다. GPU p90은 실물 촬영이나 reset의 가치를 판정하지 않으며, 알려진 pressure 폭은 배포 입력에 포함되지 않는다. 시뮬레이션 기전에 대한 근거는 강해졌지만, 새 실물과 자연적으로 생긴 contact patch에서도 손실이 줄어든다는 주장은 아직 검증이 필요하다. [정보 접근·GPU 범위](GPU_KO.md) · [영상 생성](VISION_KO.md) · [Pressure 명세](PRESSURE_PROTOCOL_KO.md)

**다음에 가를 두 설명.** 같은 x에서 fold와 ripple의 손익이 갈린 원인이 보이는 국소 curvature라면, label 없이 영상에서 추정한 **접촉점 주변 normal의 분산**을 x에 추가한 rule이 x-only보다 개선돼야 한다. 새 contour family 두 개 이상에서 x-only·x+normal dispersion·always-original을 먼저 고정하고, 순위를 예측한 뒤 검증하는 편이 좋다. 실물 또는 자연 형성되는 patch의 외부 검증에서는 같은 최초 위치를 고속 영상으로 확인하고, 별도 pressure array로 면적·centroid·정규하중을 동시에 측정해야 한다. Surface session을 통째로 분리해 1차 비교를 visual−nominal로 두면, 절대 높이를 외운 효과와 관측의 가치를 구분할 수 있다. [Pressure ablation](pressure_mechanism45.json) · [물리 pressure 측정과 전이](https://arxiv.org/html/2501.08077v1)
