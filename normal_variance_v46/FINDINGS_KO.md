## 4. 왜 source의 이득이 새 family로 옮겨가지 않았나

**Source에서의 개선은 실제였지만 family를 바꾸자 사라졌다.** V45 안에서 catalog 하나를 통째로 제외하고 나머지 11개에서 구간 경계와 risk를 다시 학습하면, joint−refit x는 평균 −0.000220 [기술적 95% CI −0.000288, −0.000152]였다. 반면 fold만 학습해 ripple을 시험하거나 그 반대로 시험하면 영상 기반 joint와 refit-x가 같은 물체에서 같은 선택을 했다. Source의 단순 catalog split이 새 family에 대한 검증을 대신하지 못했다. [Source 교차검증](source_cv46.json)

**광학적으로 측정한 신호와 상대 손익은 달랐다.** 새 family에서 위치 MAE는 0.09333 mm였다. 영상 분산과 같은 길이의 해석적 secant로 구한 참 분산 사이의 Spearman ρ는 terrace 0.672, dimple 0.645였다. 전체 절대 분산 오차의 median은 0.00160 rad²이지만 p90은 0.17765 rad²였다. 사후에 동일 물체 55,296개 장면을 독립 camera nuisance로 재촬영했을 때 분산 순위의 재현성은 ρ=0.485·0.443까지 낮아졌다. 따라서 법선 변화의 일부를 영상으로 읽을 수 있지만, 특히 꼬리 오차가 크며 새 family의 loss 차이를 맞춘다는 근거는 별개다. [광학 검산](vision_audit46.json) · [독립 재촬영](repeat_camera46.json)

**측정 오차를 없애도 always-original은 이기지 못했다.** Source 단계부터 정확한 x와 해석적 normal variance를 사용한 oracle joint의 finite Brier는 **0.161492**였다. 정확한 x만 쓴 oracle의 0.162889보다 −0.001397 [두 진단 동시 95% CI −0.001543, −0.001250] 개선됐지만, always-original의 0.155170보다 여전히 높다. 이 차이에는 feature의 측정 정밀도와 해당 feature로 source에서 다시 학습한 표의 변화가 함께 포함돼 있다. 또 target의 11개 catalog에 label 접근을 허용하고 같은 source bin·shrinkage로 다시 fit하면, 남은 catalog에서 x-only와 joint 모두 always-original의 Brier와 같았다. 이 진단은 같은 target family의 label을 쓴 별도 실험이다. Source만으로 가능한 운영 성과로 계산하지 않았다. [Oracle 및 원 점수](score46.json) · [Leave-one-target-catalog-out](transfer46.json)

**실제로 지워지는 profile 방향이 달라졌다.** Source axis와 실제 hollow↔bulge 차이 벡터의 squared cosine을 계산했다. V45의 fold·ripple에서 이 축은 변형 신호 energy의 67.93%(19 mm)·58.28%(31 mm)를 제거했다. 새 terrace에서는 3.39%·11.23%, dimple에서는 19.19%·**0.036%**였다. 즉 같은 연산이 새 물체의 같은 종류의 변형을 제거한다는 가정이 깨졌다. [Paired profile의 각도 검산](axis_transfer46.json)

**축만 교체한 직접 대조.** 이 관찰이 손실에도 영향을 주었는지 확인하기 위해 primary 결과 확인 후 추가 GPU 실험을 했다. 새 family·명목 높이별 실제 hollow−bulge profile 차이의 단위 방향을 주고, 각 물체의 영상 route, evidence, decoder bank와 target label은 그대로 두었다. Family·짝지은 variant를 아는 privileged 진단이다. 축을 구성할 때 target outcome을 쓰지 않았고 배포 지표에는 포함하지 않았다. [진단 예측 코드](target_axis46.py) · [축 교차 평가](aligned_axis_score46.json)

| 같은 영상 route에서 decoder 축만 교체 | Finite pressure Brier | Rigid PyBullet Brier |
|---|---:|---:|
| Always-original | **0.155170** | **0.161305** |
| Joint + v45 source axis | 0.162760 | 0.169232 |
| Joint + target 변형 방향 | 0.155897 | 0.162009 |
| 축 교체 후 남은 손해 [세 진단 동시 95% CI] | **+0.000727** [+0.000468, +0.000985] | **+0.000704** [+0.000450, +0.000959] |

Finite pressure에서 축 교체는 Brier를 **0.006863** 낮췄다 [−0.007324, −0.006401]. 점추정으로 기존 초과 손실의 약 **90.4%**가 줄었다. 반면 교체한 축을 쓰는 상태에서도 joint와 위치 전용의 차이는 −0.0000043 [−0.0000120, +0.0000035]였다. 같은 잠재 물성·표면·최초 위치에서 hollow를 bulge로 바꾸면 finite pressure의 stick label 자체가 terrace의 16.20%, dimple의 18.25%에서 달라졌다. 변형에 물리적 예측 신호가 포함되는 사례가 있음을 뜻한다. 이 전환율만으로 남은 손해 전부가 그 경로에서 왔다고 단정할 수는 없다. [같은 route의 교차 실험](aligned_axis_score46.json) · [동일 물체 변형 대조](paired_variants46.json)

**Physics 설정의 민감도.** 전체 pressure 조건에서 quadrature 해상도를 두 배로 올리자 11개의 label이 바뀌었다. Stick threshold를 0.010 또는 0.040 m/s로 바꿔도 joint−always-original은 +0.007535~+0.007589였다(기본 0.020 포함). Rigid 검증에서도 전체 884,736회에서 접촉이 확인됐고 지정 위치와 실제 최초 접촉 x의 MAE는 0.00204 mm였다. V45 fold는 압력 면적을 넓히자 항상 projection하는 손해가 커졌지만, v46에서는 terrace −0.001746, dimple −0.001124만큼 손해가 줄었다. 두 family 모두 넓은 면적에서도 평균적으로 original이 유리했다. 따라서 v45의 면적 효과 부호 역시 geometry를 넘어 보존되는 법칙으로 쓸 수 없다. [Physics 검산](PHYSICS_RESULTS_KO.md)

## 5. 논문 기여 가능성과 다음 판정 기준

**방법 논문으로는 주가설이 기각됐다.** 등록한 joint rule은 두 baseline을 이기지 못했고, 추가 GPU 계산을 사용했다. Source catalog 내부의 개선이나 부분적으로 양호한 normal 측정만을 제시하면 family transfer를 과대평가하게 된다. V45의 양성 결과도 당시 fold·ripple 혼합, source axis와 접촉 조건의 범위에서 보고하는 것이 타당하다. [V45 결과](../image_pressure_v45/REPORT_KO.md) · [V46 primary](score46.json)

**기여 가능성이 커진 부분은 projection이 실패하는 이유를 분리한 실험이다.** 동일한 sensor·decoder 경로에서 (1) 실제 기하를 주면 위치 전용보다 나아지는 양, (2) 같은 route에서 축만 바꾸면 회복되는 양, (3) 변형 자체가 미래 물리 결과를 바꾸는 빈도를 각각 측정했다. 특히 31 mm의 dimple에서 거의 직교하는 축을 제거하고도 이를 올바른 nuisance 제거로 해석할 수 없다는 점, 그리고 정렬된 축으로 바꿔도 original이 이긴다는 점을 두 contact solver에서 동시에 확인했다. Geometry를 측정할 수 있다는 사실과 그 geometry를 지울 때 기대 loss가 감소한다는 판단 사이에 검증이 필요한 두 단계가 있다. [Axis 검산](axis_transfer46.json) · [교차 진단](aligned_axis_score46.json)

**다음 한 번의 확인 실험은 feature를 더 넣기 전에 축과 label의 관계부터 가르는 것이 좋다.** 완전히 새로운 여러 family를 정하고, 같은 x와 비슷한 normal variance를 갖되 변형 때문에 stick 결과가 바뀌는 쌍과 보존되는 쌍을 source 단계에서 설계한다. 배포 관측으로 추정한 axis alignment, 보수적인 always-original fallback, 현재 joint를 미리 고정한다. 새 family에서 original보다 낮은 Brier와 위치 전용보다 낮은 paired loss가 함께 재현될 때만 adaptive projection의 방법적 기여를 주장한다. 실물이나 독립 카메라가 확보되면 calibration session을 분리해 pressure·충돌 속도를 별도 계측하는 검증이 필요하다. 현재 결과는 두 합성 가족과 extruded 2D 외곽, 같은 물성 field 및 사실 관측 장치를 공유하는 simulator들에 한정된다. [사전 계획 및 한계](PLAN_KO.md) · [광학 대조](VISION_KO.md)
