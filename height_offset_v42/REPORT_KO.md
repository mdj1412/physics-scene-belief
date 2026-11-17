# v42. 접촉점이 어긋날 때 어느 contour 정보를 보존해야 하는가

**접촉점을 정확히 맞추면 contour 변형 방향을 제거하는 편이 좋았다. 접촉점을 0.125 mm만 옮겨도 두 형상과 두 명목 높이를 합친 평균에서는 그 정보를 보존하는 편이 좋았다.** 여기서 projection은 카메라가 측정한 7점 외곽에서 변형 방향의 증거를 지우는 연산이다. 명목 접촉점에서는 세 외곽이 같은 기울기를 갖는다. 조금 옆에서는 기울기가 달라져, 이전까지 무관했던 변형이 충돌 결과를 예측하는 단서가 된다. 단, 같은 물체도 어느 쪽으로 빗나가느냐에 따라 결론이 달라졌다.

**네 번의 calibration으로 세 가지 보정 강도 중 하나를 고르는 방법은 주 비교에서 항상 0%를 쓰는 방법을 이기지 못했다.** Estimated projection에서 두 방법의 held-out Brier loss는 각각 0.165328, 0.165333이었다. 차이는 −0.000005이고 paired 95% CI는 [−0.000186, +0.000176]이다. 1,024개 물체의 처리 시간은 GPU p90 기준 12.62 ms 늘었다. 이 실험이 뒷받침하는 논문 기여는 접촉점의 불확실성에 따라 어느 기하 정보가 물리적 의미를 얻는지 식별한 결과다. 현재 selector를 사용한 성능 개선 주장은 지지되지 않았다.

![접촉 높이 오차별 실제 event flip과 두 decoder의 Brier 차이](offset42.png)

양의 offset은 실제 접촉점이 측정 중심보다 높음을 뜻한다. 같은 물체에 대해 contour를 바꾸었을 때 결과가 달라지는 비율과, 확률 예측에서 해당 정보를 지웠을 때의 손해는 서로 다른 양이다. 아래에서는 같은 물체·같은 높이의 대조로 둘의 관계를 검증한다.

## 같은 물체에 같은 관측을 주고, 실제 접촉점만 바꾸는 설계

로봇이 모서리를 밀기 전 카메라로 본 외곽은 같더라도, 팁이 목표 지점보다 조금 위에서 닿으면 충돌 시점의 기울기가 달라진다. v41에서는 base·hollow·bulge가 명목 접촉점에서 같은 위치와 같은 local tangent를 갖게 설계했다. v42에서는 외곽과 센서를 그대로 둔 채 실제 접촉점만 옮겼다. 첫 질문은 높이 오차가 생긴 순간 주변 외곽이 물리적으로 중요해지는지, 둘째 질문은 그 외곽을 보는 것이 실제 확률 예측을 개선하는지다.

| 항목 | 구현과 범위 |
|---|---|
| 새 catalog | v41과 독립인 12개 panel에 두 shape family(flare·roll), 명목 높이 19·31 mm, 세 contour(base·hollow·bulge), 두 contact law(saturating·crossover)를 교차했다. 형상·법칙의 생성식은 이전 연구와 같다. 24개 cell × 12 panel=288개 archive다. 각 panel에서 calibration 시작 방향을 번갈아 배치했다. |
| Signed offset | δ=실제 높이−명목 높이. 물리적 query만 0, ±0.125, ±0.25, ±0.5, ±1 mm로 옮겼다. 한 번의 실험에서 환경의 배포 물체가 같은 δ를 공유한다. 위치 센서가 그 오차를 알려 주지 않는 조건이다. |
| 물리 | Query 충돌 속도는 +0.425다. 위치가 바뀌면 해당 높이의 local tangent로 상대 속도를 분해하고 static-friction impulse 한계에서 stick/slip을 판정한다. 표면 방향 2개와 near-field 마찰 방향성 strong·weak·null을 모두 평가한다. 외곽 변형이 없는 base에서도 높이에 따른 기울기와 마찰 변화가 있어 event 자체가 달라질 수 있다. |
| 입력과 구매 비용 | 환경당 네 번의 calibration(마지막 +1.10, 23 mm). 8개 환경 각각 128개 배포 물체에 frozen K256이 query view 115개와 probe 13개를 배정했다. 8개의 sensor noise draw는 평가 반복이다. 한 결정을 내릴 때는 128개 물체에서 구매한 해당 회차 관측만 쓴다. |
| Geometry | 명목 접촉점의 ±6 mm를 2 mm 간격으로 읽은 7점 profile, noise σ=0.30 mm. Estimated는 별도 같은 물체의 두 contour 상태를 물체 32개에서 읽어 paired difference의 첫 principal component를 구한다. 동일 panel·shape·명목 높이에 한 방향을 재사용한다. 방향의 amplitude·stick label을 쓰지 않지만 좌표 등록, pair 대응, 같은 변형을 공유하는 group과 sensor scale은 필요하다. |
| Decoder | Original은 7개 좌표를 모두 쓴다. Estimated와 oracle은 각각 추정한 방향과 simulator의 실제 변형 방향을 whitened likelihood에서 제거한다. 세 방법의 probe likelihood는 동일하다. Bank는 16,384개 source hypothesis와 동일한 환경별 두 prior alignment parameter를 사용한다. Acquisition은 이전 source 기준을 유지한다. 주 비교에서 observation template과 후보 stick event는 모두 알려진 명목 높이에 맞춘다. |
| Selector | 같은 base 확률의 log-odds에 βzc를 더한다. Signed cue z, β와 optimizer의 보정량 w는 기존 설정을 쓰고 c∈{0, 0.325, w}에서 고른다. Learned selector는 v41의 label로 후보별 Brier 차이에 필요한 residual만 회귀한다. 한 배치의 물체 128개에 같은 c를 적용한다. 실제 offset·law 이름·결과는 주지 않는다. |

**Source에서 이미 드러난 불확실성.** V41로 학습한 세 정책 selector는 occupancy feature와 ridge penalty 0.01을 선택했다. 과거 panel 또는 명목 높이를 통째로 가렸을 때 estimated selector−0%는 −0.000370, −0.000210이었고, shape family를 가리면 +0.002378이었다. 이 세 검증을 동일 가중하면 상수 0%가 우세했다. 따라서 v42는 **source에서 가장 좋은 비상수 후보**와 **상수를 허용한 보수적 source 선택**을 모두 새 catalog에 적용한 실험이다. 훈련한 자료에는 이번의 hidden offset label이 없었다. [선택식·학습 분할](SELECTOR_KO.md) · [동결 계수](selector_model42.json)

**어떤 차이를 ‘손해’로 판정했는가.** 각 물체의 stick 확률 p와 실제 event Y에 대해 (p−Y)²를 계산한다. 0%에서 D(δ)=Brier(estimated)−Brier(original)로 두고, 양의 동시 95% 구간을 projection 손해로 판정한다. 9개 pooled offset과 36개 shape×height×offset 대조를 각각 별도로 보정한다. Selector의 주 대조는 0을 제외한 8개 signed offset의 균등 평균이다. 모든 paired interval의 독립 단위는 12개 panel이다. 대조를 이번 장비의 위치 오차 분포나 연속적인 임계 높이로 외삽하지 않는다. [사전 계획](PLAN_KO.md) · [분석 코드](stats42.py)

## Signed placement error와 물리적 전환

아래 flip은 같은 물체·같은 signed offset에서 hollow 또는 bulge가 base와 다른 실제 event를 낸 비율이다. 무변형 base는 flip 비율의 분모에서 제외했다. Brier는 base까지 포함한 24개 cell의 동일 가중 평균이다. 양의 Brier 차이는 앞의 방법의 손해다. 각 Δ 열의 구간은 9개 signed offset에 대한 동시 95% 구간이다.

| δ (mm) | Contour event flip | 자기 contour의 δ=0 대비 flip | Original 0% | Estimated 0% | Estimated selector | Estimated−original, 0% | Selector−0%, estimated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 23.7% | 16.0% | 0.16445 | 0.17704 | 0.17738 | +0.01258 [+0.01120, +0.01396] | +0.00035 [+0.00002, +0.00067] |
| -0.5 | 14.9% | 10.0% | 0.14017 | 0.14851 | 0.14857 | +0.00833 [+0.00711, +0.00956] | +0.00007 [-0.00024, +0.00037] |
| -0.25 | 8.9% | 6.0% | 0.12973 | 0.13527 | 0.13515 | +0.00554 [+0.00446, +0.00662] | -0.00012 [-0.00042, +0.00018] |
| -0.125 | 5.1% | 3.4% | 0.12503 | 0.12715 | 0.12672 | +0.00212 [+0.00126, +0.00298] | -0.00043 [-0.00071, -0.00015] |
| +0 | 0.0% | 0.0% | 0.12593 | 0.12218 | 0.12154 | -0.00375 [-0.00411, -0.00338] | -0.00064 [-0.00093, -0.00035] |
| +0.125 | 12.0% | 8.0% | 0.13272 | 0.14162 | 0.14158 | +0.00890 [+0.00761, +0.01018] | -0.00004 [-0.00037, +0.00028] |
| +0.25 | 21.7% | 14.5% | 0.15402 | 0.16982 | 0.16955 | +0.01580 [+0.01416, +0.01744] | -0.00027 [-0.00053, -0.00000] |
| +0.5 | 29.9% | 20.0% | 0.18248 | 0.20259 | 0.20280 | +0.02011 [+0.01825, +0.02197] | +0.00022 [-0.00006, +0.00049] |
| +1 | 35.4% | 23.7% | 0.19832 | 0.22067 | 0.22087 | +0.02235 [+0.02052, +0.02419] | +0.00019 [-0.00008, +0.00046] |

### 어느 물체에서 projection이 손해였는가

같은 0% 보정에서 estimated−original의 Brier 차이(×1,000)다. **굵은 값**은 36개 사전 대조를 동시에 보정한 구간에서 부호가 결정된 셀이다. 높이는 flare·roll의 명목 접촉 위치이고 첫 행은 실제 이동량이다.

| Shape·높이 / δ (mm) | -1 | -0.5 | -0.25 | -0.125 | +0 | +0.125 | +0.25 | +0.5 | +1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flare / 19 mm | **+21.89** | **+15.04** | **+12.08** | **+7.46** | **-2.91** | **+16.48** | **+25.22** | **+26.43** | **+26.44** |
| flare / 31 mm | **-13.77** | **-11.46** | **-8.40** | **-5.93** | **-2.95** | **+5.04** | **+8.18** | **+11.05** | **+15.30** |
| roll / 19 mm | **+16.15** | **+11.81** | **+6.55** | **+2.51** | **-2.46** | **+14.28** | **+24.39** | **+30.60** | **+30.75** |
| roll / 31 mm | **+26.06** | **+17.95** | **+11.93** | **+4.43** | **-6.67** | -0.22 | **+5.42** | **+12.34** | **+16.93** |

### 보정을 고르는 이득과 계산 비용

아래 Brier는 명목 높이를 제외한 8개 signed offset을 동일하게 가중한 주 분석이다. Oracle은 reference profile의 실제 변형 방향을 알려 준 decoder다.

| Decoder | 항상 0% | 항상 32.5% | 항상 v38 optimizer | 세 정책 selector | Selector−0%, paired 95% |
|---|---:|---:|---:|---:|---:|
| Original | 0.153367 | 0.154194 | 0.155132 | 0.153088 | -0.000279 [-0.000452, -0.000105] |
| Oracle projection | 0.166205 | 0.167025 | 0.167750 | 0.166238 | +0.000033 [-0.000143, +0.000210] |
| Estimated projection | 0.165333 | 0.166152 | 0.166899 | 0.165328 | -0.000005 [-0.000186, +0.000176] |

Primary selector−always-zero는 -0.000005 [-0.000186, +0.000176]였다. 12개 panel 중 7개에서 selector의 nonzero-offset 평균 Brier가 더 낮았다. 9개 높이를 모두 포함하면 -0.000075 [-0.000255, +0.000104]다. Estimated 안에서 selector−always-fixed와 selector−always-optimizer의 두 동시 95% 구간은 각각 -0.000824 [-0.000986, -0.000662] 및 -0.001571 [-0.001934, -0.001208]였다.

| 동일 GPU, 1,024개 물체 | Warm p90, ms [95%] | Cold p90, ms |
|---|---:|---:|
| Original + 0% | 1,426.94 [1,417.98, 1,430.19] | 1,432.84 |
| Estimated + 0% | 1,423.51 [1,419.18, 1,425.14] | 1,430.30 |
| Estimated + 32.5% | 1,432.44 [1,426.71, 1,437.66] | 1,439.28 |
| Estimated + optimizer | 1,436.84 [1,432.25, 1,437.96] | 1,443.68 |
| Estimated + selector | 1,436.13 [1,431.43, 1,439.25] | 1,442.71 |

Warm selector / zero p90 ratio는 1.0089× [1.0057, 1.0125]였다. 한 GPU에서 12개 panel × 4회 순서를 섞어 측정했고 panel을 함께 bootstrap했다. Warm에는 acquisition, posterior inference, 실제 correction 결정, 장치 동기화가 들어간다. Cold는 메모리에 있는 32쌍의 direction fitting과 template setup까지 더한 값이다. 물리적 촬영·정렬과 최초 bank 생성은 두 값에서 제외했다. [전체 수치](stats42.json)

### 실제 높이를 알려 주거나 변형 방향을 부분적으로 보존하면

첫 32개 물체와 첫 noise만 미리 정해 12개 panel × 24개 cell에서 보조 비교했다. 숨은 높이 control은 GPU와 같은 확률을 CPU로 재구성했다. Revealed control에는 실제 높이를 주고 동일 posterior의 source hypothesis마다 stick event만 새로 계산했다. Finite-variance는 같은 profile pair의 분산에서 sensor noise를 빼고, 해당 축 정보를 일부 남기는 v41 likelihood다. 모두 0% 보정이다.

| δ (mm) | 숨긴 높이 original | 숨긴 높이 estimated | 실제 높이 original | 실제 높이 estimated | 숨긴 높이 finite-variance |
|---:|---:|---:|---:|---:|---:|
| -1 | 0.16613 | 0.17870 | 0.16495 | 0.17770 | 0.17563 |
| -0.5 | 0.14159 | 0.14978 | 0.14118 | 0.14946 | 0.14754 |
| -0.25 | 0.13071 | 0.13604 | 0.13053 | 0.13591 | 0.13430 |
| -0.125 | 0.12571 | 0.12778 | 0.12566 | 0.12774 | 0.12654 |
| +0 | 0.12653 | 0.12273 | 0.12653 | 0.12273 | 0.12265 |
| +0.125 | 0.13380 | 0.14256 | 0.13387 | 0.14260 | 0.13904 |
| +0.25 | 0.15512 | 0.17075 | 0.15535 | 0.17093 | 0.16546 |
| +0.5 | 0.18371 | 0.20355 | 0.18444 | 0.20418 | 0.19719 |
| +1 | 0.19941 | 0.22163 | 0.20047 | 0.22264 | 0.21459 |

명목 높이에서 CPU와 GPU 확률의 최대 차이는 1.40e-13였다. 이 대조는 실제 높이만 알려 주며 source bank가 가진 quadratic contour 및 friction 가정은 유지한다. 유효한 접촉 기하를 완전히 아는 oracle로 해석할 수 없다. Finite-variance는 주 실험 실행 중 명세한 보조 baseline이다. [Readout control](reveal42.json) · [Finite-variance 결과](soft42.json)

## 0.025 mm 간격에서 손익이 바뀌는 범위

사전 지정한 아홉 점 사이의 경계를 좁히기 위해 같은 held-out 물체를 81개 높이에서 다시 채점했다. 모델의 새 학습이나 독립 재현은 없었다. 4개 shape·height × 81개 높이의 324개 구간을 함께 보정했다. 각 행은 0에서 해당 방향으로 움직이면서 최초로 손해가 확인된 점과 그 전에 가장 가까운 이득이 확인된 점이다. 마지막 열은 최초 손해 이후 더 멀리 가서 다시 이득이 확인됐는지를 나타낸다. 수치는 장치의 보편적 허용오차가 아니다.

| Shape / 명목 높이 | 이동 방향 | 마지막 확인 이득 (mm) | 최초 확인 손해 (mm) | 이후 확인된 이득 복귀 |
|---|---|---:|---:|---|
| flare / 19 mm | 아래 (−) | +0.000 | -0.050 | 확인 안 됨 |
| flare / 19 mm | 위 (+) | +0.000 | +0.025 | 확인 안 됨 |
| flare / 31 mm | 아래 (−) | -1.000 | 확인 안 됨 | 확인 안 됨 |
| flare / 31 mm | 위 (+) | +0.000 | +0.075 | 확인 안 됨 |
| roll / 19 mm | 아래 (−) | +0.000 | -0.150 | 확인 안 됨 |
| roll / 19 mm | 위 (+) | +0.000 | +0.050 | 확인 안 됨 |
| roll / 31 mm | 아래 (−) | -0.050 | -0.125 | 확인 안 됨 |
| roll / 31 mm | 위 (+) | +0.075 | +0.200 | 확인 안 됨 |

원래 아홉 점에서는 독립 물리 식의 label을 주 저장값과 개별 물체마다 대조했고, 여기서 재계산한 projection Brier 차이도 주 분석과 일치했다. [미세 scan 전체 구간](dense42.json) · [재계산 코드](dense42.py)

## 물리적 원인과 예측 가능성

**높이가 달라져서 생기는 변화와, 높이가 달라진 뒤 contour 자체가 만드는 변화를 분리했다.** 같은 높이의 base와 hollow·bulge를 비교하면, 실제 stick/slip이 달라지는 비율은 명목 위치에서 0%, −0.125 mm에서 5.11%, +0.125 mm에서 11.96%였다. 물체의 예측 확률을 고정했기 때문에 projection 손익의 높이별 변화는 정확히 −2 E[(p_est−p_orig)(Yδ−Y₀)]다. 같은 위치의 base label로 대체해 채점하면 contour 고유의 기여를 분리할 수 있다. +0.125 mm에서 projection 손익은 −0.00375에서 +0.00890으로 변했다. 약 +0.01264의 변화는 거의 전부 같은 위치에서 변형된 외곽이 실제 event를 바꾼 경우로 설명됐다. 이 simulator에서 마찰 한곗값을 base의 값에 고정해도, local slope가 바꾼 impulse demand만으로 실제 contour flip의 98.30%를 재현했다. [독립 방정식과 물리 검산](PHYSICS_AUDIT_KO.md)

**손해 방향은 이동 거리만으로 예측할 수 없다.** Flare의 31 mm 지점에서 −0.5 mm 이동하면 projection이 Brier를 0.01146 낮췄고, +0.5 mm 이동하면 0.01105 높였다. 음의 방향에서는 profile 전체에 적합한 apparent slope가 실제 접촉점의 slope 변화를 거꾸로 가리켜, 부정확한 단서를 지우는 편이 나았다. 보조 분석에서는 두 slope를 이용해 hollow·bulge의 손익 부호를 사전 예측했다. 32개 조건 중 30개가 맞았고, 항상 손해로 예측하면 27개가 맞았다. 실제 offset과 생성한 외곽을 알아야 계산되는 진단이다. 미래의 미지 환경에서 사용할 수 있는 배포 성능으로 환산할 수 없다. [사전 예측과 실제 결과](PHYSICS_AUDIT_KO.md) · [형상별 slope 그림](geometry42.png)

이 설계에서 배포 접촉의 일부는 명목 높이에 정확히 맞고, 나머지는 여덟 비영 offset 중 균등하게 발생한다고 가정할 수 있다. 이때 잘못 배치되는 접촉 비율이 **23.8%** [panel bootstrap 95% CI 22.3%, 25.6%]를 넘으면, 항상 0%를 적용할 때 original의 기대 Brier가 estimated projection보다 낮아진다. 실제 장비에서 측정한 오차 분포를 넣으면 경계도 달라진다. [배포 비율 민감도](deployment_mix42.json)

## Selector는 무엇을 알아냈고 무엇을 놓쳤는가

세 보정의 선택 비율은 0%가 68.8%, fixed가 16.5%, optimizer가 14.6%였다. 선택 비율을 그대로 유지하면서 환경과의 대응만 없앤 비교보다 selector의 Brier는 0.000266 낮았다. 환경별 선별에 신호가 있었지만, 항상 0%라는 더 강한 기준에 비해 줄인 loss는 약 0.000005에 그쳤다. Saturating law에서는 selector−0%가 −0.000272, crossover law에서는 +0.000262였다. 두 law의 효과가 서로 상쇄됐다. 주 대조의 구간은 사전 지정한 실용적 개선 0.003을 배제한다. 반면 original decoder에 같은 selector를 사용한 보조 대조에서는 −0.000279 [−0.000452, −0.000105]의 작은 개선이 있었다. Original+selector 조합의 GPU p90은 측정하지 않았다. [선택 빈도와 환경별 반사실 대조](SELECTOR_AUDIT_KO.md)

**어떤 보정이 좋은지 알 기회 자체는 남아 있었다.** 같은 환경 안에서 약 64개 별도 물체 각각에 무작위 비영 offset 한 개를 적용해 충돌 결과를 하나씩 얻고, 그 결과로 세 보정 중 하나를 고른 뒤 나머지 물체에서 채점했다. 실제 offset을 알려주지 않아도 항상 0%보다 Brier가 **0.003784** 낮았다 [95% CI 0.003387, 0.004181 감소]. 따라서 숨은 높이 하나만으로 주 selector의 실패를 설명하기 어렵다. 이 보조 실험은 같은 환경에서 추가 target 충돌 결과를 약 64개 구매한 설정이다. 네 번의 calibration 예산에서 해당 개선을 달성했다는 근거는 없다. [물체 분리와 획득 비용](SELECTOR_AUDIT_KO.md)

같은 물체가 여덟 숨은 접촉 높이에 걸쳐 서로 다른 event를 낸 비율은 25.7%였다. 물체와 법칙을 완전히 알더라도 어느 높이에서 닿는지를 모르면 이 균등 혼합에서 최저 가능한 Brier는 0.04833 [0.04776, 0.04890]이다. 실제 높이를 decoder의 마지막 readout에만 넘긴 보조 실험에서도 +1 mm에서는 original의 Brier가 0.19941에서 0.20047로 오히려 소폭 증가했다. 높이를 측정한 뒤 그 위치의 기울기를 표현할 수 있도록 shape hypothesis까지 바꾸는 대조가 필요하다. 변형 축의 정보를 일부 남기는 finite-variance likelihood는 같은 +1 mm에서 hard projection의 0.22163을 0.21459까지 낮췄으나 original의 0.19941에는 미치지 못했다. [숨은 높이의 loss 하한](HIDDEN_FLOOR_KO.md) · [Readout 대조](reveal42.json) · [Finite-variance 대조](soft42.json)

## 논문으로 발전시킬 판단

가장 설득력 있는 결과는 **주변 외곽을 지워도 되는 조건을 물리적으로 정의하고, 그 조건이 깨질 때 확률 예측의 손익까지 뒤집히는 과정을 같은 물체에서 확인한 점**이다. 외곽을 바꾸고 실제 접촉 위치를 고정한 대조, 위치만 바꾸고 센서를 고정한 대조, 같은 likelihood에서 보정 정책만 바꾼 대조가 서로 다른 원인을 구별한다. 이미 알려진 invariance의 위험을 넘어서는 부분은 signed placement error, local tangent, event flip과 Brier의 관계를 수치로 검증했다는 점이다. 현재 근거는 합성 benchmark의 mechanism study를 지지한다. 독립 catalog는 기존 두 shape family와 두 analytic contact law에서 추출했다. 가장 좁은 lobe의 폭은 4 mm이고 point contact를 사용했다. 실물의 허용 오차, 새로운 material에서의 재현성, 임의 shape에 대한 무감독 policy 개선은 미검증이다.

다음 실험에서 먼저 바꿀 것은 selector의 복잡도보다 **접촉 위치의 측정과 국소 기울기의 표현**이다. 실제 높이 측정 유무와 quadratic·local spline contour bank를 교차하고, tip radius를 달리해 point contact에서 보인 손익 반전이 finite contact patch에서도 남는지 본다. 실물에서는 접촉부 영상으로 위치와 stick/slip을 따로 기록하고, 새 물체 단위로 분할해 signed offset별 paired Brier를 평가한다. Selector를 계속 개발할 근거는 환경당 추가 충돌을 0·4·16·64회로 늘렸을 때 어느 비용부터 항상 0%를 일관되게 이기는지에 있다. 두 실험 모두 원래 decoder와 0%를 유지하고, 추가 센서·충돌 비용을 포함한 작업 시간도 기록해야 한다.

## 선행연구와 남는 질문

다음 문헌은 **관측을 바꾸어도 정답이 유지되는 범위**, **접촉 모델의 물리적 식별**, **기준 정책을 바꿀 근거**라는 세 질문으로 선정했다. 자료의 단위가 실물 충돌, 영상, 음성, 모의 환경으로 다르므로 성능 수치를 서로 합산하지 않았다.

| 연구 | 질문 | 방법·표본 | 결과·결론 | v42 해석의 한계 |
|---|---|---|---|---|
| Fazeli et al. | 실제 impact에서 contact parameter와 결과를 얼마나 식별할 수 있는가? | 로봇으로 평면 물체를 2,000회 낙하시켜 유효한 1,718회에 여섯 rigid-body model을 비교했다. | 모델이 표현할 수 있는 impulse 범위와 parameter 해석에 한계가 있었다. | 실물 impact 근거가 있지만 sub-mm 높이 오차와 paired contour intervention으로 확률 예측을 대조한 자료는 없다. [원문](https://arxiv.org/pdf/1710.04979) |
| Meißenhelter et al., ICRA@40 2024 | 위치의 불확실성이 contact normal과 충돌 이후 상태에 어떻게 전파되는가? | Armadillo·Bunny·Cup의 3D mesh를 sphere hierarchy로 근사하고 normal 오차를 cone으로 모델링했다. Simulation 100,000회 참조 분포와 Hellinger distance를 비교했다. | 제안한 surrogate가 linearization보다 참조 분포에 가까웠고 sampling보다 계산이 빨랐다. | Isotropic Gaussian을 맞춘 위치 분포가 metric이다. Signed offset에 따라 같은 contour가 stick/slip 정보를 얻는지와 보정 선택은 평가하지 않았다. [원문 §II–III](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) |
| Benton et al., NeurIPS 2020 | 어느 정도의 변환까지 label이 유지되는가? | Augerino가 augmentation 분포를 학습한다. 두 캐릭터와 두 방향을 조합한 Mario–Iggy 분류 및 회전 각도를 예측하는 Olivetti faces 등을 평가했다. | 전 회전에 대한 invariance가 class를 합치는 과제에서 제한된 범위의 invariance를 학습했다. | 변환 범위를 제한해야 한다는 아이디어는 선례가 있다. 영상 회전에는 접촉 위치·마찰에 따른 event 전환이 없다. [원문 §4](https://papers.nips.cc/paper_files/paper/2020/file/cc8090c4d2791cdd9cd2cb3c24296190-Paper.pdf) |
| Zhao et al., ICML 2019 | Domain-invariant representation과 낮은 source error가 전이를 보장하는가? | 반례·일반화 bound와 MNIST·USPS·SVHN 세 digit dataset의 네 transfer 방향을 평가했다. | Conditional shift가 있으면 동일 표현이 다른 정답에 대응할 수 있고 target 성능이 악화될 수 있었다. | 분포 사이의 일반 이론이다. v42의 signed offset·Brier 변화량이나 실물 임계 높이를 예측해 주지는 않는다. [원문 §4–5](https://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf) |
| Hosseini-Kivanani, TidyVoice 2026 preprint | 같은 화자의 언어별 차이를 제거해 verification을 개선할 수 있는가? | Training 3,666명, development 808명, evaluation 2,200명. 같은 화자의 cross-language embedding 차이에 PCA를 적용해 직교 방향으로 투영했다. | Development EER는 2.70%에서 2.18%로 줄었다. 같은 대조에서 minDCF는 0.64에서 0.73으로 커졌다. | Paired-difference projection의 직접적인 방법 선례다. 언어가 달라져도 화자 ID는 같고, 물리적 변형으로 접촉 정답이 바뀌는 v42와 다르다. [원문 §2.3·3.1](https://arxiv.org/html/2607.22923v1) |
| Laroche et al., ICML 2019 | 기존 정책으로 모은 제한적인 자료로 그 정책을 개선할 수 있는가? | Stochastic gridworld와 무작위 finite MDP에서 자료가 부족한 state/action에는 기존 정책을 사용하는 SPIBB를 평가했다. | 지정된 가정에서 baseline보다 악화될 위험을 통제할 수 있고, 모의 평가에서 평균 성능도 개선할 수 있었다. | Offline RL의 sequential return과 batch별 Brier는 다르다. v42의 ridge selector에는 SPIBB 보장이 없다. [원문](https://proceedings.mlr.press/v97/laroche19a.html) |
| Zhong et al., ActivePusher, ICRA 2026 | 측정과 residual physics로 실제 manipulation을 개선할 수 있는가? | 모의 물체 4종·실물 2종에서 물체당 독립 test action 1,000개를 사용했고, UR10의 실제 planning도 평가했다. | 적은 자료에서 residual physics와 선택적 data acquisition이 prediction과 planning에 도움이 됐다. | 여기는 모델과 acquisition을 함께 바꾼다. v42는 이미 구입한 관측을 고정하므로 실제 행동 성공률이나 접촉 횟수 절감을 추론할 수 없다. [원문 §V](https://arxiv.org/html/2506.04646v4) |

**Knowledge map과 증거 강도.** 물체 대응이 주어짐 → paired difference에서 변동 축 추정 → decoder가 그 축에 불변이 되는 과정은 실제 음성에서도 구현됐다. 변환이 label을 보존할 범위가 과제에 달린다는 점은 Augerino의 실험과 Zhao 등의 이론·반례가 함께 지지한다. 특정 장비의 높이 오차가 어느 방향에서 stick을 slip으로 바꾸는지는 이 연구들로 정할 수 없다. 실물 충돌에서 parameter 식별에 한계가 있고 residual physics가 실제 행동을 도울 수 있다는 관찰은 각각 근거가 있지만, 한쪽은 impulse 적합도, 다른 쪽은 새로 고른 action과 planning 성공을 측정했다. 두 연구의 결과가 달라 보이는 이유는 outcome·데이터 수집 방식·대상 물체가 다르기 때문이다. [Paired projection](https://arxiv.org/html/2607.22923v1) · [제한된 invariance](https://arxiv.org/abs/2010.11882) · [Conditional shift](https://proceedings.mlr.press/v97/zhao19a.html) · [충돌 모델](https://arxiv.org/pdf/1710.04979) · [잔차 모델과 행동](https://arxiv.org/html/2506.04646v4)

**무엇이 열려 있는가.** 검토한 일곱 논문은 동일 물체·동일 관측의 contact height만 바꿔 contour별 label 전환과 projection risk의 부호를 동시에 검증하지 않았다. 이 범위에서 v42의 차별점은 label 변화가 확률 오차를 *얼마나, 어느 방향으로* 움직였는지 counterfactual로 확인하는 데 있다. Paired PCA의 최초성, 모든 projection에 대한 실패 경계, 실제 공정에서의 안전성은 이 문헌 비교로 입증할 수 없다. 보정 선택에서도 baseline으로 돌아갈 수 있어야 한다는 선례가 있으므로 항상 0%와 이기는지 묻는 대조가 필요하다. [Augerino](https://papers.nips.cc/paper_files/paper/2020/file/cc8090c4d2791cdd9cd2cb3c24296190-Paper.pdf) · [NAP](https://arxiv.org/html/2607.22923v1) · [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html)

### 근거에 따른 후속 연구 질문 10개

| 질문 | 이 질문이 필요한 이유 | 변수·검증 설계 |
|---|---|---|
| 1. 실측한 placement error에서도 같은 부호 반전이 발생하는가? | 실제 impact model의 오차와 실험의 analytic law는 다르다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 접촉부와 비접촉부를 독립 교체해 실제 높이, event, 확률을 측정하고 물체 단위 holdout한다. |
| 2. 유한한 contact patch는 반전 폭을 넓히는가? | Point impact에선 접촉 모델이 표현할 수 있는 impulse가 제한된다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | Tip radius·compliance와 signed offset을 교차하고 point/patch model의 flip curve를 비교한다. |
| 3. 실측 camera–tool registration noise를 알려주면 어떻게 달라지는가? | 같은 표현에서도 조건 분포가 바뀔 수 있다. [Zhao et al.](https://proceedings.mlr.press/v97/zhao19a.html) | Encoder 위치·영상 정합 잔차에서 offset prior를 먼저 추정하고 marginalization과 point readout을 비교한다. |
| 4. Label을 보존할 offset 범위를 source 외부에서 정할 수 있는가? | 허용 가능한 변환 범위 자체가 학습 대상이다. [Benton et al.](https://arxiv.org/abs/2010.11882) | Local derivative와 변형 길이 척도로 위험군을 사전 예고하고 별도 shape family에서 검증한다. |
| 5. 방향을 일부만 약화하면 넓은 범위를 견디는가? | Invariance가 과도하면 class 정보가 사라진다. [Benton et al.](https://arxiv.org/abs/2010.11882) | Hard projection, finite-variance likelihood, raw likelihood를 동률 compute에서 frozen 비교한다. |
| 6. Shape grouping이나 scan pairing에 오류가 생겨도 적용 가능한가? | Paired-difference NAP는 같은 대상이라는 정보를 사용한다. [Hosseini-Kivanani](https://arxiv.org/html/2607.22923v1) | Pair mismatch와 sensor registration을 각각 바꿔 축 오차와 Brier를 측정한다. |
| 7. 언제 0%를 떠날지 정하는 보수적 규칙이 새 law에서도 우세한가? | Baseline 유지의 가치는 source support와 불확실성에 달렸다. [Laroche et al.](https://proceedings.mlr.press/v97/laroche19a.html) | Material family 단위 nested holdout에서 0%·hard selection·lower-confidence gate를 사전 비교한다. |
| 8. 단 한 번의 추가 측정으로 offset 방향을 구별할 수 있는가? | 관측이 informative할 때 selective acquisition의 이득이 난다. [Zhong et al.](https://arxiv.org/html/2506.04646v4) | 접촉 위치 sensor와 마찰 probe를 같은 시간 예산에서 비교해 sign identification 및 Brier를 채점한다. |
| 9. 환경당 한 물체만 도착하면 selector가 어디까지 작동하는가? | 조건별 데이터가 적으면 baseline improvement의 불확실성이 커진다. [Laroche et al.](https://proceedings.mlr.press/v97/laroche19a.html) | Batch size 1·4·16·128을 사전 지정하고 미래 물체를 보지 않은 채 판단한다. |
| 10. 확률·latency 이득이 작업 실패 감소로 연결되는가? | 실제 planning success와 model error는 다른 outcome이다. [Zhong et al.](https://arxiv.org/html/2506.04646v4) | Slip 비용·중단 기준·profiling 비용을 먼저 정한 다음 실물 cycle time과 실패율을 함께 잰다. |

## 서버 실행과 검산

2026년 11월 16일 `datacenter3`의 Slurm allocation 2744468에서 사용 직전 compute PID 0개, GPU memory 0 MiB를 확인한 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장으로 수행했다. PyTorch 2.7.1+cu128, CUDA 12.8를 사용했다. 주 계획·실행 코드·source selector 계수는 10:21:29 KST에 서버에서 SHA-256으로 동결했다. [동결 목록](freeze42.sha256) · [UTC 시각](frozen_utc.txt) · [실행 기록](results/completion.json)

세 agent가 catalog와 독립 물리 구현, source selector 학습·검산, GPU 실행·timing 검산을 분담했다. 288개 원자료와 12개 timing archive의 checksum을 확인했다. 서로 다른 물체 추출은 12,288회이며 복원 추출을 감안한 고유한 deployment prototype은 11,924개, calibration에서 실제 probe를 받은 고유 prototype은 384개였다. 방향 추정에는 12 panel×2 shape×2 height×32개 별도 물체의 paired profile, 총 3,072회 readout이 필요했다.

독립 물리 식으로 15,925,248개의 저장 stick label을 재생했다. 1,492,992개 환경별 Brier의 최대 재계산 오차는 8.33e-16, GPU timing 경로와 archive의 확률 차이는 최대 4.44e-16였다. 모든 contour의 oracle projection 확률 간 최대 차이는 1.05e-13, probe 가지에서 decoder 간 최대 차이는 0.00e+00였다. Brier 변화의 두 event 전환 항을 더한 값과 실제 차이의 최대 오차는 1.53e-16였다. [물리 원자료 검산](physics_audit42.json) · [수치 검산](audit_runner42.json)

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/height_offset_v42`다. 새 사본에서 기존 v18–v41 dependencies를 사용할 수 있게 한 뒤 GPU에서는 `run42.sh`를 실행한다. Raw archive는 `results/<shape>_<height>_<variant>_<law>/catalog_NN.npz`에, GPU 측정 원값은 `results/timing/`에 있다. [재현 안내](REPRODUCE_KO.md) · [GPU runner](runner42.py) · [물리 생성식](world42.py)

표와 그림을 다시 만들 때는 완료된 GPU 원자료에서 서버의 기존 Python 환경을 사용한다. 아래 CPU 단계는 과거 학습 모델이나 GPU 결과를 바꾸지 않는다.

```bash
export BELIEF_DEVICE=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" stats42.py
"$py" reveal42.py
"$py" soft42.py
"$py" dense42.py
"$py" dense_table42.py
"$py" mix42.py
"$py" display42.py
"$py" record42.py
"$py" report42.py
```
