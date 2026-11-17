# v67의 두 검증을 위한 문헌 근거와 독립 검토

이번 검토는 두 가지 판단에 필요한 원 연구 6편을 선택했다. 하나는 같은 형상을 다른 법칙으로 누르면 예측기가 버릴 수 있는 좌표도 달라지는가다. 다른 하나는 같은 충돌 영상을 다르게 읽는 일이 실제 네 번째 관측의 구매로 이어지는가다. 아래 원문이 제공하는 근거와, 그 근거에서 착안한 이번 연구의 검정을 구분했다.

## 원 연구 비교

| 연구 · 질문 | 표본 · 방법 | 관측한 결과 · 결론 | v67에서 보완할 한계 |
|---|---|---|---|
| Fazeli 등. 어떤 rigid contact law가 실측 충돌을 설명하는가? | 평면 낙하 2,000회 중 검사를 통과한 1,718회에서 6개 law를 식별하고 post-impact velocity와 허용 impulse 영역을 비교했다. | 실측이 각 law의 예측 가능 영역에 속하는 비율은 0.355–0.458이었다. 일부 spin reversal은 최적 파라미터로도 회수하기 어려웠다. | 특정 평면 장치와 연속량의 결과다. 이 수치로 v67의 stick 확률이나 Wendland 효과 크기를 정할 수 없다. [원문 §3–4](https://arxiv.org/html/1710.04979) |
| Johansson 등. Domain-invariant representation에서 정보를 잃으면 transfer risk가 어떻게 달라지는가? | 이론적 bound와 MNIST→MNIST 및 MNIST→MNIST-M에서 target digit 구성에 변화를 준 분류 실험. | Source/target density를 가깝게 하는 것만으로 target error를 보장하기 어렵고, representation의 non-invertibility와 support가 관여했다. | Contact law를 바꾸면 같은 입력의 label 자체가 바뀔 수 있다. 이론에서 v67의 Brier gap이나 부호를 가져올 수 없다. [원문 §3–5](https://proceedings.mlr.press/v89/johansson19a/johansson19a.pdf) |
| Bouthillier 등. 학습 절차에서 무엇을 반복하느냐가 개선 판단에 영향을 주는가? | 5개 task/architecture에서 data sampling, initialization, tuning 등 변동원을 분리했다. | 한 번의 split·초기화에 고정한 비교와 여러 변동원을 포함한 비교가 다른 개선 판단을 낼 수 있었다. | 어느 변동원이 이 contact 문제에서 큰지는 새 source axis와 target catalog를 교차해야 측정할 수 있다. [원문](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) |
| Shi 등. 실제 non-Lambertian 물체를 광원에 따라 관측하면 normal 추정법을 어떻게 평가할 수 있는가? | DiLiGenT 정지 물체 10개에 96개 방향 광원, 보정한 강도·방향, reference normal을 제공했다. | 동일 물체의 광원을 통제하고 non-Lambertian·uncalibrated photometric stereo를 비교할 수 있는 자료를 만들었다. | 독립 impact, marker identity, strain, 구매 기록이 없다. 하나의 물체에서 나온 여러 patch를 독립 session으로 셀 수 없다. [저자 자료 및 논문](https://sites.google.com/site/photometricstereodata/single) |
| Ren 등. Shape와 material을 독립적으로 통제할 수 있는가? | CNC 가공 형상 10종 × 재료 10종, 물체당 100개 calibrated lighting direction과 normal map. | 동일 형상을 여러 재료에서 비교할 수 있는 100개 실물 조합을 제공했다. Diffuse, specular, translucent material을 포함한다. | 정적 shape/material 요인 설계의 근거다. 두 조명에서 동일한 움직이는 물질점을 식별하거나 action을 구매한 자료는 없다. [원문 §3](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_DiLiGenT102_A_Photometric_Stereo_Benchmark_Dataset_With_Controlled_Shape_and_CVPR_2022_paper.pdf) |
| Neri 등. LED pulse의 실제 발광 시간을 어떤 계측값으로 확인할 수 있는가? | 한 driver·LED 시험계에서 shunt voltage·photodiode·camera를 비교했다. Camera 경로에서는 1–5 μs 및 100–500 ns pulse를 시험했다. | 5 μs 비교에서 상승 및 지속 시간은 대체로 맞았지만 photodiode 하강은 더 느렸다. 저자들은 readout 회로의 응답을 원인 후보로 제시했다. | 이 하드웨어의 지연 수치를 v67에 이식할 수 없다. 기준 edge, 지연 부호, exposure와의 관계, 보정 후 남은 오차를 따로 밝혀야 한다. [원문 §3.2–3.4](https://www.mdpi.com/1424-8220/24/13/4103) |

## 연구 관계와 근거의 강도

**형상에서 예측까지.** 같은 input·다른 contact law → 다른 impulse·stick/slip 가능성은 평면 충돌 실측에서 동기를 얻는다. 별도로 source 표본 → 학습된 축 → 보존·삭제한 좌표 → fitted forecast → held-out Brier라는 경로가 있다. 특정 좌표를 버리는 연산과 그 뒤에서 score를 읽는 함수가 다른 개입이라는 구분은 이론과 정합적이다. 두 경로가 이 연구에서 얼마나 겹치는지는 위 문헌에서 추정하지 않았다. [Fazeli](https://arxiv.org/html/1710.04979) · [Johansson](https://proceedings.mlr.press/v89/johansson19a/johansson19a.pdf) · [Bouthillier](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)

**빛에서 구매까지.** 광원·표면 → 영상 밝기 패턴 → 추적 위치, 그리고 발광·센서·exposure → 시간창 → 변형 부호 판독이 각각 측정 가능하다. 이 두 경로 이후의 abstention → 상위 정책 → 후속 물체의 네 번째 실제 구매는 v64/v65의 frozen 정책을 거치는 별도 단계다. 광원 통제와 시간축 응답에는 실측 근거가 있다. 잘못 읽은 marker가 실제 접촉에서 reference↔impact 구매를 바꾸는 빈도에는 직접적인 실물 근거가 부족하다. [Shi](https://sites.google.com/site/photometricstereodata/single) · [Ren](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_DiLiGenT102_A_Photometric_Stereo_Benchmark_Dataset_With_Controlled_Shape_and_CVPR_2022_paper.pdf) · [Neri](https://www.mdpi.com/1424-8220/24/13/4103) · [v65 실험 설계와 분모](../paired_light_pilot_v65/REPORT_KO.md)

**충돌하는 것처럼 보이는 결과.** v66에서 Wendland는 물리적으로 형상을 바꿀 때 label이 더 자주 뒤집혔지만, 예측기의 부분 억제는 Dispersive에 더 큰 손해를 줬다. 첫 수치는 hollow/bulge 개입에 대한 binary label 변화이고 두 번째는 같은 evidence를 읽는 두 확률 예측기의 paired Brier 차이다. 서로 다른 조건의 결과모형이나 손실이 같다는 가정이 없으므로 둘 중 하나를 오류로 볼 이유가 없다. 센서 연구에서도 Neri 등의 rise-edge 일치와 fall-edge 차이는 같은 파형의 서로 다른 측정이다. 후자를 전 구간에 통용되는 상수 지연의 증거로 쓰면 보정의 범위를 과대평가한다. [v66 확인 실험](../invariance_audit_v66/locked/LOCKED_RESULTS_KO.md) · [Neri의 Fig. 9](https://www.mdpi.com/1424-8220/24/13/4103)

## 연구 심사와 판정 규칙

- **Reversal.** 각 law에서 Wendland의 soft−full이 음수이고 Dispersive가 양수인지 각각 표시한다. 둘 다 양수가 돼도 interaction은 클 수 있다. 원래 방향의 *효과 크기 차이*, 두 family에서 *point estimate 부호가 반대인 경우*, interval로 양쪽 부호가 지지되는 경우를 따로 해석한다. Seed별 빈도와 catalog 간 변동을 함께 제시한다. [v66 출발점](../invariance_audit_v66/REPORT_KO.md) · [변동원 분리](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)
- **독립 변수.** 같은 target shape, input noise, factual observation과 forecast를 공유하고 law별 label만 교체했다면 추정량은 label mechanism에 대한 민감도다. 실제 law 변화가 factual trajectory, probe 및 forecast도 바꾸는 실험의 총 효과는 따로 측정해야 한다. 같은 seed에서 단위벡터의 부호가 뒤집혀도 projector는 동일하므로 축 간 변동은 |cos θ| 등 부호에 무관한 기준으로 읽는다. [실물 법칙 비교](https://arxiv.org/html/1710.04979) · [기존 v66 공통 readout](../invariance_audit_v66/EXP_LOCKED_RESULTS_KO.md)
- **오류의 단계별 분모.** 동일 파일 재생 sham은 결정성 대조다. 같은 조명에서 두 번 판독하는 sham은 재현성 대조다. 두 영상을 섞은 뒤 결과를 고르면 실제 한 번 구매하는 정책의 예산이 바뀐다. 검증용 strain·marker truth와 주입한 delay를 배포 입력에 주었는지 명시하고, 보정 전후 같은 pilot에서 두 조명−sham의 절대 percentage-point 차이를 비교한다. 차이를 낸 표면 수와 reference→impact, impact→reference 전환도 정보가 있다. [v64 행동 정의](../reflective_field_v64/REPORT_KO.md) · [v65 pilot 규약](../paired_light_pilot_v65/REPORT_KO.md) · [광학 timing](https://www.mdpi.com/1424-8220/24/13/4103)
- **남은 반론.** 확률 예측기의 개선에는 좌표의 정보와 finite readout의 최적화가 함께 들어간다. 또한 구매가 바뀌어도 purchase cost나 후속 Brier 개선은 보장되지 않는다. Simulated fault rate는 조작값이며 field prevalence를 추정한 수치가 아니다. 입력이 고정된 law intervention, 독립 calibration으로 얻은 delay, 실물 identity/strain 검증이라는 서로 다른 근거를 표에 표기하면 주장 범위가 명확해진다. [표현의 한계](https://proceedings.mlr.press/v89/johansson19a/johansson19a.pdf) · [실물 연구에서 비어 있는 endpoint](../paired_light_pilot_v65/REPORT_KO.md)

## 근거에서 도출한 후속 연구 질문 10개

| 질문 | 현재 공백과 중요성 | 조작·측정·설계 |
|---|---|---|
| 1. 각 law에서 형상별 손익 부호가 재현되는가? | v66은 두 형상에 한 세트의 law를 사용했다. | Shape·물체·noise를 pairing한 law intervention, 각 family의 soft−full Brier와 interaction. [v66](../invariance_audit_v66/REPORT_KO.md) |
| 2. 실제 label이 뒤집힌 곳 중 어느 곳이 손익 반전을 만드는가? | 같은 flip 빈도라도 양 정책의 확률 차이는 다르다. | 두 forecast를 고정하고 law에 따른 label 차이와 (p_soft−p_full)의 곱을 짝별로 합산. [v66](../invariance_audit_v66/REPORT_KO.md) |
| 3. 데이터 재추출과 head 초기화 중 무엇이 축 복원 효과를 흔드는가? | 두 변동원이 함께 바뀌면 단독 원인은 분리되지 않는다. | Source bank seed × head 초기화 교차, principal angle·paired Brier로 분산 비교. [Bouthillier](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) |
| 4. 같은 law라도 stick 전이 근처에서만 좌표가 필요한가? | 평면 실험에서 tangential momentum에 따른 오차 차이가 있었다. | 독립적으로 정한 pre-impact speed strata, signed 좌표가 있는 head와 없는 head의 구간별 Brier. [Fazeli](https://arxiv.org/html/1710.04979) |
| 5. Label만 바꾼 효과는 factual trajectory까지 새 law로 만들 때도 남는가? | Evidence shift와 label shift가 다른 전이 문제를 만든다. | 동일 물체에서 frozen-evidence와 law-consistent observation을 모두 생성하고 네 forecast–label 조합의 Brier를 비교. [Johansson](https://proceedings.mlr.press/v89/johansson19a/johansson19a.pdf) · [Fazeli](https://arxiv.org/html/1710.04979) |
| 6. 보정 가능한 timing error는 edge와 exposure에 어떻게 의존하는가? | 실제 광학계에서 rising/falling response가 달랐다. | Pulse 폭·강도·temperature를 바꾸고 독립 photodiode와 shutter exposure marker에서 residual delay를 측정. [Neri](https://www.mdpi.com/1424-8220/24/13/4103) |
| 7. 같은 lamp에서 반복 촬영을 하면 marker identity가 얼마나 자주 바뀌는가? | v65의 정지 영상 round trip은 독립적 재촬영을 대체하기 어렵다. | 동일한 충돌에 같은 방향의 optical split, 독립 영상 noise·coded marker로 오인식률 측정. [v65](../paired_light_pilot_v65/REPORT_KO.md) · [Shi](https://sites.google.com/site/photometricstereodata/single) |
| 8. 같은 형상에서 조명 효과의 크기가 재료에 따라 달라지는가? | 정지 이미지의 강한 반사와 충돌 중 추적 실패 사이의 연결이 아직 약하다. | Shape×material 설계에서 lamp 방향을 바꾸고 material point를 독립 채널로 확인해 sham 대비 track error를 집계. [Ren](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_DiLiGenT102_A_Photometric_Stereo_Benchmark_Dataset_With_Controlled_Shape_and_CVPR_2022_paper.pdf) |
| 9. Timing과 identity가 함께 틀릴 때 오류가 가려지는가? | 두 오류가 같은 sign 또는 반대 sign을 만들면 단독 발생률을 더해도 구매 전환을 설명할 수 없다. | 같은 pilot에서 두 요인을 독립 교차해 reader, gate, 실제 구매의 risk difference와 interaction을 계산. [Neri](https://www.mdpi.com/1424-8220/24/13/4103) · [v64](../reflective_field_v64/REPORT_KO.md) |
| 10. 관측 구매가 바뀐 pilot에서 prediction과 비용의 순이득은 얼마인가? | v65의 0.1863%는 simulation의 구매 불일치율이다. 손익이 반대인 전환을 합치면 실용성을 잘못 읽을 수 있다. | 독립 물체의 양 후보 관측과 outcome을 replay해 paired Brier·실제 측정/재설정 시간·전환 방향을 surface별 집계. [v64](../reflective_field_v64/REPORT_KO.md) · [v65](../paired_light_pilot_v65/REPORT_KO.md) |
