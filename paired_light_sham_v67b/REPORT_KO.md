# v67b · 같은 조명에서도 바뀌는 구매와, 조명 방향을 바꿀 때 더해지는 구매

**두 오류를 함께 주입하고 독립 pulse 측정으로 지연을 보정한 뒤, 같은 조명에서는 0.16624%, 다른 조명에서는 0.30660%의 경우 네 번째 구매 추천이 달라졌다.** 두 조명에서 추가된 전환은 **+0.14036 percentage points**였고, 두 primary contrast를 함께 고려한 95% interval은 **[+0.02267, +0.31278]**이었다. 여덟 번의 같은 calibration impact를 두 방식으로 판독한 결과가, 같은 후속 물체에서 네 번째 관측을 reference(R)와 impact(I) 중 다르게 추천하는 비율이다. 이 shadow decision의 실제 집행이나 Brier 개선은 별도 endpoint다. [사전 설계](PLAN_KO.md) · [결과 JSON](results67.json)

## 1. 같은 충돌을 어떻게 비교했나

v64의 24개 catalog에서 48개 simulated surface, 8개 board, 12,288개 calibration bank, 서로 다른 impact slot 98,304개를 사용했다. 각 bank는 두 물체씩 4 session에 걸친 8개 impact다. 두 lobe width와 네 가상 물리 regime에 동일 bank를 재사용했다. A 또는 B 방향 조명의 자료를 각각 frozen reader에 주고 A→A, B→B, A→B, B→A를 비교했다. 두 조명을 결합하는 reader는 평가하지 않았다.

두 acquisition slot은 물체 운동·scene texture·v64 native noise·전자 센서·reference·후속 prefix를 공유한다. 각 impact frame에 독립 Gaussian read noise(SD 2.5 gray levels)를 추가했다. 같은 slot의 noise와 injection은 lamp 비교에서 공유한다. 따라서 실제 두 카메라의 모든 noise 성분이 독립인 경우는 이 설계보다 넓다. 같은 raw pixels와 waveform을 복사한 sham의 전환은 정확히 0이었다. v64 모드로 다시 생성한 모든 camera sign·encoded bank가 저장값과 같았고, 서로 다른 lamp의 전환율 **0.186299%**를 재현했다. [원래 endpoint 재생](../paired_light_pilot_v65/DRY_REPLAY_KO.md) · [실행 코드](sham67.py)

둘째 slot에 두 종류의 오류를 독립적으로 주입했다.

- **Photodiode delay.** LED와 camera frame을 고정하고 diode ADC 파형만 board별 0.625–1.3125 ms 늦췄다. Impact마다 독립 ±0.125 ms jitter를 더했다. 별도 commissioning에서 board마다 정상·지연 branch 각각 128개 pulse를 만들었다. 양 branch의 median diode-minus-switch edge 차이로 board offset을 추정해 test waveform에 적용했다. 각 test impact의 실제 delay는 추정기에 주지 않았다. Board별 nominal과 추정치의 절대 차이 최댓값은 0.0281 ms였다. Commissioning 후 offset이 유지된다는 가정이다.
- **Coherent marker decoy.** Impact별 확률 15%로 8 frame 전체에서 moving ROI의 96%를 반대 방향으로 움직이는 ghost와 혼합했다. 총 14,668/98,304 slot에 주입했다. 실제 reader가 이 ghost를 항상 선택하거나 오독한다고 가정하지 않았다. 원래 조명별 반사·가림·transient는 유지했다. Simulated true sign은 accepted-but-wrong 진단에서만 사용했다.
- **구매.** Reader가 보류하면 기존 electronics fallback을 유지한다. Frozen router가 fixed 3R+I를 고르면 네 번째는 I, v54 gate를 고르면 세 개의 공유된 reference로 R 또는 I를 추천한다. 각 bank와 같은 material의 admissible prefix 2,560개를 각각 짝지었다. Capture 자체가 유실된 사례는 0개로 설계했다. 두 route가 달라도 v54가 I를 택하면 terminal은 같다. [v54 gate](../adaptive_witness_v54/policy54.py) · [router](../sensor_reset_v59/route59.py)

## 2. 구매 추천이 달라진 비율

단위는 %다. 두 lobe와 순서를 동일하게 가중했다. 모든 bank와 raw에서 발생한 abstention을 포함했다.

| 둘째 slot에 넣은 오류 | Sham upper route | 두 조명 upper route | Sham 최종 구매 | 두 조명 최종 구매 |
|---|---:|---:|---:|---:|
| 추가 오류 없음, 두 slot의 read noise 독립 | 0.03866 | 0.31738 | 0.02511 | 0.20567 |
| Marker 단독 | 0.25584 | 0.47251 | 0.16491 | 0.30527 |
| Delay 단독, 보정 전 | 0.37130 | 0.51371 | 0.23927 | 0.33181 |
| Delay 단독, 보정 후 | 0.04069 | 0.31942 | 0.02644 | 0.20700 |
| Marker + delay, 보정 전 | 0.49591 | 0.60476 | 0.31941 | 0.39052 |
| **Marker + delay, 보정 후** | **0.25787** | **0.47455** | **0.16624** | **0.30660** |

**첫 번째 primary contrast.** 두 오류를 넣고 보정했을 때 두 조명−sham은 +0.14036 pp [동시 95% +0.02267, +0.31278]였다. Lobe width 8°에서 두 조명/sham은 0.27882/0.16263%, 26°에서는 0.33437/0.16985%였다. A가 먼저일 때 두 조명/sham은 0.29593/0.17984%, B가 먼저일 때는 0.31726/0.15264%였다. 한 catalog씩 제외한 차이 범위는 +0.12994~+0.14780 pp였고, 한 board씩 제외하면 +0.10107~+0.16338 pp였다. [독립 검산](attribution67.json)

**두 번째 primary contrast.** Delay 단독 sham에서 보정 전과 후의 차이는 0.21283 pp [동시 95% 0.08263, 0.38043]였다. Marker 단독과 기본 sham의 차이는 +0.13980 pp [개별 95% +0.04334, +0.26300]였다. Joint corrected의 갈림은 sham에서 R→I 0.1216%, I→R 0.0447%였다. 두 조명에서는 각각 0.192%, 0.115%였다. 어느 방향이 실제 예측을 개선했는지는 이 구매 지표로 판별할 수 없다. [세부 결과](results67.json)

각 cell에는 같은 prefix를 다시 쓰는 503,316,480개 bank–prefix 조합이 있다. Joint corrected의 분자는 sham 836,718개, 두 조명 1,543,154개다. 같은 catalog와 여러 catalog에 반복 등장한 board identity를 각각 복원추출하는 crossed bootstrap 12,000회로 paired difference를 구했다. 두 primary contrast에는 Bonferroni percentile interval을 적용했다. 8개 board와 lobe 두 종류, 같은 연산 구조·deployment bank에 조건부인 interval이다. [집계 block](aggregate67.npz)

## 3. 어떤 오류가 최종 구매까지 이어졌나

같은 bank, 같은 noise, 같은 prefix에서 둘째 slot의 counterfactual 판독을 네 번 비교했다. 추가 오류 없는 경우엔 일치하다 joint cell에서 처음 갈라진 사례에서 delay 단독·marker 단독도 해당 갈림을 만드는지 검사했다. 아래 분모는 전체 sham bank–prefix 조합이다.

| Joint cell에서 최종 추천이 다른 이유 | 보정 전 (%) | 보정 후 (%) |
|---|---:|---:|
| 기본 sham에서 이미 갈렸고 joint에서도 갈림 | 0.015556 | 0.023846 |
| Delay만 있어도 갈림, marker 단독은 일치 | 0.185703 | 0.001329 |
| Marker만 있어도 갈림, delay 단독은 일치 | 0.036119 | 0.141067 |
| 각각 단독으로도 갈림 | 0.018708 | 0 |
| 둘이 함께 있을 때에만 갈림 | 0.063323 | 0 |
| 합계 | 0.319410 | 0.166242 |

기본 sham에서 갈렸지만 joint cell에서 사라진 전환도 보정 전 0.009553%, 보정 후 0.001264%였다. 따라서 두 전환율의 차이는 새로 들어온 사례와 사라진 사례의 순효과다. 보정 후 새로 갈라진 sham 중 marker 단독으로 충분한 비중은 약 99.1%였다. 보정 후 **전체 sham 전환**을 분모로 삼으면 이 비중은 84.86%다. [경로별 계산](path67.py) · [경로별 수치](paths67.json) · [정수 분자 검산](attribution67.json)

같은 bank에서 reader의 어떤 변화가 있었는지도 검사했다. Marker 단독 개입으로 새로 생긴 sham terminal 0.141067% 가운데 0.138777 pp는 accepted slot 집합이 같고 accepted mismatch polarity만 달라졌다. 나머지 0.002290 pp는 accepted coverage만 달라졌다. Delay 단독의 새 전환은 모두 accepted coverage 변화였고, 보정 전 0.225035%, 보정 후 0.001329%였다. 복수 slot이 달라진 bank도 포함하므로 개별 이미지 한 장의 단독 인과효과로 해석할 수 없다. [경로 검산](path67.py)

## 4. 문헌, 증거와 후속 검정

| 원문과 연구 질문 | 방법·표본·주요 결과 | 이번 연구와의 관계와 한계 |
|---|---|---|
| Yoda et al.은 움직이는 장면의 다중 조명 normal을 어떻게 복원했나? | 3개 동기 조명과 multi-tap CMOS prototype. 움직이는 평판의 normal RMSE가 미보정 0.490에서 pixel별 보정 시 0.432 rad로 낮아졌고, 떨어지는 공·손 변형도 촬영했다. | 시간 대응·채널별 보정을 따로 검사할 근거다. Endpoint는 surface normal이며 contact purchase나 marker identity label이 없다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5876938/) |
| Guo et al.은 같은 형상에서 소재에 따라 photometric stereo가 어떻게 달라지는지 어떻게 검정했나? | Roughness 9수준 × translucency 6수준으로 만든 실물 반구 54개를 100개 방향으로 촬영했다. 형상을 고정해 소재별 판단 가능한 영역을 관찰했다. | 반사율을 정량적으로 통제할 설계를 제공한다. 같은 충돌의 dynamic strain과 후속 구매는 표본에 없다. [원문](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_DiLiGenRT_A_Photometric_Stereo_Dataset_with_Quantified_Roughness_and_Translucency_CVPR_2024_paper.pdf) |

두 연구와 이번 실험은 “조명과 운동 → waveform과 pixel displacement → accepted sign 또는 abstention → bank code → router → 구매” 중 서로 다른 지점을 측정한다. 광학 복원 성공과 낮은 구매 전환율을 서로 상충하는 관찰로 다룰 근거는 없다. 여기서 보탠 증거는 frozen policy의 end-to-end 오류 경로와 same-light 기술적 배경값이다. Simulation에서는 지연 보정 후에도 두 조명과 sham의 차이가 남았다. 실물 장비의 event rate, surface의 반사율, marker identity, 구매 유용성을 직접 입증하는 증거는 부족하다.

| 다음 연구 질문 | 바꿀 변수와 검증 방법 |
|---|---|
| 1. 시편 roughness에 따라 corrected excess가 유지되는가? | 같은 geometry에서 roughness를 층화해 A/A와 A/B를 paired 비교한다. |
| 2. 실제 diode offset이 session 중 drift하는가? | 시작·종료 commissioning pulse 차이로 drift를 추정하고 다음 session에 적용한다. |
| 3. 실제의 낮은 ghost 빈도에서도 marker가 terminal을 지배하는가? | 실측 범위로 ghost 빈도·strength를 바꾸고 gross-path를 분해한다. |
| 4. 속도나 가속도로 timing error가 부호까지 바꾸는가? | 같은 shape·조명에서 속도를 층화하고 independent displacement를 동기 측정한다. |
| 5. 독립 camera에서 sham은 얼마나 큰가? | 두 head의 sensor·exposure·registration noise를 실측하고 head 순서를 교차한다. |
| 6. 편광판이 persistent ghost 경로를 줄이는가? | Optical transmission을 맞춘 crossed/parallel 상태에서 independent ID를 판독한다. |
| 7. Capture가 일부 유실되면 결과가 얼마나 달라지는가? | 잘못 대응된 시간과 유실 raw를 구분하고 전체 등록 분모에 대한 bounds를 제시한다. |
| 8. False acceptance 감소가 useful read를 잃게 하는가? | Held-out surface에서 read 수를 맞춘 abstention과 identity reader를 비교한다. |
| 9. R→I와 I→R 중 어느 쪽이 물리 예측에 유용한가? | 공통 prefix와 paired action outcome으로 held-out Brier와 reset 비용을 측정한다. |
| 10. 작은 field pilot에서 mechanism을 먼저 확인할 수 있는가? | 반사성 높은 소재를 별도 diagnostic으로 등록하고 대표 표면의 발생률과 분리한다. |

현재 논문에 넣을 수 있는 범위는 timing correction으로 사라지는 background와 marker 오류 때문에 남는 구매를 같은 예산에서 분리한 **simulation diagnostic**이다. 실제로는 새 시편·board·날짜를 교차한 pilot에서 marker identity를 독립 판독하고, 두 head의 read noise와 교체 후 drift를 계측해야 한다. 구매 성능을 주장하려면 마지막 행동 뒤의 contact outcome도 별도로 확보해야 한다. [실물 절차](../paired_light_pilot_v65/RIG_KO.md)

Datacenter3 Slurm job 2744468 compute step에서 GPU를 숨기고 CPU 3개를 사용했다. Plan과 실행 코드를 고정한 뒤 24개 catalog block을 생성했다. v64 raw sign과 code, v54 prefix 재계산, 정수 분자의 합, terminal 복원을 검산했다. 해당 project의 이 폴더에서 bash run.sh 3을 실행한 후 같은 Slurm 환경에서 ../continuous_v2/.venv/bin/python path67.py로 경로를 계산할 수 있다. 서버의 paired_light_sham_v67b/blocks/에 per-slot 결과를 보존했다. [실행 로그](server.log) · [동결 목록](freeze.sha256)
