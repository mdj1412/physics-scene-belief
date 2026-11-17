# v62 · LED pulse width와 shutter exposure가 판독률과 오판에 미친 영향

같은 8회 impact와 유료 관측을 사용했을 때, 선택적으로 보류한 reader는 기존 v61 dual이 받아들인 79,701개 변형 쌍 가운데 37,448개를 남겼다. **보류한 42,253개 모두 v61에서 맞게 판독한 쌍**이었다. Phase를 알려주지 않고 고정 window만 쓰는 nominal reader의 조건부 오판율은 38.05%였다. 오류는 동기화 전에는 충분히 있었지만, 이번 synthetic holdout에서는 v61 dual을 통과한 쌍 안에 confidence로 걸러낼 오판이 남아 있지 않았다. 판독 성공은 reference와 impact 양쪽에서 변형 부호를 읽은 경우다. [광학 원자료](vision62.json) · [사전 명세](VISION_PLAN_KO.md)

## 같은 표면과 board에서 비교한 세 가지 reader

각 표면에서 4개 session의 2개 물체에 2.0 N reference와 impact를 한 번씩 적용했다. 각 pilot의 16개 contact는 같은 sensor board를 사용했다. 8개 새 board 모두를 48개 새 surface에서 사용했고, 조건·표면별 512개 pilot에는 board당 64개 반복이 있다. 각 board는 switch와 diode bias·gain, camera skew, LED driver timing·contrast, ADC noise, dropout, local latch, 전자센서의 sign error가 다르다. Board별 parameter vector 8개가 서로 다르고 v61의 8개와도 중복되지 않음을 검사했다. Source router와 reader에는 board ID, 실제 지연, 물리 가설과 target outcome이 제공되지 않는다. [Generator](vision62.py) · [Board parameter](vision62.json)

1,000 fps camera의 8-frame buffer에서 물체 표식과 fixture의 상대 위치를 읽는다. LED pulse width(0.20 또는 0.80 ms)와 shutter exposure(0.25 또는 0.80 ms)를 2×2로 교차한다. LED ROI의 밝기는 발광 구간과 shutter가 겹친 길이를 exposure로 나눈 값에 비례한다. 물체 표식은 exposure 구간의 4개 midpoint에서 렌더링한 이동 PSF를 평균한다. 16 kHz 동시 ADC의 photodiode pulse도 실제 LED 폭을 따른다. 각 pilot의 여덟 impact에서 (LED, camera) 지연 네 조합(각각 0 또는 +2 ms)을 독립적으로 섞어 두 번씩 배정했고 이 배정과 noise를 네 pulse/exposure 조건에 공유했다. [구현](vision62.py)

**V61 dual**은 이전 버전의 함수 그대로 camera 내 단일 발광 frame을 찾고 switch 및 photodiode rising edge 차이로 접촉 시점을 복구한다. **Selective dual**은 같은 부호가 나온 뒤 LED 최고 contrast ≥48, 두 번째 ≤28, 변형 절댓값 ≥0.75 pixel, 1→4번째 및 2→5번째 frame의 두 변화가 최종 부호 방향으로 각각 ≥0.20 pixel인 경우만 남긴다. 이 기준은 결과를 열기 전에 고정했다. Trigger·contact 시각, 재질, exposure 설정을 알려주지 않으며 관측을 새로 구매하지 않는다. 기본 dual이 이미 보류한 관측은 그대로 결측이다. 전자센서 관측은 광학 부호를 보류해도 쓸 수 있다. 모든 유효 selective 부호가 기본 dual 부호와 일치하는지 검사했다. [동결 기록](vision_plan62.utc) · [구현 checksum](vision_code62.sha256)

## 어떤 판독이 사라졌나

| LED pulse / shutter (ms) | V61 dual 판독률(건수) | Selective 판독률(건수) | Selective의 기존 유효 판독 유지율 | 8회 impact 내 광학 쌍을 한 번도 못 읽은 pilot, dual → selective |
|---|---:|---:|---:|---:|
| 0.20 / 0.25 | 4.614% (18,214) | 2.532% (9,997) | 54.89% | 84.90% → 91.06% |
| 0.20 / 0.80 | 0.423% (1,669) | **0.0010% (4)** | 0.24% | 98.36% → 99.996% |
| 0.80 / 0.25 | 7.518% (29,679) | 4.268% (16,847) | 56.76% | 77.09% → 85.92% |
| 0.80 / 0.80 | 7.635% (30,139) | 2.685% (10,600) | 35.17% | 77.49% → 90.92% |
| 네 조건 균등 평균 | **5.047% (79,701)** | **2.372% (37,448)** | **46.99%** | **84.46% → 91.97%** |

분모는 각 행에서 394,760개 physically active reference/impact 쌍이다. 같은 slot에서 대응시킨 P/C와 A/B potential observation 네 개를 포함한다. 넓은 exposure와 짧은 pulse의 조합에서 selective가 받아들인 **4건은 한 개의 같은 물체 slot**에서 나왔다. 개별 board의 모든 조건을 합친 판독률 범위는 v61 dual 0.116~19.386%, selective 0.036~10.977%였다. 표의 네 조건을 독립 board 네 개처럼 세지 않았다. [쿼리 가능한 지표·분모](vision62.json) · [Catalog × board 원시 집계](vision62_cells.npz)

모든 유료 슬롯을 분모에 넣으면 조건을 합친 3,145,728개의 counterfactual 중 v61 dual의 판독률은 2.534%, selective는 1.190%였다. Inactive인데 양쪽 부호가 모두 읽힌 경우는 두 방법 모두 0, active인데 잘못 읽힌 경우도 모두 0이다. 따라서 **맞는 판독을 전달한 비율**은 active당 5.047%에서 2.372%로 내려갔다. 24개 catalog와 8개 board를 독립적으로 재추출한 5,000회 paired bootstrap에서 올바른 판독 손실은 **2.676 pp** [pointwise 95% CI 1.074, 4.662], 기본 판독 유지율은 **46.99%** [33.41, 54.10]였다. 오류 분모가 0이므로 제거한 전체 오류의 비율은 정의되지 않는다. 유효 판독에서 관측된 오류 0건이 실제 촬영의 오류율 0을 보장하지 않는다. [계산 코드](vision_score62.py) · [검산 결과](vision62.json)

짧은 pulse(0.20 ms)가 긴 shutter(0.80 ms)에 완전히 들어와도 exposure 중 빛이 켜지는 비율의 최댓값은 **25%**다. 수광 gain과 noise에 따라 최고 contrast가 48을 넘기 어렵다. 반대로 0.80 ms pulse가 여러 frame을 밝히면 단일 frame을 요구하는 v61부터 결측이 늘 수 있다. Raw fixture에서 768개 고유 reference/impact clip 중 단일 optical flash가 보인 건수는 네 조건에서 각각 168, 51, 599, 596개였다. 그중 두 ADC edge와 필요한 frame window까지 확보한 건수는 152, 42, 266, 265개였다. Frame 단위의 이 수치는 양쪽 변형을 모두 요구하는 표의 판독률과 분모가 다르다. [Raw fixture](fixtures/panel_00.npz) · [Pulse와 shutter의 계산](vision62.py)

## 해석과 재현

이번 결과에서는 nominal reader의 오판을 v61 계측이 걸러낸 뒤 selective가 추가로 버린 것은 맞는 관측이었다. 신뢰도 선별이 잘못된 부호를 골라냈는지에 대한 양성 근거는 없다. 그래도 downstream loss가 변할 수 있다. 몇 개의 정확한 관측을 보류하면, 장치 단서를 과신한 outer policy가 다른 행동을 덜 살 수 있기 때문이다. 그 기여는 동일 판독률의 무작위 보류 및 실제 held-out loss와 대조해야 한다. 적은 자료 때문에 추가 손실이 생겼을 가능성도 함께 남는다. [분모와 risk](vision62.json)

LED는 직사각형 pulse, shutter는 global shutter로 모사했다. 노출 시간에 따른 절대 광자 수보다 duration으로 나눈 평균 밝기를 사용했고, exposure 변화에 따른 shot noise·read noise 배율 변화나 실물의 specular reflection, rolling shutter는 primary에 넣지 않았다. 이 선택은 기존 reader를 같은 스케일로 평가하기 위한 것이다. 유사한 광학계를 실제로 구축할 경우 brightness gain, dark current, duty cycle과 phase 분포를 함께 측정해야 한다. [계측 명세](VISION_PLAN_KO.md)

서버에서 CPU simulation 중 CUDA를 숨기고 24개 catalog를 생성했다. Catalog별 지정 pilot의 16개 raw 8-frame clip과 두 128-sample ADC waveform을 저장하고, 세 reader가 full pilot과 같은 부호를 반환하는지 검사했다. 모든 episode의 부호, optical truth, lag 배정, 전자센서 부호, reset ID와 acquisition code도 저장했다. Policy에 전달한 것은 실제로 선택한 부호와 결측뿐이다. [Raw state와 결과 checksum](vision_bundle62.sha256) · [생성 코드](vision62.py) · [요약 코드](vision_score62.py)
