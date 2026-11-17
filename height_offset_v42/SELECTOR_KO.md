# v42에서 보정 없이 계산할 선택지를 포함하는 방법

환경마다 source calibration 4회와 128개 deployment 물체에서 구매한 관측을 입력받는다. 물체별 stick 확률을 p, signed cue를 z, v38의 continuous 보정 세기를 w라 두면 허용 action은 [0, 0.325, w]이다. 각 환경에 속한 128개 물체에 동일한 action을 적용한다. 실제 target label, contact law·shape의 이름, 실제 또는 nominal height와 두 값의 차이는 selector input에 없다. Base decoder가 이미 사용하는 nominal height는 p에 간접 반영된다.

## Brier 차이의 추정

보정 세기 c에서 k=βzc이고 β=0.828976이다. q=logistic(logit(p)+k), d=q−p, h=d/k로 정의한다. k=0일 때는 h=p(1−p)로 연장한다. 한 환경의 평균 Brier 차이는 평균(d²)−2평균(d)·r이고, r은 h로 가중한 잔차 (Y−p)의 평균이다. 평균(d²)와 평균(d)는 이미 구매한 관측으로 정확히 계산할 수 있다. 새로운 환경의 Y를 보려면 배포 이후까지 기다려야 하므로, 과거 v41 source label로 r만 학습한다. Fixed 및 v38 optimizer 두 후보에 각각 ridge regression을 적용해 Y 없이 얻은 예상 차이를 0과 함께 비교한다. 예상 순위가 같으면 배열에서 먼저 나온 0%, fixed, optimizer 순으로 선택한다. 구현은 noise draw마다 별도로 계산하며 다른 반복의 관측을 모으지 않는다. [구현](selector42.py)

## Source와 모델 선택

V41의 독립 panel 12개, 24개 cell, 두 start block, original·estimated decoder, 여덟 environment, 두 orientation, 세 regime에서 noise draw 0·16만 사용했다. V41의 oracle projection과 v42 새 target 자료는 fitting 대상에서 제외했다. 역사적으로 nominal과 실제 contact height가 일치하는 경우만 학습했다. 따라서 실제 placement error에 대응하는 정책을 개발하면서 target label을 미리 보는 경로가 없다. 각 후보 action은 |평균(d)|에 비례하는 fitting weight를 사용해 보정 자체가 거의 일어나지 않는 환경의 영향을 낮춘다.

다음 항목을 source 안에서 비교했다.

- Feature set 1은 base probability의 평균, 불확실성, query/probe 가지 차이, 높은 확률의 물체 비율, signed cue와 optimizer 세기 및 이들 간의 상호작용이다. Feature set 2는 여기에 구매한 네 calibration 반응의 속도와 방향 통계를 더한다.
- Ridge penalty는 0.01·0.1, 적용하기 위한 예측 Brier 이득의 하한은 0·0.00025를 비교했다. 추가로 아무 보정도 사용하지 않는 후보를 점수 0으로 포함했다. 예측 때 source의 평균·표준편차로 표준화해 ±5로 제한했다.
- 평가 panel의 parity 전체를 학습에서 빼는 split, 여기에 평가 nominal height 또는 shape family까지 학습에서 빼는 split을 사용했다. 세 split에서 estimated decoder의 selector−zero Brier 평균이 가장 낮은 adaptive 후보를 선택해 모든 source panel에서 다시 학습했다. 0%를 후보에 포함한 보수적 결정도 별도로 기록했다. 같은 panel의 서로 다른 contour 또는 noise를 학습과 검증 사이에 나눠 넣지 않았다. Law family까지 holdout한 split도 별도 진단했다. 이는 v42가 기존 두 contact law를 재사용하므로 hyperparameter 선택 점수에는 들어가지 않았다.

모든 수치의 단위는 128개 물체의 batch risk다. Source에서 후보를 결정하는 데 사용한 risk는 새 catalog의 독립 검증값과 구별한다. 같은 128개 물체의 다수 sensor 조건, 보정 세기, noise draw를 독립 catalog 수로 세지 않는다. [Fitting](fit_selector42.py) · [동결한 계수](selector_model42.json) · [후보별 source 검증](selector_source42.json)

**Target 평가 전 결정.** Adaptive 후보 중 occupancy feature, ridge 0.01, 예상 이득 하한 0이 가장 좋았다. Source estimated decoder에서 selector−zero Brier는 leave-panel −0.000370, leave-height −0.000210, leave-shape +0.002378, leave-law +0.000461였다. 세 split의 macro score는 +0.000599이므로 상수 정책까지 허용한 보수적 방법은 항상 0%를 권한다. 사전 계획의 실제 질문은 세 action 사이에서 선택을 할 때의 효과이므로, 가장 좋은 adaptive 정책을 frozen challenger로 삼고 이 보수적 권고는 동일 GPU에서 평가하는 zero baseline으로 유지했다. 이 결정은 v42 결과를 보기 전에 내렸다.
