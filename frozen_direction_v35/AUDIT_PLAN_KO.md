# v35 독립 검증 계획

이 계획은 새 target의 loss 및 timing을 읽기 전에 고정한다. Primary 방법은 v34 저장 순서의 첫 fold인 rounded_fine에서 schedule별 k=4 reader와 estimated β=0.8289760825845313을 그대로 쓴다. 새 family 식별자, 방향 정답, deployment labels, 추가 probe를 inference에 제공하지 않는다. Paid record (x, y, signed action)로 q를 얻고 z=2q−1로 고정한다. 원본 physics probability h를 h exp(βz)/(1−h+h exp(βz))로 변환한다. 다른 세 fold는 sensitivity다. diag_models.json의 SHA-256은 b65e66c0952076a3aa7447a50cd6312584eb0f2c531392ae25c3048c531a2464이고 anchored34.json은 f2df999e57f47198df349d18ea4dcabd6ed3fbd6f47340f31a5c4c943c89b3eb이다.

Primary estimand는 각 probing schedule에서 correction − original의 paired held-out Brier 차이다. 가능한 한 archive의 paid records와 원래 decoder 확률로 CPU에서 별도로 q, correction, Brier를 다시 계산한다. 양방향 friction orientation과 두 schedule을 같은 draw 안에 보존하고, object × surface cell과 첫 probe 방향별 block을 복원 추출한다. 각 cell 및 orientation에 같은 가중치를 준다. Seed 350417, 12,000회 paired cluster bootstrap과 두 schedule의 max-standardized 95% simultaneous interval을 사용한다. Interval 상한이 0보다 작으면 개선, −0.003보다 작으면 사전에 정한 실용적 개선으로 판정한다. 경계에 걸치면 그 판정은 보류한다. Object별, surface별, cell별, orientation별 loss, 양·음 orientation의 q confidence 및 분류 정확도는 secondary다. Loss의 정확한 대수적 분해는 E[(h'−h)²] + 2 E[(h'−h)(h−y)]다. 배포 label과 correction 방향의 정합성을 평가한다. 동일 실험에서 여러 sensor draws, orientation mirror, deployment worlds를 독립 반복으로 세지 않는다. 고정 설계된 소수 family 밖의 모집단에 interval을 일반화하지 않는다.

Primary timing은 양 방법이 실제 배포하는 동일 service boundary에서 synchronized wall time의 p90이다. 입력, 전처리, output 소비, 재사용 가능한 cache와 warm-up 구간을 기록한다. Correction에는 reader, correction, 동기화가 포함된다. 장치 compute, 실제 물리적 probing 대기 시간, queue delay가 포함되는지 표시한다. 가능한 한 교차 순서를 균형화하고 paired timing cluster를 resample하여 동일 schedule의 corrected/original p90 ratio의 95% interval을 계산한다. 사전 기술 기준은 ratio 1.02다. Timing session 수와 order effect를 함께 제시한다. 하나의 GPU session이면 interval은 해당 장치·session 내 변동을 반영한다.

무결성 검증은 source hash와 stored beta, target fitting 여부, 서로 같은 fixed4 action 및 realization, 시작 방향 균형, 새 object shape 및 surface law의 코드와 support를 포함한다. 보조 true sign, 다른 fold 및 역방향 cue 실험은 primary 방법을 바꾸는 근거로 쓰지 않는다.

시행 전 확정한 design은 sinusoid/quartic × atan/algebraic의 4개 cell 및 isotropic control이다. 각 primary cell의 8개 block을 사용한다. Timing은 cell당 2개 이상 block을 얻으면 cell 내 재표집한다. 2개일 때 시작 방향으로 더 층화하면 변동을 추정할 수 없어 timing bootstrap에서는 해당 층화를 생략한다. Session 내 order별 차이와 block 수에 따른 불확실성을 명시한다.

design35.json에 명시된 두 schedule 평균에 대해서도 동일 paired bootstrap의 95% percentile interval을 계산한다. Schedule별 simultaneous interval은 두 schedule 각각에서 이득이 유지되는지 판정한다. 결과를 보기 전에 지정한 두 estimand를 모두 공개한다.
