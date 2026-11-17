# V63 정책·위험의 사전 분석 계획

24개 새 catalog × 2개 surface × 8개 board의 crossed panel에서 각 surface당 256개 pilot(각 board 32개)을 사용한다. 물체 8개, session 4개, 물체마다 reference와 impact 1회가 각 정책에 공통이다. 주 target에서 physical persistence와 common-mode fault가 같은 비율이며, C의 sensor fault 빈도와 미래에 유리한 행동을 독립시킨다. 네 cell (transient, coherent)의 가중치는 같다. Calibration truth, 검증용 독립 센서와 deployment label은 policy에서 차단한다.

v61 전자 reset과 같은 raw stream을 사용한다. 모든 policy에서 v59 source likelihood와 v56 source cutoff를 동결한다. 기존 v61 판독 n개 중 confidence가 k개를 남긴 pilot마다 무작위 정책은 n개 중 정확히 k개를 균등하게 뽑는다. 8개 slot의 가능한 모든 subset을 열거한다. 보류한 slot의 reference/impact 광학 정보만 제거하며 전자센서와 결측 상태는 보존한다. Random arm의 평균은 실제로 한 subset을 골랐을 때의 risk를 주변화한다. Session loss p90도 행동마다 발생 가능한 session 손실 분포를 혼합한 뒤 quantile을 구한다. Gate는 똑같은 pilot budget을 지불하지만 고정된 v54 gate를 후속 deployment에 적용한다. 기존 v61, 광학 전부 보류, fixed 3R+I는 참고 대조다.

주 endpoint는 네 cell을 같은 비율로 합친 held-out mean Brier다. Confidence − same-count random 및 confidence − v54 gate에 대해 6,000회 paired two-way product bootstrap으로 catalog와 board를 독립 복원추출해 endpoint별 두 비교의 simultaneous 95% CI를 구한다. 음의 upper bound인 비교만 개선으로 판정한다. 두 대조 모두 이겨야 전반적 우위 주장을 인정한다. 표면·물체·counterfactual은 독립 hardware 단위로 재추출하지 않는다. 4개 cell별 두 비교에는 8개 contrast 동시 보정, factorial main effects와 interaction은 탐색적이다. 원인(P/C)과 실제 평균 loss가 낮은 정책의 balanced accuracy를 보고한다. 사후 올바름·오류 정보는 독립 optical audit에서 집계한다. Deployment label을 사용한 policy 최적화는 수행하지 않는다.

두 binary session replay의 p90, fine–ultra solver 및 각 binary replay의 mean Brier 차이를 확인한다. 물리적으로 활성인 쌍과 전체 지불한 쌍의 판독률을 구분한다. 0<k<n인 pilot 비율, 이들 안에서 confidence − random risk 차이, 두 정책이 서로 다른 행동을 고를 확률을 보고한다. 같은 판독 수 기준의 표적 선택 효과는 낮은 coverage와 작은 action gap 때문에 잘 추정되지 않을 수 있다.
