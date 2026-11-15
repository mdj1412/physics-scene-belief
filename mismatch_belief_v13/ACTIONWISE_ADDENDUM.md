# Action-conditioned calibration — post-primary independent replication

Primary 결과를 본 뒤 발견한 design limitation: whitened predictive NLL이라도 measurement dimension이 다르면 score scale이 다르다. Primary pooled post-threshold15.37은 7D query_view의 matched score에 지배되어 2D probe mismatch를 놓친다. Primary 결과와 실패 판정은 변경하지 않는다.

수정안은 기존 matched calpost512에서 **실제 primary policy가 선택한 action별 subgroup**에 대해97.5% upper quantile을 따로 계산한다. Pre-threshold, model family/prior, inference, action choice는 그대로다. Action2 subgroup이 없으면 기존 pooled threshold를 fallback으로 사용한다. 이 fallback의 새로운 조건부 calibration 보증은 없다. Sample별 conditional coverage를 주장하지 않는다.

새 independent base worlds256, seed2400101+137*index, 동일 six conditions×16 noise에서 검증한다. 이 후속 실험은 이미 본 law families의 새 worlds에서 하는 replication이며, 새 law-family discovery로 표현하지 않는다. Test truth를 threshold fitting에 사용하지 않는다.

동일 actions에서 original pooled gate와 action-conditioned gate를 비교한다. Primary follow-up contrasts는 새 saturating_full/weakening_full에서 actionwise gate vs original gate / base QMC16-probe / expanded QMC16-probe의6개, Bonferroni99.1667% paired world-bootstrap CI(30000). Matched harm ratio도 별도 보고한다. Expanded QMC64 및 second scramble도 유지한다. 모든 model fitting은 원래 code를 재사용하며 결과는 `results_actionwise/`에 격리한다.

Compute는 original/actionwise gates를 같은48 cases×3에서 GPU와 CPU로 비교하고, 각 flag/nonflag stratum도 condition별4events×3으로 측정한다. 이 follow-up은 post-primary adaptation이며 최초 primary 실험의 confirmatory success로 합치지 않는다.
