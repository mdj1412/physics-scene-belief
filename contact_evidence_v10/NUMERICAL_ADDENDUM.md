# Numerical audit와 supplementary control

Primary policies를 모두 freeze한 후 order64 common decoder의 order96 audit에서 최대 prediction 차이 1.42e-4 m/s를 발견했다. 이에 기존 결과를 보존하고 모든128 test worlds에서 common decoder를 order128로 다시 계산했다. Order192와 모든 actions/noise repeats를 비교한 최대 차이는9.48e-13 m/s였다. Acquisition choices는 변경하지 않았다.

Original evaluation은 results/test/evaluation, refined evaluation은 results/test/evaluation_refined에 있다. Original summary는 results/analysis/summary_order64.json이며 최종 summary.json은 refined decoder 기준이다. Preplanned primary contrasts는 그대로 유지했다. Decoder refinement는 결과 확인 후 numerical validation에 의해 추가한 조치라는 점을 숨기지 않는다.

추가한 boundary plug-in + visibility/query rule 결과는 supplementary end-to-end control이다. Geometry posterior mean을 한 surface로 대입하고 restitution posterior mean과 조합했다. 이를 MAP estimator 또는 calibrated predictive distribution이라고 부르지 않는다. Strong selector가 아니라 간단한 rule과 point estimate로도 결과가 유지되는지 검증한다.

Conditional independence: 이 model의 factual impact는 central flat region이므로 restitution과 local amplitudes가 분리된다. 따라서 이번 결과는 correlated geometry–restitution posterior의 필요성을 입증하지 않는다. v8과 다른 목적의 controlled observation-ambiguity diagnostic이다.
