# Compact multi-model belief v12 — test 이전 protocol

작성일 2026-11-12 KST.

## 목적

Full quadrature posterior의 moments를 받아오는 v11 mixture plug-in을 넘어서, 독립적으로 생성·갱신하는 compact belief의 decision loss와 compute cost를 측정한다. 새로운 objective나 Gaussian-sum filtering의 발명을 주장하지 않는다.

## Compact inference

4 discrete geometry/contact models 각각에3D latent Gaussian mean/covariance와 model probability를 유지한다. Standard-normal latent를 Gaussian CDF로 변환하여 v11과 동일한 bounded uniform parameter prior를 표현한다. State는4*(1+3+9)=52 float64 values=416 bytes다. Full covariance를 유지하되 full posterior arrays는 만들거나 입력받지 않는다.

Gauss–Hermite sigma points(order3 또는5)를 이용한 assumed-density Gaussian update를 구현한다. Factual profile와 factual velocity를 joint measurement로 처리하고, likelihood tempering steps1 또는4를 사용한다. Gaussian predictive likelihood로 model probabilities를 갱신한다. 이 finite-moment approximation은 multimodal within-model posterior를 정확히 표현하지 않는다.

Dev64 matched worlds에서 GH order3/5 × tempering1/4 네 configurations를 비교한다. Main compact configuration은 native decision loss가 가장 작은 configuration으로 고정한다. Test outcome으로 선택하지 않는다. Diagonal-covariance ablation과 hard-model/mixture-point decoders도 평가한다. Compact는 reference posterior, teacher labels, v11 test data를 이용해 fit하지 않는다.

## Worlds

v11 boundary/contact setup과 fixed visibility/query/sensor를 유지한다. 새로운 independent base worlds256개를 만든다. 각 base world에 matched, boundary-only shift, contact-only shift의3 conditions를 구성한다. 같은 s,a,e,c와 sensor noise를 공유한다. Actual additional observation noise는16 repeats다. Pilot16, dev64는 matched만 사용한다. Seeds는 pilot1200101, dev1300101, test1400101에 world index*137을 더한다.

Boundary-only: g를 third family f2=L*(z²/2-.15z⁴+.06z⁶)로 바꾸고 contact law는 그대로 둔다.

Contact-only: boundary를 그대로 두고 frictional class(c=1)의 tangential impulse를 Coulomb cap 대신 viscous law j_t=lambda*v_t, lambda~Uniform[.25,.65]로 바꾼다. Normal restitution은 그대로다. Frictionless class(c=0)는 변화가 없는 negative control이다. 따라서 전체 contact-only 결과와 실제 변경된 c=1 subset을 구분한다. 이는 계수만 바꾸는 것이 아니라 tangential response의 functional form을 바꾸는 실험이다. Synthetic single-contact law이며 real material identification을 주장하지 않는다.

## Strong baselines

- Full quadrature posterior(order61) + LMMSE acquisition, native posterior predictive mean.
- Prior scrambled Sobol QMC banks4096/16384/65536 + exact likelihood reweighting + 동일 LMMSE acquisition. New observation에도 factual weights를 갱신한다.
- v11에서 고정한 depth3 rule + reference/QMC16384 decoder. 이 rule의 offline teacher training은 baseline provenance로 명시하며 compact에 제공하지 않는다.
- Model-averaged point decoder와 single-model decoder; native/common decoder comparisons로 acquisition과 representation 효과를 분리한다.

모든 primary selectors는 same-cost actions none/query-view/factual-view/probe 중 하나를 고른다. LMMSE를 공통으로 사용하므로 primary 차이는 주로 belief approximation/update에 있다. 새로운 nested selector의 우위를 검증하는 실험은 아니다.

## Evaluation

Primary loss=||predicted query velocity−true velocity||²/2. Independent base world 단위로16 repeats를 평균한다. Conditions는 paired이므로 shift effects도 같은 base world끼리 비교한다. Main compact vs reference 및 QMC16384를3 conditions에서 비교하는6 primary contrasts에 Bonferroni99.1667% paired world-bootstrap CI를 사용한다(30,000 resamples). 나머지는 descriptive95% CI다.

Engineering screening 기준으로 mean loss ratio의 simultaneous CI upper bound가1.10 이내이고 serial end-to-end median latency가 baseline의 절반 이하이면 accuracy–cost trade-off가 유망하다고 본다. 10%와2x는 이번 diagnostic의 사전 기준이며 보편적인 application requirement가 아니다. CI가0을 포함한다고 equivalence로 해석하지 않는다. Sample256은 compute design이며 formal power guarantee는 없다.

Timing은 같은 GPU, float64, batch1, warmed execution, explicit CUDA synchronization으로 factual inference/selection/one selected observation update+prediction을 나눠 측정한다. 16 fixed test cases×3 repetitions, repeats를 독립 data points처럼 취급하지 않는다. Bank construction/setup, shared cache bytes, per-world belief bytes, forward-point counts와 batched experiment throughput을 별도로 기록한다. Sensor acquisition 비용은 측정하지 않는다.

## Integrity and numerical checks

Compact results를 생성·freeze한 뒤 reference module을 import한다. Private test truth는 policy freeze 후 evaluation에서만 읽는다. Full reference의 decoding은 first16 base worlds×3 conditions에서 order81로 확인한다. Numerical audit는 main choices를 바꾸지 않는다. Energy nonincrease, covariance PSD, model weights normalization, finite predictions, CPU/GPU-free leakage inspection을 확인한다. 기존 v11 artifacts는 수정하지 않는다.
