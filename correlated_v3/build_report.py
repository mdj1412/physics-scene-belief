from pathlib import Path
import json

P=Path(__file__).resolve().parent
A=P/'results/analysis'
load=lambda name:json.loads((A/(name+'.json')).read_text())
summary=load('summary');contrasts=load('contrasts');diagnostics=load('posterior_diagnostics')
def row(method,sigma=.001,signed=-1):return next(x for x in summary if x['method']==method and x['sigma']==sigma and x['signed']==signed)
def contrast(method,metric='mse',sigma=.001):return next(x for x in contrasts if x['method']==method and x['sigma']==sigma and x['signed']==-1 and x['metric']==metric)
def interval(method,metric='mse',sigma=.001):
    x=contrast(method,metric,sigma);lo,hi=x['ci95'];return f"{x['difference']:.6f} [{lo:.6f}, {hi:.6f}]"
def table(methods,sigma=.001):
    lines=['| Method | MSE ↓ | CRPS ↓ | Brier ↓ | Design regret ↓ | Action agreement ↑ | Coverage90 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        r=row(m,sigma);reg=f"{r['regret']:.7f}" if 'regret' in r else '—'
        lines.append(f"| {m} | {r['mse']:.6f} | {r['crps']:.6f} | {r['brier']:.6f} | {reg} | {r['agreement']*100:.2f}% | {r['coverage90']*100:.2f}% |")
    return '\n'.join(lines)

text=r'''# Factual grazing contact의 correlated posterior: 서버 실험과 연구 가치 재검토

작성일: 2026-11-04. 이전 `continuous_v2`를 확장한 `correlated_v3` 결과다. 설명은 한글로, technical terms는 English로 표기한다. Primary experiment와 numerical audit를 모두 `datacenter3`의 기존 Slurm allocation에서 실행했다.

**Factual grazing contact가 geometry와 restitution을 보상 관계로 묶는 curved posterior를 만들었다. 작은 uniform grid와 prior importance particles는 narrow posterior에서 prediction과 observation selection 모두 실패했다. 그러나 strong baseline인 tempered SMC와 conditional quadrature는 reference에 가까운 결과를 회복했다. 현재 결과는 posterior dependence의 중요성을 보여주지만, 새로운 inference 또는 acquisition algorithm이 필요하다는 증거는 아니다.**

또한 실제 추가 observation을 선택하는 adaptive policy의 이점은 예상보다 작았다. Validation으로 정한 query-type rule과의 MSE 차이에 대한 95% CI가 0을 포함한다. 이번 결과에서 가장 강한 메시지는 “새 acquisition rule이 더 좋다”가 아니라 “posterior의 support와 dependence를 잃으면, 더 정확한 factual observation을 받아도 downstream 결과가 악화될 수 있다”이다.

## 1. 이전 실험에서 무엇을 바꿨는가

이전 실험에서는 geometry와 restitution의 prior 및 observation likelihood가 factorized되어 있었다. 이번에는 **실제 collision 이후의 normalized outgoing velocity**를 factual measurement로 사용했다. 동일한 velocity를 설명하는 여러 geometry–restitution 조합이 생기므로 posterior가 factorize되지 않는다.

이번에는 이 단순한 frictionless collision의 exact event solution을 primary forward model로 사용했다. 기존 PyBullet lookup을 작은 noise까지 그대로 사용하면 interpolation, timestep, collision-margin 차이가 posterior approximation의 실패처럼 보일 수 있기 때문이다. 별도 PyBullet audit에서 이 차이를 확인하고 원인을 분리했다. 따라서 아래 main table은 **exact physical model과 일치하는 synthetic sensor setting의 결과**이며, PyBullet image observation이나 real robot 결과가 아니다.

| 항목 | 설정 |
|---|---|
| 물체 | radius 0.02 m sphere, 고정된 box, half-width 0.03 m |
| Dynamics | Friction·gravity·damping이 없는 single rigid impact |
| Unknowns | Lateral geometry offset `g`, restitution `e` |
| Prior | `|g| ~ Uniform[0.032, 0.0498] m`, `e ~ Uniform[0.1, 0.9]`, 처음에는 independent |
| One-branch | `g > 0`를 알고 있으며, branch 내부의 curved dependence를 추정 |
| Two-branch | `sign(g)`가 ±일 prior probability가 각각 0.5; factual velocity만으로 sign을 구분할 수 없음 |
| Factual sensor | `y_f = v_x_out / v_x_in + Gaussian noise` |
| Factual noise SD | `0.03, 0.01, 0.003, 0.001` |
| Additional view | `y_view = g + N(0, σ_g²)`, `σ_g`는 family마다 1–4 mm |
| Additional probe | 별도 reset의 head-on restitution measurement, `y_probe = e + N(0, σ_e²)`, `σ_e`는 0.04–0.15 |
| Query | 모두 `translate_wall_y(δ)`이며 `δ`만 달라짐 |
| Query 배치 | −80, −50, −25, −8, +4, +12, +25, +50 mm 근처에 ±2 mm jitter; hidden truth와 독립 |
| Target | Counterfactual normalized outgoing `v_x`, contact event probability |
| Acquisition budget | Factual observation 이후 추가 observation 최대 1회; `view`, `probe`, `stop` |
| Utility | Expected target MSE reduction − `0.0005 × observation count` |

Additional sensors는 calibrated scalar sensor로 이상화했다. RGB detection이나 learned perception을 구현한 결과로 해석하지 않는다. Factual measurement는 outgoing velocity의 한 component만 관측하므로 ambiguity가 남는다. Full 2D velocity와 contact normal까지 정밀하게 관측하면 식별 가능성이 크게 달라질 수 있다.

## 2. Correlated posterior가 생기는 이유

Grazing 구간에서 `u=(|g|−0.03)/0.02`, `A(g)=1−u²`라고 두면,

\[
f(g,e)=v_x^{out}/v_x^{in}=1-(1+e)A(g),
\qquad
v_y^{out}/v_x^{in}=-(1+e)\sqrt{A(g)}\,u\,\mathrm{sign}(g).
\]

Factual likelihood는 다음과 같다.

\[
p(g,e\mid y_f)\propto p(g)p(e)
\exp\left[-\frac{(y_f-f(g,e))^2}{2\sigma_f^2}\right].
\]

`|g|`가 커지면 collision이 더 grazing해지고 `A(g)`가 작아진다. 같은 `y_f`를 설명하려면 `e`가 커져야 한다. Small-noise limit에서 posterior는 대략

\[
e(g)=\frac{1-y_f}{A(g)}-1
\]

를 따르는 curved ridge에 집중된다. Prior에 임의 covariance를 넣은 것이 아니라 **factual collision likelihood가 dependence를 만든 것**이다.

Two-branch에서는 `f(g,e)=f(−g,e)`이므로 서로 대칭인 두 ridge가 남는다. 이 경우 `Corr(g,e)`가 0이어도 independent가 아니다. 각 branch의 `|g|`와 `e`는 강하게 결합되어 있고, `translate_wall_y(δ)` 이후 두 branch의 contact 결과는 달라질 수 있다.
'''
refdiag=next(x for x in diagnostics if x['method']=='reference' and x['signed']==0 and x['sigma']==.001)
text+=f"\nHeld-out family에서 `σ_f=0.001`일 때 reference의 평균 `Corr(|g|, e)`는 **{refdiag['corr_abs_g_e']:.4f}**였다. Two-branch의 평균 `Corr(g,e)`는 numerical zero이지만 동일한 absolute-geometry dependence가 유지된다. Histogram MI는 binning에 의존하는 보조 diagnostic으로만 사용했다.\n\n"
text+=f"![Correlated posterior의 support 비교]({P}/results/analysis/posterior_support.png)\n\n"
text+=r'''그림은 사전에 저장 대상으로 정한 첫 held-out family 32의 결과다. Red cross는 truth이며 inference에 제공되지 않는다. Upper row는 one-branch, lower row는 two-branch다. 각 row에서 x-axis 범위를 통일했다. 작은 grid와 prior particles는 ridge의 일부만 남기며, SMC와 conditional quadrature는 구조를 유지한다. `factorized_marginals`는 geometry와 restitution의 marginal을 대략 유지하면서 dependence를 제거한 ablation이다. Scatter의 개별 point 크기는 weight를 반영하며, 연속 density plot과 동일하지 않다.

## 3. 비교 방법과 공정성

| Method | Posterior 구성 | 실제 support 크기 | 해석 |
|---|---|---:|---|
| Reference | Posterior-adapted `256×16` conditional quadrature | 4,096 / 8,192 | One-/two-branch의 approximate Bayes reference |
| Grid17×17 | Fixed Cartesian trapezoid grid + factual likelihood | 289 / 578 | 매우 작은 uniform grid |
| Grid33×17 | 동일, 더 촘촘한 geometry grid | 561 / 1,122 | 이전 small-grid 수준과 가까운 조건 |
| Grid65×33 | 동일, 두 축을 더 촘촘히 배치 | 2,145 / 4,290 | Grid scaling baseline |
| Prior512 / Prior2048 | Prior에서 sampling 후 importance weighting | 512 / 2,048 | Proposal adaptation과 rejuvenation 없음 |
| Posterior512 | 4,097-point marginal CDF에서 `g` sampling, conditional truncated Normal에서 `e` sampling | 512 | Model structure를 이용하는 strong sampling baseline |
| SMC512 | Adaptive likelihood tempering, systematic resampling, covariance-scaled Metropolis rejuvenation | 512 | Generic continuous inference에 가까운 strong baseline |
| Conditional_quad512 | Factual-compatible geometry interval과 conditional restitution CDF에서 quadrature | 항상 512 | One-branch `32×16`, two-branch `2×16×16` |
| Factorized_marginals | Reference의 각 marginal을 64-point quantile로 근사한 뒤 product 구성 | 4,096 | Dependence 제거 diagnostic; reference를 사용하는 privileged computation |
| Query-type rule | Validation에서 고른 fixed `view`/`probe`/`stop` + reference posterior | Reference와 동일 | Noise level과 branch regime을 알고 선택하는 강한 fixed rule |

Conditional quadrature는 `e | g,y_f`가 truncated Normal임을 이용한다. 이때 `mean=(1−y_f)/A(g)−1`, `sd=σ_f/A(g)`다. Geometry integration interval은 likelihood의 ±8σ 범위에 맞춰 줄이고 Gauss–Legendre nodes를 배치한다. 일반 black-box simulator에 곧바로 적용할 수 있는 새로운 adaptive-grid algorithm이 아니다.

SMC의 next temperature는 conditional ESS가 약 `0.8N`이 되도록 고른다. 매 단계 resampling 후 10회 Metropolis move를 수행한다. Two-branch의 sign은 factual likelihood에 대해 정확히 대칭이므로 마지막에 exact conditional sign refresh를 사용한다. 따라서 일반적인 unknown multimodal landscape의 mode exploration을 해결했다는 결과는 아니다. 추가 view/probe 이후에는 모든 particle baseline을 동일하게 likelihood reweighting하며, 별도 rejuvenation은 하지 않았다.

모든 방법은 동일한 prior, factual measurement, query values, candidate sensors, acquisition cost를 사용한다. Truth는 inference grid와 독립적인 continuous draw다. Acquisition은 hidden `g,e`, target truth, 아직 선택하지 않은 sensor realization을 입력으로 받지 않는다. Evaluation에서 양쪽 sensor를 모두 계산하는 것은 matched-information 비교를 위한 것이며 policy에는 전달되지 않는다.

**동일 atom 수와 동일 compute budget은 다르다.** SMC는 support가 512여도 여러 temperature와 Metropolis proposal을 평가한다. Conditional sampling도 4,097개의 marginal evaluations를 사용한다. 따라서 아래 결과를 동일 simulator-call budget의 우위로 해석하지 않는다. 이번에는 accuracy, support size, 실제 소요 시간과 evaluation count를 함께 기록했다.

## 4. Data split과 numerical validation

- 96 independent family: 16 dev, 16 validation, 64 held-out test.
- 각 family의 continuous truth, sensor quality, query jitter와 noise는 seed로 재현 가능하다.
- 2 branch regimes × 4 factual noise levels × 8 numeric queries. Held-out은 4,096 query contexts다.
- Stochastic inference는 3개의 independent inference seed로 반복했다. 동일 family의 worlds/noise/queries는 method 간 공유한다.
- 총 18 method/seed configurations로 110,592개의 raw metric row를 저장했다. 이 수가 independent sample size는 아니다.
- CI는 query와 seed를 family 내부에서 평균한 뒤 64 family를 paired bootstrap 10,000회 resampling했다. Multiple comparison correction은 하지 않은 exploratory CI다.
- Reference refinement는 dev 128 conditions에서 `512×32` quadrature와 513-point observation integration으로 비교했다. 모든 action이 같았다.
- 별도 2,100,225-point uniform grid로 dev 16 conditions의 posterior moments를 교차 검증했다.

Primary reference는 exact posterior의 analytic closed form 전체가 아니라 **convergence를 검사한 numerical approximation**이다. Contact indicator는 discontinuous하므로 velocity-based decision accuracy만으로 모든 contact probability의 완전한 수렴을 보장하지 않는다.
'''
verification=load('verification');audit=json.loads((P/'results/audit/uniform_grid_check.json').read_text())
text+=f"\nReference refinement의 maximum acquisition-gain 차이는 `{verification['reference_gain_max_error']:.3g}`, maximum predictive-mean 차이는 `{verification['reference_prediction_max_error']:.3g}`였다. 독립 uniform-grid audit의 maximum counterfactual mean 차이는 `{max(r['vx_max_error'] for r in audit['checks']):.3g}`였다.\n\n"
text+=r'''## 5. Narrow posterior에서의 결과

다음 표는 가장 작은 `σ_f=0.001`에서 one-/two-branch를 동일 비중으로 평균한 held-out 결과다. MSE는 normalized velocity의 squared error다. Incoming speed 0.6 m/s로 해석하면 physical velocity MSE는 표의 MSE에 0.36을 곱한다.

Design regret는 reference posterior에서 선택한 action의 expected MSE reduction minus cost가 최선보다 얼마나 낮은지다. 실제 미래를 아는 clairvoyant oracle과 비교한 값이 아니다. `Coverage90`은 central predictive interval의 empirical coverage이며, no-contact의 point mass 때문에 정확히 90%가 되어야 하는 지표는 아니다.

'''
text+=table(['reference','grid17x17','grid33x17','grid65x33','prior512','prior2048','posterior512','smc512','conditional_quad512','factorized_marginals','query_type_rule'])+'\n\n'
text+=f'''| Reference 대비 paired difference | Mean difference [95% CI] |
|---|---:|
| Grid33×17의 MSE 증가 | {interval('grid33x17')} |
| Prior512의 MSE 증가 | {interval('prior512')} |
| SMC512의 MSE 차이 | {interval('smc512')} |
| Conditional_quad512의 MSE 차이 | {interval('conditional_quad512')} |
| Factorized marginals의 MSE 증가 | {interval('factorized_marginals')} |
| Query-type rule의 MSE 차이 | {interval('query_type_rule')} |

Small grid와 prior particles의 열화는 크고 CI도 0을 넘는다. 반면 SMC와 conditional quadrature의 차이에 대한 CI는 0을 포함한다. 이는 “완전히 동등함”을 입증한 equivalence test는 아니지만, 이 실험에서는 reference와 비교한 큰 열화가 관측되지 않았다는 뜻이다.

## 6. Noise가 작아질수록 어디서 실패하는가

| Factual noise SD | Reference MSE | Grid33×17 MSE | Prior512 MSE | SMC512 MSE | Conditional_quad512 MSE |
|---:|---:|---:|---:|---:|---:|
'''
for sigma in [.03,.01,.003,.001]:text+='| '+str(sigma)+' | '+' | '.join(f"{row(m,sigma)['mse']:.6f}" for m in ['reference','grid33x17','prior512','smc512','conditional_quad512'])+' |\n'
text+=f'\n![Noise sweep]({P}/results/analysis/noise_sweep.png)\n\n'
text+=r'''### Grid: ridge를 가로지르는 resolution이 부족하다

Factual noise가 넓으면 작은 grid도 posterior를 충분히 덮는다. Noise가 작아지면 대부분의 grid point가 ridge에서 벗어나고, 우연히 likelihood가 큰 1–2개 point가 posterior mass를 차지한다. 이때 geometry와 restitution의 서로 다른 plausible 조합을 유지하지 못한다.

예를 들어 `g=0.042 m, e=0.5`에서 `A=0.64`, `∂f/∂g=90 m⁻¹`이다. `33-point geometry grid`의 spacing은 약 0.556 mm이며, 이 방향의 factual velocity 변화는 약 0.050이다. Restitution spacing 0.05도 velocity를 약 0.032만큼 바꾼다. 둘 다 `σ_f=0.001`보다 크다. 다만 이것은 local scale 설명이며, diagonal ridge와 grid의 정렬까지 포함하는 엄밀한 error bound는 아니다.

### Prior particles: likelihood weighting 이후 support가 사실상 사라진다

Prior에서 뽑힌 fixed particles는 좁은 posterior tube를 거의 맞추지 못한다. Weight만 바꾸면 새로운 support를 만들 수 없다. Two-branch에서는 한 branch에 남은 우연한 particle이 대부분의 weight를 가져가고, 다른 branch가 없다고 잘못 믿게 된다. 그러면 sign을 알려주는 view의 value도 작게 평가할 수 있다.
'''
for signed in [0,1]:
    d=next(x for x in diagnostics if x['method']=='prior512' and x['signed']==signed and x['sigma']==.001)
    text+=f"\n- {'One' if signed==0 else 'Two'}-branch Prior512: median ESS `{d['ess_median']:.2f}` / 512"
    if signed:text+=f", branch mass가 5% 미만 또는 95% 초과로 쏠린 비율 `{d['mode_loss_rate']*100:.2f}%`"
    text+='.'
text+=r'''


여기서 ESS는 importance weights의 concentration을 나타낸다. Resampling된 SMC의 weight ESS=512가 512개의 독립적인 posterior sample을 뜻하지는 않는다. SMC의 실효 정확도는 reference 대비 prediction/action 및 seed variation으로 확인했다.

### Strong baseline: representation을 늘리기보다 배치를 바꾸면 회복된다

Conditional quadrature는 ridge의 conditional structure에 맞춰 nodes를 놓는다. Posterior sampling은 factual evidence를 반영한 proposal을 쓴다. SMC는 likelihood를 점진적으로 강화하면서 particles를 이동시킨다. 모두 narrow posterior에 mass를 두는 방식이다. 따라서 이번 결과가 지지하는 것은 **prior proposal을 고정한 importance sampling과 uniform grid의 한계**이며, particles 또는 small-support approximation 전체의 한계가 아니다.

## 7. Dependence와 observation selection을 분리한 해석

`factorized_marginals`는 reference의 marginal quantile들을 product로 조합한 numerical ablation이다. 완벽히 동일한 continuous marginals는 아니며 64-point approximation을 사용한다. 이를 true independent posterior라고 부르지 않는다.
'''
text+=f"\n`σ_f=0.001`에서 dependence를 제거하면 full-pipeline MSE가 `{row('reference')['mse']:.6f}`에서 `{row('factorized_marginals')['mse']:.6f}`로 증가했다. 모든 방법에 **같은 view**를 강제로 제공해도 factorized ablation의 MSE 증가는 `{interval('factorized_marginals','view_mse')}`였다. 따라서 이 조건에서의 문제는 observation 선택만으로 설명되지 않으며, geometry measurement가 restitution posterior를 함께 갱신해야 한다는 점이 중요하다.\n\n"
text+=r'''같은 query type 안에서 최적 action이 달라지는지도 다시 검사했다. 다음은 `σ_f=0.001`, 64 held-out family × 8 queries의 reference 선택이다.

| Regime | View | Probe | Stop | 한 family 안에 action이 2개 이상 |
|---|---:|---:|---:|---:|
'''
for r in load('within_type_actions'):
    if r['sigma']==.001:text+=f"| {'One-branch' if r['signed']==0 else 'Two-branch'} | {r['counts'][0]} | {r['counts'][1]} | {r['counts'][2]} | {r['families_multi']}/64 |\n"
text+=r'''
**Two-branch에서는 모든 query가 view를 선택했다.** 이 setting은 sign ambiguity를 측정하는 데는 유용하지만, 어려운 acquisition ranking benchmark는 아니다. One-branch에서는 action이 달라지지만 대부분 probe와 stop 사이의 차이다. Useful view와 useful probe가 자주 뒤바뀌는 과제라고 과장해서는 안 된다.

Validation-only query-type rule은 모든 noise level에서 one-branch에는 probe, two-branch에는 view를 선택했다. 이 rule은 regime과 noise level을 알고 있는 강한 fixed baseline이며 numerical query value와 factual measurement 값에 따라 action을 바꾸지 않는다.
'''
text+=f"\n전체 noise 평균 MSE는 adaptive reference `{row('reference',-1)['mse']:.6f}`, query-type rule `{row('query_type_rule',-1)['mse']:.6f}`였다. Paired difference는 `{interval('query_type_rule',sigma=-1)}`다. Observation count는 `{row('reference',-1)['cost']:.3f}` 대 `1.000`으로 adaptive policy가 더 적지만, **MSE 우위는 통계적으로 명확하지 않다.**\n\n"
text+=r'''## 8. PyBullet audit: timestep 외에 collision geometry도 달랐다

96 continuous parameter cases에 대해 4 timestep의 PyBullet rollouts를 실행했다. 기존 `physics.py`의 설정을 그대로 사용한 첫 audit에서는 timestep을 줄여도 analytic sharp-box solution과 차이가 남았다. 조사 결과 PyBullet box의 default collision margin이 0.001 m였다.

Rounded box는 core half-width를 `0.03−margin`, sphere와 합쳐지는 corner radius를 `0.02+margin`으로 해석할 수 있다. 이 rounded model의 exact solution과 비교하거나, PyBullet box margin을 0으로 설정하면 차이가 크게 줄었다. 따라서 처음의 차이를 전부 timestep error라고 해석하면 틀린다.

| `dt=1/30720` 비교 | Normalized velocity component RMSE | Maximum error |
|---|---:|---:|
'''
margin=json.loads((P/'results/audit/margin.json').read_text())
for label,key in [('Default margin PyBullet vs sharp exact','default_margin_vs_sharp'),('Default margin PyBullet vs rounded exact','default_margin_vs_rounded'),('Zero-margin PyBullet vs sharp exact','zero_margin_vs_sharp')]:
    r=margin[key];text+=f"| {label} | {r['rmse']:.6f} | {r['max_error']:.6f} |\n"
text+=r'''
총 PyBullet rollouts는 480회다. 첫 384회의 contact/no-contact 판정은 exact model과 모두 일치했다. 그러나 velocity의 numerical discrepancy는 완전히 0이 아니다. Primary table은 이런 오차를 observation noise에 숨기지 않고 exact model 안에서 posterior approximation만 비교했다.

이 audit는 실제 연구 방향에도 중요하다. Visual mesh와 collision proxy가 조금만 달라도 material parameter가 geometry/model error를 대신 설명할 수 있다. 그렇다고 이번 audit만으로 real-world model discrepancy correction을 검증한 것은 아니다. 다음에는 같은 factual fit을 보이는 서로 다른 collision proxies가 held-out physical intervention에서 얼마나 다르게 실패하는지 확인해야 한다.

## 9. 실행 evidence, 비용, 재현 방법

Remote directory:

`/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/correlated_v3`

기존 `continuous_v2/.venv`를 재사용했고 이전 실험 source/data/results는 수정하지 않았다. PyTorch 2.7.1+cu128, CUDA 12.8, float64 numerical inference를 사용했다. Neural network training이나 foundation-model inference는 수행하지 않았다. GPU는 posterior likelihood integration과 acquisition calculation에 사용했다.

| Slurm job | Node | Main shard elapsed time | Peak CUDA allocation |
|---|---|---:|---:|
'''
for path in sorted((P/'results/main').glob('runtime_*.json')):
    r=json.loads(path.read_text());text+=f"| {r['job']} | {r['host']} | {r['seconds']:.2f} s | {r['peak_memory']/1024**2:.1f} MiB |\n"
text+=r'''
4개 GPU 모두 사용했지만 이 2D problem은 작아 GPU memory나 compute를 포화시키지 않는다. Node별 elapsed time 차이도 있으므로 이를 대규모 scalability benchmark로 해석하지 않는다. 기존 allocation을 취소하거나 사용자의 다른 process를 종료하지 않았다.

Held-out 평균으로 SMC512는 약 57,459 factual likelihood evaluations를 사용했다. Prior512는 512개, Prior2048은 2,048개다. SMC의 평균 end-to-end method 계산 시간은 약 22.7 ms, Prior512는 약 8.7 ms였다. 이 측정은 8개 query와 모든 sensor choice의 evaluation을 포함한 이번 구현의 시간이다. Simulator가 비싸면 이 비율은 달라진다.

`runtime.json` 집계의 reference method timer는 앞서 계산한 reference를 재사용하는 시간만 측정하므로 다른 method와 speed comparison에 사용할 수 없다. Factorized ablation 역시 reference 계산을 선행하므로 standalone efficiency baseline이 아니다. 모든 method의 실제 atom 수와 likelihood evaluation 수는 `timings_*.csv`에 저장했다.

Main prediction 결과 생성 후 histogram MI 계산의 tiny-probability multiplication에서 underflow를 발견해 log-marginal 형태로 수정했다. Original diagnostics를 `original_diagnostics/`에 보존하고 동일 seed로 diagnostic만 재계산했다. 다른 diagnostic의 maximum change가 `1e-10` 미만인지 검사하며, prediction/action raw metrics는 변경하지 않았다. Main run 당시 `core.py`는 `provenance/core_at_main_run.py`에 별도로 보존했다.

새로 실행할 때는 current allocation이 idle인지 확인하고 `launch.sh`의 job IDs를 현재 값으로 바꿔야 한다. 기존 결과를 보존하려면 `run.py --out`에 새 directory를 지정한다. Analysis script의 `ROOT`도 해당 directory로 맞춘다. Code와 data를 login node에서 실행하지 않는다.

```bash
# 할당받은 compute node의 Slurm step 안에서 실행
../continuous_v2/.venv/bin/python run.py --shard 0 --out results/new_run
# shard 0,1,2,3을 각각 실행한 다음
../continuous_v2/.venv/bin/python analyze.py
../continuous_v2/.venv/bin/python audit_physics.py
../continuous_v2/.venv/bin/python audit_margin.py
../continuous_v2/.venv/bin/python audit_reference.py
```

## 10. 이번 결과로 주장할 수 있는 것과 없는 것

| 주장 | 판단 |
|---|---|
| Factual grazing contact가 geometry–restitution dependence를 만든다 | Exact model과 numerical posterior에서 확인 |
| 더 작은 factual noise가 fixed approximate inference의 결과를 악화시킬 수 있다 | Noise sweep에서 확인 |
| 작은 uniform grid와 prior importance particles가 ridge/branch를 잃는다 | ESS, branch mass, predictive score, design regret로 확인 |
| Posterior dependence가 추가 observation 이후의 prediction에 중요하다 | Matched-view factorization ablation이 지지 |
| 모든 particle method가 실패한다 | 틀림. SMC와 posterior sampling이 잘 작동 |
| 더 큰 learned joint scene model이 필요하다 | 현재 실험으로 입증하지 못함 |
| Query-conditioned acquisition이 fixed rule보다 큰 이득을 준다 | 현재 setting에서는 뒷받침되지 않음 |
| Real visual scene이나 실제 robot에서 효과가 검증되었다 | 아직 아님 |
| 최신 논문보다 우수하다 | 논문 implementation을 재현하지 않았으므로 주장하지 않음 |

이후의 선행연구 검토와 연구 방향 제안은 이 한계를 전제로 한다.

---

'''
text+=(P/'literature_review.ko.md').read_text()
text+='\n\n---\n\n재현 artifact: `results/main/metrics_*.csv`, `results/analysis/*.json`, `results/audit/*.json`, `logs/`, `provenance/`. Source와 artifact hash는 `artifact_manifest.json`에 기록한다.\n'
(P/'REPORT.ko.md').write_text(text)
print('Report bytes',len(text.encode()))
