# v68 · 사후 target 해상도 재검증

이 실험에서는 standard history로 얻은 동일한 봉인 forecast를 사용했다. 사전에 지정했던 32/128 표본의 정밀도 검산에서 일부 부호가 바뀌어 결과를 본 뒤 확대했다. Compliant terminal solver를 12개 catalog 전체에서 9 sites → 17 sites로, time step을 약 4.63 → 2.31 μs로 바꿨다. 각 environment의 첫 8개 물체에서는 33 sites 및 약 1.16 μs 계산까지 수행했다. 모든 환경·law·shape·surface와 radius를 유지했다. 기존 32개 물체의 fine slip speed와 label, 미분리 상태를 전수 bit 단위로 다시 맞췄다. [추가 계획](REFINEMENT_KO.md) · [계산 코드](refined68.py) · [채점 코드](refined_score68.py)

Full fine label을 계산하는 동안 별도 fine-history forecast를 만들었다. 양쪽의 seal이 생성된 후에만 여기서 전체 결과를 채점했다. 첫 32개 물체의 이전 민감도 결과는 이미 알려져 있었고 추가 생성은 동시 진행했다. [fine-history 봉인](fine_history/FORECASTS_SEALED) · [전체 label 봉인](refined_labels.sha256) · [채점 release](REFINED_RELEASED)

| 계산 및 사용 물체 | Contact law | Wendland soft − full | Dispersive soft − full | 두 family의 restored − dropped |
|---|---|---:|---:|---:|
| 9 sites 128/128 | strengthening | -0.001036 [-0.001426, -0.000646] | -0.000128 [-0.000818, +0.000561] | +0.000559 [-0.000548, +0.001667] |
| 9 sites 128/128 | weakening | -0.001525 [-0.001903, -0.001148] | +0.000965 [+0.000150, +0.001780] | -0.006015 [-0.007432, -0.004599] |
| 17 sites 128/128 | strengthening | -0.000680 [-0.001037, -0.000323] | +0.000175 [-0.000542, +0.000892] | +0.000024 [-0.001091, +0.001139] |
| 17 sites 128/128 | weakening | +0.000376 [+0.000040, +0.000713] | +0.001124 [+0.000328, +0.001919] | -0.006621 [-0.007990, -0.005252] |
| 17 sites 첫 8/128 | strengthening | -0.000471 [-0.001511, +0.000569] | +0.000309 [-0.001035, +0.001654] | -0.000487 [-0.004007, +0.003033] |
| 17 sites 첫 8/128 | weakening | +0.000421 [-0.000256, +0.001099] | +0.001269 [-0.000352, +0.002889] | -0.006687 [-0.010082, -0.003291] |
| 33 sites 첫 8/128 | strengthening | +0.000135 [-0.000939, +0.001208] | +0.000227 [-0.001155, +0.001608] | -0.000625 [-0.004212, +0.002963] |
| 33 sites 첫 8/128 | weakening | -0.000094 [-0.000697, +0.000508] | +0.001236 [-0.000404, +0.002875] | -0.006954 [-0.010302, -0.003605] |

같은 해상도·같은 표본에서 두 law×세 contrast에 대한 동시 95% paired t interval을 사용했다. 네 source seed를 먼저 평균하고 12개 catalog를 독립 단위로 삼았다. 17 sites 첫 8개와 33 sites 첫 8개를 직접 비교해야 수치 해상도 변화와 구성 sample의 차이를 구별할 수 있다. 8개 물체를 대상으로 한 재계산만으로 모든 고해상도 수치가 수렴했다고 판단할 수 없다.

| 해상도 · 사용 물체 | Family | Law | 미분리/전체 | Soft−full의 임의 binary label 허용 범위 | Restored−dropped의 임의 binary label 허용 범위 |
|---|---|---|---:|---:|---:|
| 17 sites 128/128 | wendland | strengthening | 37/442,368 | [-0.000680, -0.000680] | [+0.004675, +0.004677] |
| 17 sites 128/128 | wendland | weakening | 49/442,368 | [+0.000376, +0.000376] | [+0.002600, +0.002603] |
| 17 sites 128/128 | dispersive | strengthening | 0/442,368 | [+0.000175, +0.000175] | [-0.004627, -0.004627] |
| 17 sites 128/128 | dispersive | weakening | 0/442,368 | [+0.001124, +0.001124] | [-0.015842, -0.015842] |
| 33 sites 첫 8/128 | wendland | strengthening | 2/27,648 | [+0.000135, +0.000135] | [+0.004846, +0.004850] |
| 33 sites 첫 8/128 | wendland | weakening | 5/27,648 | [-0.000094, -0.000094] | [+0.002853, +0.002860] |
| 33 sites 첫 8/128 | dispersive | strengthening | 0/27,648 | [+0.000227, +0.000227] | [-0.006096, -0.006096] |
| 33 sites 첫 8/128 | dispersive | weakening | 0/27,648 | [+0.001236, +0.001236] | [-0.016760, -0.016760] |

각 알려지지 않은 충돌에 대해 동일 target label을 모든 source seed와 noise replicate에 공유하고, 두 binary completion 중 해당 contrast의 평균을 작게/크게 만드는 경우를 선택했다. 이 범위는 미분리 label의 민감도이고 표본 신뢰구간은 위 표에 별도로 실었다. [전체 catalog와 구간](refined68.json) · [기본 물리 검산](SOLVER_KO.md)
