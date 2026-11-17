# v54 · 교체 후 첫 세 쌍으로 네 번째 관측 고르기

## 실제 정책 입력

세 물체 각각에 이미 지출한 weak 0.4 N·reference 2.0 N displacement를 입력한다. 함수는 정확히 세 쌍을 요구한다. 환경 ID, 실제 e·κ·drift event, 교체 전 네 쌍, 네 번째 물체의 weak reading과 impact 발생 여부는 받지 않는다. 호출할 때 출력 1은 impact, 0은 reference다. 선택이 끝나면 terminal decoder에만 기존 네 쌍과 실제 구매한 네 번째 쌍을 공개한다. 네 번째 물체는 어느 policy에서도 weak를 먼저 읽어보고 두 번째 측정을 고를 수 없다. [선택 함수](policy54.py) · [선택 bank 생성](routes54.py)

## Expected value of sample information

새 표면에서 active 변형이 weak→reference 방향을 유지할 확률을 k, reference→impact 반전 확률을 e라 하면 q=e+(1−2e)k는 배포 물체의 weak→impact 방향 유지율이다. Continuous joint는 최종 posterior mean q로 두 가지 signed terminal forecast를 혼합한다. 가능한 네 번째 관측은 t=k를 측정하는 reference 쌍과 t=q를 측정하는 impact 쌍이다. V53의 instrument specification에서 두 장치마다 t=0과 t=1일 때의 관측 분포 P₀와 P₁을 계산한다. 그 사이의 분포는 P(z|t)=P₀(z)+t(P₁(z)−P₀(z))다. Reference의 unreadable 구간은 하나의 무응답으로 처리한다. Impact가 inactive라면 방향을 알 수 없다.

어떤 hypothetical old history를 주었을 때 m=E[t]이고 D=P₁−P₀라 하자. 관측 뒤 posterior mean q의 이동량은 Cov(q,t)D(z)/(P₀(z)+mD(z))다. 이를 가능한 z에 대해 제곱 평균하면 Cov(q,t)² ΣD(z)²/(P₀(z)+mD(z))가 된다. Reference에서는 Cov(q,t)=(1−2E[e])Var(k), impact에서는 Cov(q,t)=Var(q)다. V52에서 고정된 source Brier risk의 quadratic coefficient가 방향에 관계없이 같아서, 이 양에 0.05023을 곱하면 모델이 예측하는 Brier 감소량이 된다. 이는 실측 Brier 개선의 보장이 아니다.

실제 교체 전 기록을 정책이 사용할 수 없으므로 가능한 old history를 적분한다. 각 old pair의 두 endpoint likelihood 비율을 5단계로 묶고 네 쌍에서 가능한 70개 count pattern을 모두 열거했다. 각 history의 확률은 k와 v52의 drift prior 0.25에 조건부로 계산한다. 현재 세 쌍은 측정 크기에 대한 연속 likelihood를 쓰고, hypothetical future sensor만 수치 bin으로 근사한다. 최종 결정은 두 예상 이득의 차이가 양수면 impact, 그 외에는 reference다.

## 정책별 예측과 선택

| Acquisition | 예측한 sensor 환경 | 예측 단계가 가정한 terminal | 61,440개 post prefix에서 impact 선택률 |
|---|---|---|---:|
| Primary EVSI | e=.5×Beta(1,4), 평균 .10 | 동일 nominal continuous joint | 0.00% |
| Sign-count prefix EVSI | nominal, 첫 세 쌍은 sign count | nominal continuous joint | 0.00% |
| Old 정보 사용을 생략한 근사 | nominal | 교체 후 관측만 있는 것으로 예상 | 0.00% |
| Broader-prior EVSI | e~Uniform[0,.5], 평균 .25 | broader-prior continuous joint | 7.54% |
| Source risk under broader error | e~Uniform[0,.5] | v53 nominal continuous joint 고정 | 35.29% |

마지막 행은 예상 환경과 terminal의 가정을 분리했다. Old 및 네 번째 sensor 관측의 분포를 broader prior로 재가중하고, 관측 후 행동은 nominal posterior에 따르게 한다. 각 선택에 대해 frozen source Brier quadratic을 적분한다. 이 정책은 plausible error가 더 클 때의 기대 손실을 최소화한다. Worst-case 최적화는 수행하지 않았다. Broader-prior EVSI의 route만 가져와 nominal terminal에 연결하는 ablation도 별도로 가능하다. [사전 설계](POLICY_PLAN_KO.md) · [선택률과 계산 오차](routes54.json)

Primary의 예상 impact 이득에서 예상 reference 이득을 뺀 값은 −0.00011278~−0.00001294였다. 모든 첫 세 쌍에서 고른 관측과 예측은 fixed 4R와 같아지고 acquisition 계산 시간만 추가된다. 마지막 행의 정책은 실제 k=0, .25, .50, .75, 1 조건에서 사후 집계상 각각 49.18%, 28.51%, 21.36%, 28.23%, 49.19% impact를 택했다. 정책이 k를 직접 보지는 않는다. 서로 독립적인 물체·재질이 61,440개라는 뜻도 아니다.

별도의 label-free prior sensitivity에서는 e/.5~Beta(1,.5/μ−1)로 prior 평균을 .10부터 .25까지 .01 간격으로 올렸다. Sensor와 terminal의 가정을 함께 움직일 때, μ=.20까지는 한 번도 impact를 사지 않았다. 처음 전환이 관찰된 μ=.21에서 7.06%, μ=.25에서 7.54%였다. 이는 해당 prefix 분포와 prior family에서 조사한 grid상의 전환점이다. 실제 environmental e가 .21이면 전환하라는 법칙으로 해석할 수 없다. [Prior sensitivity](priors54.json)

## 검산

독립적으로 future sensor의 양 끝 likelihood와 노이즈를 모든 bin에서 적분해 EVSI 및 broader-error source risk의 계산식과 대조했다. 96개 기본 분할을 160개로, old evidence의 5단계를 9단계로 늘려 3,840개 systematic prefix를 다시 계산했다. 5개 정책 각각에서 route 변경은 0건이었다. Nominal과 broader-error policy의 예상 이득 차이가 바뀐 절대량은 최대 1.06×10⁻⁸, 7.46×10⁻⁹이었다. 첫 세 쌍만 읽도록 구성된 코드에서 네 쌍을 직접 전달하면 예외를 일으킨다. Slurm node55에서 수행한 route 생성은 target outcome/risk를 로드하지 않는다. [Frozen source hashes](policy_freeze54.sha256) · [수치 결과](routes54.json)
