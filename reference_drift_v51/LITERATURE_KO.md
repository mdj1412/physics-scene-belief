## 선행연구와 독립 검토

검토할 질문은 *교체 전에 같은 물체에서 약한 하중과 2.0 N 하중을 짝지어 관찰하면, 교체 뒤 다른 물체가 충돌할 때의 stick 확률을 얼마나 잘 예측하는가*다. 두 하중에서 기록한 변형 부호, 충돌 중 변형 부호, 최종 stick/slip label은 서로 다른 측정 대상이다. 아래 여섯 편의 원문에서 관련된 부분을 비교했다.

| 원논문 | 질문·방법 | 표본·데이터 | 결과·결론 | v51로 일반화할 때의 한계 |
|---|---|---|---|---|
| Tuononen, *Scientific Reports*, 2016 | 같은 rubber–glass interface가 어떻게 미끄러지기 시작하는가? High-speed imaging, digital image correlation, force 기록과 finite-element analysis를 병행했다. | Dry/wet glass의 여러 normal load. 600 N 영상을 제시했고 400 N에서 dwell time 2–600 s를 바꿨다. | Dry에서는 접촉 가장자리에서 순차적으로 분리됐고 wet에서는 훨씬 급격하게 미끄러졌다. Macroscopic force만으로 onset 시점을 고르기 어려웠다. | 실물에서 환경과 관측 시점이 중요하다는 근거다. 2.0 N과 충돌 사이의 polarity mismatch 비율이나 v51 센서 분해능을 알려주지는 않는다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) |
| Ma & Rodriguez, *RA-L*, 2018 | 반복 planar pushing의 여러 결과 군집은 왜 생기는가? 기존 데이터의 순서·초기 자세를 분석하고 anisotropic friction simulation과 비교했다. | 이전에 수집된 반복 push 2,000회의 sequence 등을 재분석 | 표면의 마찰 방향성과 자동 수집 과정이 다음 초기 자세의 분포를 바꿔 관측된 군집을 설명할 수 있었다. | 반복 횟수가 많아도 독립적인 표면 교체 횟수는 작을 수 있다. 이 자료에서 v51의 부호 반전을 직접 관측한 것은 아니다. [원문](https://arxiv.org/pdf/1802.10089) |
| Dong et al., *ICRA*, 2019 | 진행 중인 grasp에서 incipient slip을 검출할 수 있는가? GelSlim의 dense motion field와 rigid-motion fit의 차이를 측정했다. | 실물 10개, 240회 검출 평가와 bottle-cap control 실증 | 24 Hz에서 86.25% detection accuracy를 보고했다. | 주로 rigid object를 가정하고 이미 접촉 중인 물체를 관찰한다. Surface 교체 후 독립 물체의 미래 충돌에 대한 probabilistic forecast와 시간 순서 및 endpoint가 다르다. [원문](https://arxiv.org/pdf/1810.13381) |
| VanderWeele, *Biometrics*, 2013 | Surrogate에서 관찰한 방향을 최종 outcome으로 옮길 충분조건은 무엇인가? Causal models와 이론적 반례를 분석했다. | 수리적 모형·반례. 물리 실험 표본 없음 | Surrogate와 outcome의 양의 상관만으로 intervention effect의 방향 보존을 보장할 수 없음을 보였다. | 논문은 treatment effect에 관해 증명한다. V51은 미래 outcome 예측 과제로, 이론이 접촉 부호의 반전을 증명하지 않는다. 두 관계를 따로 검증해야 한다는 방법론적 선례다. [원문](https://projects.iq.harvard.edu/files/applied_stats/files/surrogates_biom.pdf) |
| Gneiting & Raftery, *JASA*, 2007 | 어떤 scoring rule이 정직한 probability forecast를 유도하는가? Strict propriety와 score divergence를 분석했다. | 확률분포에 대한 정리와 forecast 사례. 로봇 접촉 표본 없음 | True conditional probability를 보고하면 expected quadratic loss를 최소화한다. | 어떤 estimator가 이동한 환경의 conditional probability를 더 잘 근사하는지는 데이터에 달려 있다. [원문](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) |
| Cameron, Gelbach & Miller, *Review of Economics and Statistics*, 2008 | Cluster 수가 적을 때 추론의 크기 왜곡을 줄일 수 있는가? Cluster bootstrap-t 접근을 Monte Carlo와 응용에서 비교했다. | 생성한 회귀 데이터와 기존 경제 데이터 기반 실험 | 검토한 조건 일부에서 cluster 구조를 보존한 추론이 과도한 귀무가설 기각을 줄였다. | 접촉 Brier 차이에 대한 특정 CI의 정확도를 보장하지 않는다. Calibration prefix와 session 변화가 공유되는 실험에서 독립 단위를 명시해야 하는 이유를 제공한다. [저자 원고](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) |

### 문헌에서 남는 간격과 평가 기준

**근거 지도.** Surface condition과 접촉 이력은 local detachment와 초기 자세를 바꾸고, dense tactile observation은 진행 중인 contact의 slip을 검출할 수 있다. 이 두 연결에는 실물 근거가 있다. 그다음 연결인 **교체 전 reference → 교체 후 impact deformation → 미래 stick 확률**에 대해, 위 실물 연구들은 하나의 자료에서 세 단계를 함께 측정하지 않았다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) · [Dong et al.](https://arxiv.org/pdf/1810.13381)

**엇갈려 보이는 관찰.** Dong et al.이 실물 slip을 검출한 결과와 Tuononen이 macro force에서 단일 onset을 정하기 어렵다고 한 결과는 함께 성립한다. 전자는 연성 sensor의 국소 영상에서 일부 영역이 미끄러지는지를 판단했다. 후자는 rubber–glass의 분리 전파를 보며 국소와 전체 sliding의 시점이 다를 수 있음을 관찰했다. 표본·측정 위치·시간 해상도·판정 기준을 맞춘 head-to-head 비교가 없어 방법 간 일반적 우위를 정할 근거는 부족하다. [Dong et al.](https://arxiv.org/pdf/1810.13381) · [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/)

**왜 단순 sign vote가 강한가.** V51에서 양쪽 신호가 threshold를 넘은 쌍의 부호가 모두 믿을 만하고 같은 환경을 공유한다고 가정하면, 부호 일치·불일치는 retention을 알려주는 Bernoulli trial이다. Beta(1,1) prior에서 같은 부호가 a개이고 유효한 쌍이 n개이면 posterior mean은 (1+a)/(2+n)이다. Soft vote도 이 값으로 두 endpoint를 혼합하므로 likelihood만의 추가 정보는 주로 측정 크기별 신뢰도 차이다. Binary Brier에서 두 정책의 conditional expected loss 차이는 참 stick 확률과 각 예측의 제곱거리 차이이므로, κ 추정 RMSE나 sign accuracy만으로 winner를 정할 수 없다. 이는 이 설계에 대한 계산이며, probability forecast를 점수화하는 이론적 근거는 다음과 같다. [Gneiting & Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

**무엇을 분리해 주장할 수 있는가.** V51의 symmetric flip 모형에서는 calibration 환경의 centered retention (2κ−1)에 reference–impact 부호 오류가 (1−2e), session이 반전될 확률이 (1−2g)를 곱한다. 조건이 독립이면 같은 곱을 가진 (e,g)는 같은 물체의 평균적인 방향 전달률을 낼 수 있다. 둘이 모두 반전하면 부호가 복구될 수도 있다. 그러나 e는 물체별이고 g는 환경별이므로 **한 session에 여러 물체를 배포했을 때의 공동 실패 양상**은 다르다. 이는 실험 모형의 성질이다. 평가자가 e와 g를 독립 조작했다는 사실은 배포 정책이 두 원인을 관측만으로 분리할 수 있다는 증거가 아니다. 실제 표면의 intervention·측정 절차를 분리해야 하는 근거가 있는 연구는 다음과 같다. [VanderWeele](https://projects.iq.harvard.edu/files/applied_stats/files/surrogates_biom.pdf) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089)

**논문 주장 점검.** 다수 물체와 noise draw는 한 session에서 공유한 네 쌍의 calibration 정보량을 늘리지 않는다. Catalog별 paired 차이, 독립 replacement session의 분포, 작은 이득의 불확실성을 함께 보고해야 한다. 지정된 e·g에서의 실험은 입력 조건별 정책 선택의 근거이고, 실제 표면에서 그 e·g가 얼마나 흔한지는 별도의 실험 대상이다. 단순 vote 대비 이득과 unsigned 대비 이득은 분리해서 읽어야 한다. 특히 장비에서 unsigned가 calibration을 생략할 수 있을 때는 소요 시간과 물체 교체·마모를 포함해야 한다. [Cameron et al.](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) · [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/)

### 후속 연구 질문 10개

| 질문 | 근거와 필요성 | 검증 변수와 방법 |
|---|---|---|
| 1. Reference sign과 impact sign의 불일치는 어느 접촉 시각에 집중되는가? | Detachment가 시공간에 걸쳐 전파된다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) | Impact 전후 high-speed optical marker 궤적, impulse 시각, 부호 역전을 함께 기록한다. |
| 2. Surface만 교체했을 때의 drift와 reset 자세 변화의 효과는 각각 얼마인가? | 초기 자세·수집 순서가 반복 push의 분포를 바꿨다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 동일 재질의 sham replacement와 다른 재질로의 교체 순서를 무작위화하고 자세를 별도 측정한다. |
| 3. 표면 교체 뒤 한 쌍만 재측정해도 session-wide 반전을 검출할 수 있는가? | 교체 뒤 초기 상태가 달라질 수 있다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 총 네 쌍을 교체 전·후 4+0, 3+1, 2+2로 나눠 환경별 실패와 손실을 비교한다. |
| 4. 약한 신호의 부호 판독 오차와 진짜 deformation reversal을 분해할 수 있는가? | 국소 영상과 전체 force가 서로 다른 접촉 현상을 관측한다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) | 독립 camera 두 시점과 force channel을 동기화하고 near-zero 범위를 미리 정한다. |
| 5. 단일 scalar에 크기에 따른 신뢰도 정보가 실제로 남는가? | Dense motion field에는 단일 부호에 포함되지 않은 공간 정보가 있다. [Dong et al.](https://arxiv.org/pdf/1810.13381) | 같은 paid pairs에서 sign vote, amplitude likelihood, 고정 길이 영상 요약을 비교하고 resolution을 변화시킨다. |
| 6. 동일 평균 Brier라도 일부 session에서 여러 물체를 함께 실패하는가? | 공유된 오차는 독립 관측처럼 취급할 수 없다. [Cameron et al.](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) | 물체별·session별 failure correlation과 session loss의 상위 분위수를 별도 추정한다. |
| 7. 물체별 flip이 특정 형상·속도에 집중되는가? | 서로 다른 지점·조건에서 slip 전파가 달랐다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) | Contour, tip radius, impact velocity에 따라 e를 직접 계측하고 blind group에서 재검증한다. |
| 8. 어떤 경우에는 교체 사실만 알아도 unsigned로 돌아가는 편이 유리한가? | Surrogate의 방향 전달에는 조건이 필요하다. [VanderWeele](https://projects.iq.harvard.edu/files/applied_stats/files/surrogates_biom.pdf) | Replacement indicator만 공개한 policy를 고정하고, blind·recalibration·oracle 사이의 loss를 비교한다. |
| 9. 실제 sensor 비용을 넣어도 네 쌍의 calibration 가치가 남는가? | 장시간의 dwell과 초기 상태가 friction에 영향을 줬다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 접근·reference contact·reset·재시도·배포 개수별 비용을 기록하고 session당 순이득을 계산한다. |
| 10. 참 κ 추정이 조금 개선되어도 Brier가 바뀌지 않는 경우는 무엇인가? | Proper score는 최종 event probability를 평가한다. [Gneiting & Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) | 사전 고정한 두 endpoint 확률의 절대 차이 구간별로 paired Brier 차이와 estimator 오차를 함께 본다. |
