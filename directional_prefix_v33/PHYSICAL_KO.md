## 실제 표면에서 확인할 수 있는 부분

Tramsen et al.의 [공개 S1 측정표, Tables 1–2](https://doi.org/10.6084/m9.figshare.c.3950773.v2)에 실린 20개 조합을 전부 읽어 pulling 방향 두 가지의 fitted friction coefficient를 대조했다. 저자가 보고한 Δμ = μ_against − μ_along은 9개 조합에서 양수, 11개에서 음수였고 범위는 −0.378~+0.273이었다. 다만 저자가 제시한 Δμ의 standard error는 약 0.05로, |Δμ| > 0.10인 조합은 10/20개였다. 20개 모두에서 부호를 통계적으로 확인했다는 뜻은 아니다. 예를 들어 같은 flat soft substrate에서 stiff sample은 +0.025, soft sample은 −0.248이었다. 첫 값은 측정 오차에 비해 작다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [계산 코드](physical33.py) · [20개 조합의 재계산 결과](physical33.json)

양쪽 coefficient 차이를 합으로 나눈 signed amplitude (μ_against−μ_along)/(μ_against+μ_along)의 실제 범위는 −0.170~+0.122였다. v33에서 설정한 0.17/0.22는 속도 함수가 포화할 때의 최대 amplitude다. 특히 0.22는 이 측정표의 최대 절댓값보다 크다. 이 표가 뒷받침하는 것은 물체·표면에 따라 강한 마찰 방향이 달라질 수 있다는 점이다. Quasistatic pulling 측정은 v33의 impact law를 fit한 자료도, 합성 surface 이름에 대응하는 재질도 아니다. 이 표에는 v33의 query view·probe·별도 충돌의 stick label 조합이 없으므로 실제 장비에서의 Brier gain은 추정할 수 없다. [실제 contact에서 부분적으로 공통인 또 다른 데이터의 측정 항목](https://web.mit.edu/mcube/push-dataset/)

서버 데이터 파일: `/scratch2/mdj1412/datasets/tramsen2018_friction_supplement.pdf`. 원저자의 MD5 `92fa510e1a75c46ab09372e8000ff8e3`와 대조했다. 서버 CPU와 로컬에서 같은 표를 독립 파싱하고, 모든 방향별 계수 차이를 저자 표의 반올림 값과 0.001 이내에서 대조했다.
