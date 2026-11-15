## 5. 새 물체의 factual frame과 접촉 응답은 각각 무엇을 바꾸는가?

새 물체의 factual frame을 X, 유료 접촉의 응답을 Y라고 두었다. 원래 prefix의 loss에서 X만 추가한 loss를 뺀 값이 factual gain이다. X만 본 loss에서 X와 Y를 모두 본 loss를 뺀 값이 conditional probe gain이다. 같은 물체·batch·decoder에서 계산했으므로 두 signed gain의 합은 primary gain과 정확히 같다. 아래 값은 미리 지정한 각 law의 첫 block, 총 24개 cohort에서 얻은 기술 통계다.

| 추가한 evidence | 2→4 평균 signed gain | 4→8 평균 signed gain | 2→4 평균 절대 gain | 4→8 평균 절대 gain |
|---|---:|---:|---:|---:|
| X와 Y 함께 | −0.00026 | −0.00153 | 0.00582 | 0.00592 |
| X만 | −0.00065 | −0.00124 | 0.00477 | 0.00522 |
| 같은 X에서 Y 추가 | +0.00039 | −0.00029 | 0.00561 | 0.00594 |

Y를 추가했을 때 cell별 변화의 크기는 평균 약 0.006 Brier였다. 방향을 포함해 평균하면 두 stage에서 부호가 달랐다. 같은 Y의 환경별 gain 순서가 두 축을 전부 새로 뽑아도 유지된 비율은 50.8%와 51.7%였다. Factual frame만 추가해도 선택이 바뀌므로 fresh-object 실험 전체의 변동을 모두 유료 접촉의 효과로 돌릴 수 없다. 절대 gain은 부호가 상쇄되기 전 크기여서 두 항의 절댓값을 더해 전체 절댓값을 복원할 수 없다. 이 control은 law당 한 block이므로 interval을 부여하거나 law 우열을 판정하지 않았다. [Control 결과](results/control31.json) · [재계산 코드](control31.py)
