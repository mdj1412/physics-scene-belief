### 실물 접촉 연구에 비추어 본 외적 타당성

| 연구 질문 | 방법과 표본 | 결과와 한계 |
|---|---|---|
| Yu, Bauza, Fazeli & Rodriguez. 같은 조건의 planar pushing은 어느 정도 반복되는가? | 물체·표면·접촉 위치·방향·속도·가속도를 통제했다. §VII에서 직선 push를 2,000회 반복하는 설정으로 네 표면을 비교했다. | 표면별 translation SD는 3.4–11.7 mm였고, 일부 최종 pose 분포에는 여러 mode가 관측됐다. 실물에서는 같은 조작에서도 구조적인 변동이 생긴다. 이 연구의 출력은 pose여서 v31의 Brier variance 비율을 실물에서 입증하는 자료로 쓸 수 없다. [원문 §VII, Table IV](https://arxiv.org/pdf/1604.04038) |
| Tremblay, Meger, Hogan & Dudek. 접촉할 행동을 학습해 개별 물체의 물성을 더 정확히 추정할 수 있는가? | Bayesian filtering과 information-gathering MPC로 질량·높이·전도 높이의 simulation 세 과제를 평가했다. 실물 높이 과제는 7 cm 범위에서 학습 후 10 episodes로 평가했다. | 50,000 timesteps 학습 뒤 실물 높이 MAE가 1.19 cm였다. 물체 한 개에서 그 물체에 대해 얻는 정보는 실제로 유용할 수 있다. 다른 물체들의 deployment observation을 바꿨을 때의 Brier gain은 별도로 검증해야 한다. [원문 §§4–5.2](https://arxiv.org/pdf/2312.00215) |

V31의 다음 실물 설계에는 두 관찰이 함께 중요하다. 표면에서 연속 실행한 모든 push를 독립 trial로 간주하면 drift가 불확실성에 섞일 수 있다. 또한 동일 물체를 다시 측정해 물성 추정이 좋아지는 것과 접촉을 경험한 뒤 다른 물체의 관측을 더 잘 선택하는 것은 평가 대상이 다르다. 전자는 반복 물체별로, 후자는 calibration 물체 집단과 disjoint deployment 집단을 분리해 측정하는 설계를 제안한다. 이는 위 두 논문을 바탕으로 한 v31의 설계 제안이다. [Yu et al.](https://arxiv.org/pdf/1604.04038) · [Tremblay et al.](https://arxiv.org/pdf/2312.00215)
