# v43 · 이름을 가린 새 물체·표면·마찰 법칙에서 geometry와 correction을 분리한 결과

**사전 지정한 두 질문의 답은 다르다.** Source-trained robust decoder의 v38 optimizer는 temperature를 이기지 못했다. 접촉 위치가 어긋나면 projection이 손해가 되는 현상은 12개 catalog 모두에서 재현됐다. 다만 접촉점을 정확히 맞췄을 때 projection이 평균적으로 유리하다는 조건은 충족하지 못했다. V42에서 보인 완전한 손익 반전은 v43에서 형상에 따라 달랐다.

| 주 질문 | Blind OOD 결과 | 판정 |
|---|---|---|
| Robust decoder에서 optimizer를 써야 하는가? | Shifted Brier가 optimizer 0.143837, temperature 0.143771. Optimizer−temperature +0.000066, 동시 95% CI [−0.002586, +0.002719]. | 우위 미확인. 이 조건에서 optimizer가 Brier를 0.003 이상 줄인다는 가설은 구간 밖이다. |
| 변형 방향을 지우는 projection은 언제 손해인가? | 보정 없이 privileged−original이 nominal +0.000012 [−0.000680, +0.000703], shifted +0.004550 [+0.003872, +0.005227]. Interaction +0.004538 [+0.004206, +0.004870]. | 정확히 맞춘 때보다 어긋난 조건에서 projection의 상대 손해가 컸다. Pooled nominal 이득은 미확인이다. Fold에서만 nominal 이득과 shifted 손해가 함께 나타났다. |
| 새 decoder의 이득은 어디서 생겼는가? | 같은 구조·학습량에서 nominal-only ERM의 shifted Brier는 0.155147, offset 결과까지 학습한 robust는 0.143347. 후자가 12/12 panel에서 개선됐다. | Source에서 실제 접촉점이 바뀌었을 때의 outcome을 학습한 효과가 컸다. 접촉점이 정확할 때는 원래 decoder의 0%가 robust 0%보다 유리했다. |

동일 GPU에서 512개 물체의 warm p90은 **robust+temperature 474.75 ms**, **robust+optimizer 479.60 ms**였다. Optimizer가 4.85 ms 느렸다. 비율은 1.0102배 [paired bootstrap 95% CI 1.0077, 1.0124]로 사전 비용 한도 +2% 안에 들었지만, 실용적 Brier 개선 기준은 충족하지 못했다. 열두 조합을 같은 장치에서 비교한 표를 아래에 실었다.

![새 family에서의 보정·projection 손익·GPU 지연시간](blind_ood_v43.png)

그림 A는 여덟 비영 placement offset의 균등 평균이다. B의 음영은 아홉 offset에 대한 동시 95% 구간이며 양수는 projection의 손해다. C는 512개 물체의 paid evidence를 처리하는 warm GPU p90이다.
