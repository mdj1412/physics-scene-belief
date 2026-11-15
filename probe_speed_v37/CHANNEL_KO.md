두 후보 표면과 같은 latent 물체를 주었을 때 weak/null 정확도 상한은 56.0%, 60.1%, 63.3%였다. 두 표면의 clean velocity가 같은 비율은 38.5%, 18.4%, 3.9%였다. 동일한 충돌 상태가 어느 속도에서 바뀌었는지 묶어서 보면 상한 개선이 어디에서 생겼는지 더 구체화할 수 있다.

| +0.50에서 올린 속도 | 처음부터 양쪽 모두 slip | 어느 한 표면이라도 stick → slip | 나머지 |
|---|---|---|---|
| +0.75 | 58.8% / +0.40 pp | 22.9% / +3.69 pp | 18.4% / +0.00 pp |
| +1.10 | 58.8% / +0.49 pp | 37.3% / +6.82 pp | 3.9% / +0.00 pp |

각 칸은 전체 물체 중 해당 비율과, 그 집단이 전체 privileged accuracy 변화에 기여한 percentage point다. 세 열의 기여 합은 해당 속도와 +0.50의 상한 차이와 같다. 속도를 올리면 slip 상태 외에도 tangential velocity, normal impulse, velocity-dependent friction modulation이 함께 바뀐다. 아래 부록에서 kinematics와 modulation을 따로 고정해 이 두 경로를 비교한다. [속도 요인 분해](PHYSICS_FACTOR_KO.md)
