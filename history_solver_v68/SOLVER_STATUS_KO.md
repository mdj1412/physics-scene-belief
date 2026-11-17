# 수치 종료 조건의 별도 처리

GPU forecast를 봉인한 뒤 target simulation을 시작하자 첫 compliant batch의 512개 물체 중 2–5개에서 3.333 ms 이후에도 엄격한 release 검사를 통과하지 않아 중단됐다. 이 시점에 최종 Brier는 집계하지 않았다. 모든 기존 성공 행의 binary와 출력은 그대로 둔다. 기존 integrator의 release flag가 0인 행에만 동일 oscillator·friction·step size로 상한을 33.33 ms로 늘린다. Step마다 모든 contact force가 0이고 모든 local plane에 대해 속도가 분리 방향이면 종료한다. 최초 limit 전 완료된 속도·slip은 원 결과를 그대로 유지한다. 모든 history tape는 원래 guard를 통과한 경우에만 생성되고 이미 봉인됐으므로 fallback이 적용되는 history는 0개다. 원 시작 실패, 연장 후 결과, 연속 하중과 중간 무하중 step, 최종 미해결 수를 집계한다.

33.33 ms에서도 일부 trajectory가 모든 local plane에 대해 분리 중이라는 조건을 만족하지 않았다. 진단한 두 물체는 0.750 또는 0.764 ms 이후 더 이상 하중이 없었으나, 멀리 있는 초기 tangent plane 하나에 대해 여전히 접근 속도를 가져 이 조건이 성립하지 않았다. 이때도 Brier 수치를 보지 않았다. 이 사건에는 마지막 33.33 ms의 finite-time velocity와 별도 unresolved mask를 함께 보존한다. 원래 한도 내 성공한 trajectory 및 모든 history tape는 원 결과를 그대로 쓴다. Extended가 true인 행만 재적분했고 그중 unresolved가 true인 행은 confirmed separation으로 해석하지 않는다. 모든 binary label이 뒤집힐 수 있도록 둔 보수적 loss bound와 미해결 수를 따로 제시한다.
