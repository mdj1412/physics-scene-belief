# 수치 종료 조건의 별도 처리

GPU forecast를 봉인한 뒤 target simulation을 시작하자 첫 compliant batch의 512개 물체 중 2–5개에서 3.333 ms 이후에도 엄격한 release 검사를 통과하지 않아 중단됐다. 이 시점에 최종 Brier는 집계하지 않았다. 모든 기존 성공 행의 binary와 출력은 그대로 둔다. 기존 integrator의 release flag가 0인 행에만 동일 oscillator·friction·step size로 상한을 33.33 ms로 늘린다. Step마다 모든 contact force가 0이고 모든 local plane에 대해 속도가 분리 방향이면 종료한다. 최초 limit 전 완료된 속도·slip은 원 결과를 그대로 유지한다. 모든 history tape는 원래 guard를 통과한 경우에만 생성되고 이미 봉인됐으므로 fallback이 적용되는 history는 0개다. 원 시작 실패, 연장 후 결과, 연속 하중과 중간 무하중 step, 최종 미해결 수를 집계한다.
