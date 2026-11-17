# Rigid contact에서 실제 최초 접촉 위치를 맞춘 외부 대조

환경별 처음 여덟 물체를 사용한 target-free 보조 설계를 먼저 실행했다. 이 소표본에서는 영상 기반 선택의 점추정 방향이 전체 pressure 실험과 달랐다. Sampling과 query dynamics 변화의 효과를 분리하기 위해 같은 rigid sphere를 환경당 128개 물체 전체에 사후 확장했다. 앞 여덟 개의 label, terminal speed, 최초 하중 좌표와 valid/loaded flag 모두를 별도 worker가 다시 계산해 원래 archive와 값 단위로 완전히 일치함을 확인했다. 전수 확장의 첫 8개와 원래 파일은 동일 표본이다. 최종 비교의 독립 단위는 12개 catalog다.

| 물체 범위 | 충돌 수·유효 수 | 명령 대비 실제 최초 위치 MAE (mm) | p90 / 최대 오차 (mm) | Point control과 label 불일치 | Stick 비율 |
|---|---:|---:|---:|---:|---:|
| 처음 8개/환경 | 55,296 / 55,296 | 0.00505 | 0.01055 / 0.10029 | 2.195% | 48.39% |
| 나머지 120개/환경 | 829,440 / 829,440 | 0.00511 | 0.01091 / 0.10030 | 2.296% | 47.27% |
| 전체 128개/환경 | 884,736 / 884,736 | 0.00510 | 0.01089 / 0.10030 | 2.290% | 47.34% |

Oracle router는 실제 최초 하중 좌표를 세 위치 중 가장 가까운 곳으로 양자화한다. 전체 충돌에서 의도 위치를 사용한 oracle과 다른 bin을 선택한 경우는 없었다. 영상은 command 위치에 정렬한 synthetic pre-impact pin rendering을 사용했다. 같은 scene의 rigid sphere에 camera를 달아 simulator 영상으로 위치를 측정한 실험은 후속 검증 범위다. 모든 충돌은 마지막 step에서 양의 하중을 벗어났다. [독립 solver의 입력과 좌표](BULLET_PROTOCOL_KO.md) · [전체 물리 검산](bullet_full_audit45.json) · [전체 archive SHA-256](bullet_full_outputs45.sha256)
