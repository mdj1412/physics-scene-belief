## 서버 및 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/image_pressure_v45`다. 영상 추정, pressure solver, GPU 예측과 latency를 나누어 실행했고, statistical scoring은 할당된 compute node에서 시행했다. 기존 v44 파일은 그대로 참조했다.

| 단계 | 실행 자료 |
|---|---|
| 사전 등록과 정보 차단 | [Primary 계획](PLAN_KO.md) · [Source-matched 규칙 사전 정의](SOURCE_CONTROL_KO.md) · [평가 코드·source lookup checksum](evaluation_freeze45.sha256) |
| 영상과 source axis | [Frame 생성·위치 추정](camera45.py) · [Clip 재현과 pixel checksum](capture45.py) · [실제 raw clip 표본](camera45/panel_00/pixels_fold_hollow_h31_d0.npz) · [Axis 추정](source_axis45.py) |
| Held-out prediction과 simulation | [GPU runner](runner45.py) · [Pressure model](pressure45.py) · [Rigid dynamics](bullet45.py) · [전체 물체 확장](bullet_full45.py) · [Rigid archive 검산](BULLET_RESULTS_KO.md) · [Primary physics checksum](pressure_outputs45.sha256) |
| 점수표 및 latency | [Primary 결과](score45.json) · [Source-matched](score45_matched.json) · [Rigid paired 결과](external_full_score45.json) · [두 GPU timing run](timing_comparison45.json) · [그림](display45.py) |

평가 시 보관한 확률의 최댓값이 float64 누적 반올림으로 1.0000000000000004여서 엄격한 입력 검사에서 중단됐다. 입력 검사에만 1e−12 허용오차를 둔 사본으로 서버에서 재실행했다. 확률 clipping이나 표본 제외 없이 같은 제곱오차 식을 사용했고, 직접 계산한 제곱오차와 전개식도 검산했다. 사전 동결한 코드와 실행한 파일을 모두 보관했다. [동결본](score45.py) · [실행본](score45_checked.py)
