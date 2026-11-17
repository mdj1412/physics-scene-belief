## 재현 자료

서버 작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_patch_v44`다. GPU 예측, 독립 solver, 사전 forecast를 나누어 실행했다. 서버에서 생성한 primary 결과와 두 해상도 대조, 종료시점 검산을 포함한 label archive 3,456개의 checksum을 로컬 사본과 대조했다. 총 705,024회는 같은 물체의 수치 재실행을 포함한 계산량이다. 추정의 독립 단위는 12개 catalog다. [충돌 archive manifest](solver_outputs44.sha256) · [GPU 종료 기록](gpu_release44.log)

| 재현 단계 | 보관 자료 |
|---|---|
| 사전 예측과 분석 기준 | [Python](forecast44.py) · [동결된 48개 예측 및 deformed-only 분석](forecast44.json) · [분석 동결 checksum](analysis_freeze44.sha256) |
| Source model과 GPU 경로 | [V43 frozen artifact](../blind_decoder_v43/frozen_model43.json) · [GPU runner](runner44.py) · [GPU 독립 검산](audit_gpu44.json) |
| 독립 충돌과 검산 | [PyBullet 구현](solver44.py) · [Neutral fixture](neutral_controls44.json) · [수치 정밀도와 종료시각](NUMERICAL_KO.md) |
| Primary 평가 및 추가 대조 | [동일 물체 Brier와 paired contrast](stats44.py) · [사전 예측 채점](forecast_eval44.py) · [정밀도별 forecast 재채점](numerical_forecast44.py) · [그림 재생성](display44.py) |
