# Geometry–contact ambiguity v11

한국어 결과 보고서는 `REPORT_KO.md`, test 이전 설계는 `PROTOCOL.md`와 `DESIGN_ADDENDUM.md`에 있다. `LMMSE_ADDENDUM.md`는 training 단계에서 추가한 moment baseline 기록이다.

## 재현

Sealed results를 덮어쓰지 말고 code를 새 directory에 복사한 뒤, 사용자가 허용한 allocation의 **idle GPU**에서 실행한다. 기존 jobs를 중단하지 않는다. Python dependencies는 PyTorch, NumPy, Matplotlib이며 이번 실행은 parent `continuous_v2/.venv`를 사용했다.

```bash
python guard.py pilot.py
python guard.py pipeline.py
python guard.py extra.py
python guard.py numerical.py
python analyze.py
python report.py
```

`pilot.py`는 최종 설정의 split 자료와 numerical pilot을 생성한다. `pipeline.py`는 training/validation acquisition → rule fitting → test/misspec acquisition → policy freeze → actual evaluation → independent ranking audit 순서로 실행한다. `extra.py`는 private truth를 읽지 않는 LMMSE policy를 별도로 freeze한다. `numerical.py`는 frozen choices를 바꾸지 않는 high-order decoding audit이다.

위 code는 `SLURM_JOB_ID`를 요구한다. CPU analysis도 login node가 아닌 allocated compute node에서 실행한다. GPU guard는 foreign compute process가 나타나면 자신의 worker만 중단한다.

`report.py`는 numerical·statistical 결과와 사람이 검토한 Korean interpretation files를 합친다. 다른 설정으로 재실행하면 기존 interpretation을 그대로 재사용하지 말고 결과에 맞게 다시 작성해야 한다.

## 결과 구분

- `pilot_initial_results/`: 채택하지 않은 최초 설정. Final test 결과와 섞지 않는다.
- `design_scan.json`: pilot-only amplitude scan.
- `results/{train,validation}`: rule development, test performance로 보고하지 않는다.
- `results/{test,misspec}/public`: acquisition 입력.
- `results/{test,misspec}/private`: evaluation 전용 target과 actual observations.
- `policies`, `freeze.json`: primary policy provenance.
- `extra_policies`, `extra_freeze.json`: LMMSE provenance.
- `evaluation_61`: main actual predictions/losses.
- `results/numerical`: 별도 high-order audit.
- `results/analysis`: world-level losses, bootstrap summaries, figures.

“Unknown”은 primary에서는 finite model menu 안의 uncertainty다. Natural RGB, 3D camera planning, long-horizon dynamics, real hardware validation을 수행한 code는 아니다.
