# v40 실행 기록과 원자료

서버 작업 경로는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tangent_transfer_v40`다. 2026-11-15 09:20:44 UTC(18:20:44 KST), `results` 경로가 존재하지 않음을 확인한 뒤 사전 계획, 예측표, 데이터 생성, 실행·분석에 쓰는 14개 파일의 SHA-256을 기록했다. Manifest 자체의 digest는 `1bad064943fc0beaf59dfe1bb8a8222e57a59818535a7609f50f6014c8f7d059`이다. V38 optimizer 파일의 digest는 `9ba308386d5e3275cf22160d34a2d6b030f8fbc5e746a76aa87bfa6742e857e8`이다. V38 정책과 source 자료의 기존 manifest도 새 worker에서 검증한다. [동결 기록](freeze40.sha256) · [서버 시각](frozen_utc.txt) · [GPU 환경](results/run.json)

Slurm allocation 2744468의 node55에서 compute PID가 없고 memory 0 MiB 및 utilization 0%인 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. 두 정책은 동일 hardware에서 순서를 섞어 측정했다. 서버 프로젝트 안에서 새로운 작업 사본에 재실행하는 명령은 다음과 같다. 기존 사본에서 중간에 끊긴 실행을 재개할 때는 manifest와 SHA-256이 모두 일치하는 block만 재사용한다.

```bash
sha256sum -c freeze40.sha256
srun --overlap --jobid="$GPU_ALLOCATION" -n1 bash run40.sh
```

`results/<family>_<contour>_<law>/catalog_<id>/block_<id>.npz`마다 2개 orientation × 3개 regime × 두 정책 × 8개 환경의 loss, calibration 관측·보정량, held-out label, 선택된 sensor의 base probability와 timing을 저장한다. 같은 경로의 JSON에는 물체 추출 seed·QMC bank seed·SHA-256이 있다. 한 쌍의 물체 비교에서 target label, sensor 선택과 acquisition의 입력을 함께 검산할 수 있다. [GPU 실행 코드](run40.py) · [Brier 독립 재계산](audit40.py) · [생성물 목록과 완료 기록](results/completion.json)

독립 CPU 검산과 논문 표의 재계산은 같은 서버 환경의 compute node에서 실행한다.

```bash
export CUDA_VISIBLE_DEVICES='' BELIEF_DEVICE=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" audit40.py
"$py" stats40.py
```
