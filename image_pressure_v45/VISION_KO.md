## 영상에서 최초 접촉 위치를 얼마나 정확히 읽었나

V45의 영상은 접촉 전 장면을 렌더링한 세 장의 grayscale frame이다. 검은 tip이 등속으로 접근하고, 접촉할 contour는 아래쪽 회색 경계로 보인다. 두 행 모두 같은 물체와 카메라를 썼으며 실제 접촉 결과는 어느 frame에도 없다. 위의 두 작은 표식은 15·35 mm의 고정 rail fiducial이다. 아래 그림의 좌표 문구는 원본 frame 바깥에 평가용으로 붙였다.

![같은 fold 물체에서 접촉 예상 위치만 바꾼 pre-contact video의 세 frame](camera_examples45.png)

**영상 생성.** V44와 같은 식으로 각 새 물체의 곡면을 만들고 최초 접촉의 world abscissa를 19·31 mm에서 −0.5, 0, +0.5 mm 이동했다. Tip의 접근 방향은 수평 +0.425, 수직 −1이며, 마지막 frame에서도 tip은 최종 접촉점 높이보다 0.52–1.55 mm 위에 있다. 96×32 pixel frame에 카메라 배율(3.2–4.0 px/mm), 평행 이동, tip 너비, 노출, 센서 잡음, subpixel pose jitter를 넣었다. 마지막 frame 일부에 독립적으로 occluder가 지나갈 수 있다. 면적이 달라지는 처리는 최초 접촉 이후의 solver에만 적용해 같은 물체의 point·narrow·broad 조건은 모든 pixel을 공유한다. 영상 생성기는 물체와 pose를 알고 추정기는 `uint8[N,3,32,96]` 한 개만 받는다. [구현](camera45.py)

**위치 추정.** 두 fiducial의 밝기 중심으로 매 clip의 pixel→mm 축을 보정한다. 어두운 tool silhouette에서 각 frame의 가로 중심과 apex를 읽고, 첫 frame의 회색 surface 경계를 추출한다. 세 frame의 운동을 경계까지 외삽한다. 마지막 중심이 가려지거나 앞선 운동과 일치하지 않으면 앞의 두 frame을 사용한다. 결과 좌표 하나만으로 과거 v43의 D를 이용한 lookup table에서 projection을 고른다. 이 표는 두 shape의 historical D를 평균한 뒤 가장 가까운 abscissa를 택한다. D가 0보다 작은 30.5·31.0 mm에서 projection이 선택되고 나머지 네 곳에서 original이 선택된다. Lookup table에는 shape, pressure area, 법칙, 물체 식별자, 실제 offset이 들어가지 않는다. V43의 D는 privileged axis에서 배웠고 v45 primary에서는 source에서 추정한 공통 axis를 지우므로 그 이득까지 보장하는 규칙은 아니다. [사전 동결 checksum](camera_freeze45.sha256) · [historical D](../contact_patch_v44/forecast44.json)

| 위치 입력 | MAE, mm | 6개 위치 bin 일치율 | Oracle 위치와 같은 decoder 선택 |
|---|---:|---:|---:|
| 3-frame camera | 0.09535 | 97.11% | 99.27% |
| 마지막 frame의 scene을 가림 | 0.18764 | 83.13% | 95.53% |
| 측정 위치에 −0.35 mm systematic bias 추가 | 0.32247 | 50.43% | 87.47% |
| 측정 위치에 +0.35 mm systematic bias 추가 | 0.38042 | 41.71% | 85.50% |
| Nominal height만 사용 | 0.33333 | 33.33% | 83.33% |

12개 catalog의 paired 설계에서 새 물체·variant·height·offset 조합의 clip은 221,184개다. 서로 다른 surface·law·sensor noise·pressure area에 같은 clip을 재사용했다. 3-frame 위치 MAE의 catalog 95% CI는 [0.09370, 0.09700] mm였고, 절대 오차의 median·p90·p99는 각각 0.07697·0.19599·0.36629 mm였다. 부호 있는 평균 오차는 +0.02995 mm였다. 19 mm의 세 위치에서 고르는 decoder가 같으므로 실제 위치를 틀려도 route는 같다. 결정 경계와 가까운 31 mm의 정확한 접촉점에서는 oracle route 일치율이 96.30%, +0.5 mm에서는 99.41%였다. 시야의 마지막 frame을 통째로 가린 진단과 출력 좌표에 systematic bias를 더한 진단은 서로 다른 오류를 시험했다. 이 수치만으로 Brier 개선이 입증되지는 않으며 같은 배포 입력을 쓰는 paired loss 비교가 별도로 필요하다. [범위별 통계와 confusion matrix](camera_audit45.json)

**재현과 범위.** 추정기와 capture 파일은 2026-11-16 19:15:55 KST에 서버에서 동결했다. Archive에는 clip별 예측치, evaluator용 좌표, 같은 환경의 512개 clip 묶음별 원 pixel SHA-256을 보관했다. 물체 128개 단위로 명시된 seed를 재사용해 원본을 복원한다. 일부 원본 clip도 별도 보관했다. 실제 camera의 3D viewpoint, motion blur, lens distortion, 실물 접촉과 시간 동기화에 관한 측정은 후속 검증 대상이다. 이 실험에서 GPU p90에 포함되는 영상 경로는 취득된 raster의 전송과 판독이며 실물 camera exposure·readout이나 로봇 실행 시간은 포함되지 않는다. [영상 재현 코드](capture45.py) · [검산 코드](camera_audit45.py)
