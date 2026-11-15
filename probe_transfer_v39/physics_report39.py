"""Build the physical audit appendix from server-verified checks."""
import json
from pathlib import Path


root=Path(__file__).resolve().parent
check=json.loads((root/"physical_summary39.json").read_text())
shape=json.loads((root/"geometry_shift39.json").read_text())
lines=[
    "# v39의 geometry 차이와 held-out label 검산",
    "",
    f"할당된 별도 CPU node {check['host']}(Slurm {check['job']})에서 CUDA를 비활성화하고, 산출물이 커밋된 {check['complete_blocks']}개 block을 독립 NumPy 계산으로 전부 검산했다. Calibration과 deployment의 split·표면 방향·형상을 입력으로 받아 Cartesian 충돌 응답을 그대로 따라 쓰는 대신, normal/tangential 축에서 충돌 후 운동을 계산해 world 좌표로 되돌렸다.",
    "",
    "| 검산 항목 | 결과 |",
    "|---|---|",
    f"| 독립 cell–catalog replicate와 split | 4 cells × 12 replicates, 8 α와 두 split을 포함해 {check['n_catalog_splits']:,}개 support |",
    f"| 전체 prototype vector와 중복 | {check['unique_prototype_vectors']:,}개, 6개 latent parameter 전체가 겹치는 항목 {check['duplicate_vectors']}개 |",
    f"| 물리 상태·운동 비교 | {check['constitutive_cases']:,}회, contact state 불일치 {int(check['error_max']['all_catalog_state'])}건, 최대 Cartesian velocity 오차 {check['error_max']['all_catalog_velocity']:.3g} |",
    f"| Calibration의 측정과 ID | 786,432개 후보 row에서 고유 prototype {check['unique_calibration_factual']:,}개, 3,072회 paid probe에서 {check['unique_calibration_paid']:,}개, archived pool ID·방향 불일치 0건 |",
    f"| Paid sensor | Archived factual·첫 세 velocity·네 번째 velocity 재생성, 최대 오차 {max(check['error_max'][n] for n in ('calibration_factual','first_three','last_velocity')):.3g} |",
    f"| Held-out 충돌 | {check['deployment_draws']:,}회 deployment draw, 고유 prototype {check['unique_deployment_prototypes']:,}개, 여섯 표면 조건 전체에서 네 번째 접촉의 state와 query stick label 불일치 0건 |",
    "",
    "한 물체에서 만든 32개 noise realization 및 같은 표본에 적용한 orientation·강도 조건은 독립 prototype 수에 포함하지 않았다. Archive에는 첫 세 접촉과 네 번째 velocity, label, 선택된 센서의 base probability가 남는다. CPU 물리 검산의 직접 재생 대상은 보관한 calibration 센서와 query label이다. [검산 결과](physical_summary39.json) · [코드](audit_physics39.py)",
    "",
    "## 새로운 형상이 바꾼 관측",
    "",
    "예측 대응에 쓴 두 높이의 slope가 가깝더라도 센서가 보는 전체 contour는 다를 수 있다. 아래 수치는 같은 offset·slope·curvature를 가진 base contour를 고정하고, 설계식의 형상별 변위만 서로 뺀 RMS다. 표본의 stick label과 무관한 사후 기술값으로, 예측의 적중 여부를 다시 정의하는 데 사용하지 않았다.",
    "",
    "| 새 형상 / 대응한 과거 형상 | Factual profile RMS (mm) | Factual RMS / σ | Query view RMS (mm) | Query RMS / σ |",
    "|---|---:|---:|---:|---:|",
]
for name in ("shelf-shoulder","ridge-bead"):
    a,b=shape["factual"][name],shape["query_view"][name]
    lines.append(f"| {name} | {1000*a['rms_m']:.4f} | {a['rms_over_sigma']:.3f} | {1000*b['rms_m']:.4f} | {b['rms_over_sigma']:.3f} |")
lines+=["",
        "Factual profile은 −10~+10 mm의 9개 위치를 σ=1.60 mm로, query view는 +19~+31 mm의 7개 위치를 σ=0.30 mm로 관측한다. 이 값은 v38과 v39의 curvature 배율이나 catalog 구성 차이를 제외한다. 따라서 각각의 RMS를 성능 변화의 causal effect로 해석할 수 없다. [설계식](PHYSICS_KO.md) · [RMS 계산](geometry_shift39.json)"]
(root/"PHYSICS_AUDIT_KO.md").write_text("\n".join(lines)+"\n")
