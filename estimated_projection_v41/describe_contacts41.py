"""Descriptive contact frequencies from archived outcomes and the independent physics audit."""
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = ("flare", "roll")
HEIGHTS = (19, 31)
LAWS = ("saturating", "crossover")


def slope_offset(mm, family):
    weights = {"flare": (.8, -.85, .3, .95), "roll": (-.45, .55, .95, -.55)}[family]
    knots = (11,19,23,31,39)
    for a, b, coefficient in zip(knots, knots[1:], weights):
        if a < mm < b:
            x = (mm-a)/(b-a)
            return .9*coefficient*32*x*(1-x)*(1-2*x)/(b-a)
    return 0.


def main():
    audit = json.loads((HERE/"physics_audit41.json").read_text())
    error = audit["max_absolute_errors"]
    stats = {(f, h): [] for f in FAMILIES for h in HEIGHTS}
    for panel, block in itertools.product(range(12), range(2)):
        world = {}
        for f, h, law in itertools.product(FAMILIES, HEIGHTS, LAWS):
            folder = f"{f}_h{h}_base_{law}/catalog_{panel:02d}"
            with np.load(HERE/"results"/folder/f"block_{block:02d}.npz") as bank:
                world[f,h,law] = bank["labels"].astype(bool)
        for (f,h), store in stats.items():
            sat, cross = [world[f,h,law] for law in LAWS]
            alt_h = 31 if h == 19 else 19
            height_flip = np.mean([np.mean(world[f,h,law] != world[f,alt_h,law]) for law in LAWS])
            store.append((sat.mean(), cross.mean(), (sat != cross).mean(), height_flip))
    table = {f"{f}_{h}": dict(zip(("sat_stick", "cross_stick", "law_flip", "height_flip"),
                                   np.mean(values, axis=0).tolist()))
             for (f, h), values in stats.items()}
    (HERE/"contact_counts41.json").write_text(json.dumps(table, indent=2)+"\n")
    report = ["# v41 물리 검산과 contact height 민감도", "",
              f"서버 {audit['host']}의 CPU에서 {audit['complete_blocks']}개 archive의 SHA-256을 "
              "확인한 뒤, 별도의 NumPy 구현으로 접촉을 normal/tangential 좌표에서 풀어 Cartesian "
              "velocity와 stick state를 대조했다. 측정 noise, 물체 추출, probe 순서, 표면 방향도 "
              "재생성했다. 모든 계산에 적용한 절대 오차 허용치는 3×10⁻¹²다.", "",
              "| 검산 대상 | 결과 |", "|---|---:|",
              f"| Calibration·deployment support 고유 물체 | {audit['unique_contact_support_prototypes']:,} |",
              f"| 별도 passive pair 물체 / 전체 중복 | {audit['independent_passive_prototypes']:,} / {audit['duplicate_prototypes']:,} |",
              f"| Support의 독립 충돌 재계산 / stick state 불일치 | {audit['independent_collision_recalculations']:,} / {error['contact_states']:.0f} |",
              f"| Cartesian velocity / contour slope 최대 절대 오차 | {error['velocity_basis_transform']:.2e} / {error['slope_of_profile']:.2e} |",
              f"| 저장된 query label / 불일치 | {audit['archived_stick_labels']:,} / {error['archived_stick_label']:.0f} |",
              f"| Paid factual·velocity 재계산 최대 절대 오차 | {max(error[k] for k in ('paid_factual','first_three_velocity','fourth_velocity')):.2e} |",
              f"| Deployment draw(복원 추출) / 실제 고유 prototype | {audit['raw_deployment_draws']:,} / {audit['distinct_deployment_prototypes']:,} |",
              f"| Calibration pool / 실제 네 probe에서 쓰인 고유 prototype | {audit['distinct_calibration_pool_prototypes']:,} / {audit['distinct_paid_prototypes']:,} |",
              "", "모든 support에서 접근 속도, kinetic/static friction의 순서, sliding 도중 예측된 "
              "접선 속도의 부호와 충돌 직후 kinetic energy의 비증가를 검사했다. 물체와 반복 noise가 "
              "여러 조건에 대응되므로 저장된 label 수는 독립 표본 수와 다르다. "
              "[검산 수치](physics_audit41.json) · [독립 구현](physics_audit41.py)", "",
              "## 같은 물체에서 바뀐 실제 contact outcome", "",
              "Base contour의 보관된 query label을 사용했다. 두 orientation, 세 regime, 여덟 "
              "환경, 128개 물체를 같게 가중한 뒤 12 panel·2 block을 평균했다. "
              "나머지 두 contour variant는 설계상 같은 접촉 결과를 낸다. Height flip은 law별로 "
              "같은 latent object에서 두 높이의 결과가 다른 비율을 구한 뒤 두 law를 평균했다. "
              "두 높이에서 같은 대조를 읽으므로 이 열은 행 쌍마다 동일하다.", "",
              "| Shape | Height | Saturating stick | Crossover stick | Law flip | Height flip |",
              "|---|---:|---:|---:|---:|---:|"]
    for f,h in itertools.product(FAMILIES, HEIGHTS):
        v = table[f"{f}_{h}"]
        report.append(f"| {f} | {h} mm | "+" | ".join(f"{v[k]:.2%}" for k in
                      ("sat_stick","cross_stick","law_flip","height_flip"))+" |")
    report.extend(("", "[Archive 집계](contact_counts41.json) · [집계 코드](describe_contacts41.py)", "",
                   "## 실제 접촉이 명목 높이에서 벗어나는 경우", "",
                   "기본 설정은 수직 위치가 정확한 point contact다. 디자인한 quartic을 미분하여 "
                   "각 명목 높이에서 실제 hit가 위 또는 아래로 δ만큼 벗어났을 때 "
                   "가능한 두 |bulge slope − base slope| 중 큰 값을 계산했다. 표는 "
                   "slope(dx/dy)의 차이를 나타낸다. Hollow의 차이도 절대값이 같다.", "",
                   "| Shape | 명목 높이 | δ=0 mm | δ=0.10 mm | δ=0.25 mm | δ=0.50 mm |",
                   "|---|---:|---:|---:|---:|---:|"))
    for f,h in itertools.product(FAMILIES,(19,23,31)):
        spread = (max(abs(slope_offset(h+s*d,f)) for s in (-1,1)) for d in (0.,.1,.25,.5))
        report.append(f"| {f} | {h} mm | "+" | ".join(f"{v:.3f}" for v in spread)+" |")
    report.extend(("", "이는 사후의 geometry-only 민감도 분석이다. Nominal height에서 "
                   "보존되는 contact state가 오차가 있는 contact center나 유한 contact patch에서도 "
                   "보존되는지는 별도의 실험이 필요하다. 이 오차에서 Brier loss는 측정하지 않았다."))
    (HERE/"PHYSICS_AUDIT_KO.md").write_text("\n".join(report)+"\n")


if __name__ == "__main__":
    main()
