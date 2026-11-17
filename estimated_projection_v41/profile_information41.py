"""Secondary label-free profiling: linear geometry precision after rank-one projection."""
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
COORDINATE = np.arange(-3., 4.)
DESIGN = np.column_stack((np.ones(7), COORDINATE, COORDINATE**2))
FULL_COV = np.linalg.inv(DESIGN.T @ DESIGN)


def variance_ratio(unit):
    direction = unit / np.linalg.norm(unit)
    projected = DESIGN - np.outer(direction, direction @ DESIGN)
    if np.linalg.matrix_rank(projected) != 3:
        raise ValueError("Projected quadratic coefficients are not jointly identified")
    covariance = np.linalg.inv(projected.T @ projected)
    return np.diag(covariance) / np.diag(FULL_COV)


def main():
    rows = []
    split_checks = []
    for family in ("flare", "roll"):
        for height in (19, 31):
            for panel in range(12):
                name = f"panel_{panel:02d}_{family}_h{height}.npz"
                with np.load(HERE / "results" / "directions" / name) as archive:
                    pairs = archive["pairs"]
                    delta_true = (pairs[:, 1] - pairs[:, 0]) / .0003
                    halves = np.split(delta_true, 2)
                    d_half = [np.linalg.svd(h - h.mean(0), full_matrices=False)[2][0]
                              for h in halves]
                    cross_variance = .5 * sum(np.var(halves[1 - j] @ d_half[j], ddof=1)
                                               for j in range(2))
                    split_angle = np.degrees(np.arccos(np.clip(abs(d_half[0] @ d_half[1]), 0, 1)))
                    split_checks.append(dict(shape=family, height_mm=height, panel=panel,
                                             split_angle_deg=split_angle,
                                             heldout_variance_ratio=cross_variance / 2))
                    delta = np.roll(pairs[:, 1], 1, axis=0) - pairs[:, 0]
                    _, _, right = np.linalg.svd(delta - delta.mean(0), full_matrices=False)
                    for method, vector in (
                        ("oracle", archive["oracle"]),
                        ("paired", archive["estimated"]),
                        ("unpaired", right[0]),
                    ):
                        ratios = variance_ratio(vector)
                        residual = np.sqrt(max(0, 1 - (vector @ archive["oracle"])**2))
                        rows.append(dict(shape=family, height_mm=height, panel=panel,
                                         method=method, offset_ratio=ratios[0],
                                         tangent_ratio=ratios[1], curvature_ratio=ratios[2],
                                         nuisance_residual_norm=residual))
    output = [
        "# v41 projection이 깨끗한 local tangent 추정에 주는 비용", "",
        "이 보조 계산에는 label이나 decoder output을 쓰지 않았다. 물체별 local profile을 세 계수 "
        "(offset, local tangent, curvature)의 quadratic으로 두고, 일곱 높이 좌표에서 "
        "독립이고 분산이 같은 Gaussian sensor noise를 가정했다. 높이의 단위는 2 mm다. "
        "기본 곡선의 결정적 깊이는 빼거나 이미 알고 있다고 가정한다. 원래 설계 "
        "행렬을 X, rank-one nuisance projection을 P라 할 때, local tangent의 분산 "
        "배율은 (XᵀPX)⁻¹ 및 (XᵀX)⁻¹의 tangent 대각 원소 비율이다. "
        "다른 두 계수도 함께 추정한다. 이 배율은 원래 geometry로부터 얻는 "
        "정보의 손실을 뜻한다. 변형이 존재할 때 생길 수 있는 bias 감소는 별도다.", "",
        "| Shape | Height | Direction | Panel 중앙 tangent 분산 배율 [최소, 최대] | 평균 잔류 변형 길이 |",
        "|---|---:|---|---:|---:|",
    ]
    for family in ("flare", "roll"):
        for height in (19, 31):
            for method in ("oracle", "paired", "unpaired"):
                group = [r for r in rows if (r["shape"], r["height_mm"], r["method"])
                         == (family, height, method)]
                ratio = [r["tangent_ratio"] for r in group]
                remaining = np.mean([r["nuisance_residual_norm"] for r in group])
                output.append(f"| {family} | {height} mm | {method} | {np.median(ratio):.3f}× "
                              f"[{min(ratio):.3f}, {max(ratio):.3f}] | {remaining:.1%} |")
    output.extend(("", "계산은 geometry-only 센서 모델의 정밀도를 설명한다. 실제 Brier "
                   "loss에는 factual evidence, contact law, query/probe 선택, model "
                   "mismatch와 보정도 관여한다. 표의 비율로 Brier loss 변화를 추론할 수는 없다.",
                   "", "## Oracle 없이 추정의 재현성을 확인하는 방법", "",
                   "같은 target group에서 최초 16쌍과 나머지 16쌍의 PCA 방향을 독립적으로 계산했다. "
                   "추정한 방향에 반대편 16쌍의 paired differences를 투영하여 계산한 sample variance를 "
                   "sensor noise만 있을 때의 분산 2σ²로 나누고, 두 fold를 평균했다. "
                   "1보다 크면 그 축에 noise 이상의 측정 변화가 있다는 뜻이다. "
                   "같은 변형 모드가 실제 동역학에 무관한지는 이 지표만으로 판정할 수 없다.",
                   "", "| Shape | Height | 평균 16-vs-16 split angle | Held-out variance / noise variance |",
                   "|---|---:|---:|---:|"))
    for family in ("flare", "roll"):
        for height in (19, 31):
            group = [r for r in split_checks if r["shape"] == family and r["height_mm"] == height]
            output.append(f"| {family} | {height} mm | {np.mean([r['split_angle_deg'] for r in group]):.2f}° | "
                          f"{np.mean([r['heldout_variance_ratio'] for r in group]):.2f}× |")
    output.extend(("", "[각 panel의 계수별 결과](profile_information41.json)"))
    results = dict(projection_information=rows, split_calibration=split_checks)
    (HERE / "profile_information41.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (HERE / "PROFILE_INFORMATION_KO.md").write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
