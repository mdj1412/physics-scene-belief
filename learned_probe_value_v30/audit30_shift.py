"""Does the measured paid probe prefix move outside the Coulomb source range? CPU only."""
import json
from pathlib import Path

import numpy as np

from analyze30 import LAWS, ROOT, RESULTS, digest, quota, selected

SOURCE = ROOT.parent / "adaptive_stopping_v29" / "results"
FEATURES = ["probe_v2_mean", "probe_v1_mean", "paired_factual_v1_mean"]
STAGES = ["2to4", "4to8"]
EXTRA = ["probe_v2_sd", "probe_v1_sd", "paired_velocity_cov"]
SENSOR_MODEL_INDEX = {0: 10, 1: 12, 3: 11, 4: 13, 5: 14}



def summarize_prefix(folder, source):
    meta = json.loads((folder / "metrics.json").read_text())
    lock = folder / ("choice_lock.npz" if source else "choices.npz")
    assert digest(lock) == meta["choices_sha256"]
    assert digest(folder / "metrics.npz") == meta["npz_sha256" if source else "metrics_sha256"]
    with np.load(lock) as packed:
        x = packed["x_cal" if source else "cal_x"]
        order = packed["probe_order" if source else "order"]
        y = packed["probe_y"]
    rows = []
    for prefix in (2, 4):
        responses = y[:, :prefix]
        paired = x[np.arange(8)[:, None], order[:, :prefix], -2]
        v1, v2 = responses[:, :, 0], responses[:, :, 1]
        cov = ((v2-v2.mean(1, keepdims=True)) * (paired-paired.mean(1, keepdims=True))).mean(1)
        rows.append(np.stack([v2.mean(1), v1.mean(1), paired.mean(1),
                              v2.std(1), v1.std(1), cov], axis=1))
    with np.load(folder / "metrics.npz") as data:
        gaps = data["audit_gap" if source else "gap"]
        costs = data["cost"][:, :, -1 if source else 0, :]
        risk = selected(costs[:, :, None], quota(gaps)).mean(1)[:, 0, :]
    return np.stack(rows, axis=1), risk[:, :-1] - risk[:, 1:]


def read_blocks(base, n, source):
    rows = [summarize_prefix(base / f"block_{block:03d}", source) for block in range(n)]
    return tuple(np.stack(arr) for arr in zip(*rows))


def describe(values, gains, limits):
    summaries = []
    for stage, label in enumerate(STAGES):
        variables = {}
        for j, feature in enumerate(FEATURES + EXTRA):
            sample = values[:, :, stage, j]
            lo, hi = limits[:, stage, j]
            variables[feature] = dict(mean=float(sample.mean()), median=float(np.median(sample)),
                                       own_p05_p95=np.quantile(sample, [.05, .95]).tolist(),
                                       n_below=int((sample < lo).sum()), n_above=int((sample > hi).sum()),
                                       fraction_outside=float(((sample < lo) | (sample > hi)).mean()))
        gain = gains[:, :, stage]
        summaries.append(dict(stage=label, environments=int(gain.size), features=variables,
                              mean_gain=float(gain.mean()), mean_abs_gain=float(np.abs(gain).mean()),
                              gain_p05_p95=np.quantile(gain, [.05, .95]).tolist()))
    return summaries


def main():
    assert (RESULTS / "completion.json").exists()
    devx, devg = read_blocks(SOURCE, 24, True)
    limits = np.quantile(devx, [.05, .95], axis=(0, 1))
    source = describe(devx, devg, limits)
    target = {}
    models = json.loads((RESULTS / "model_lock.json").read_text())["models"]
    contributions = {}
    for law in LAWS:
        xx, gain = read_blocks(RESULTS / law, 16, False)
        target[law] = describe(xx, gain, limits)
        predictions = []
        for block in range(16):
            with np.load(RESULTS / law / f"block_{block:03d}" / "choices.npz") as data:
                predictions.append(data["scores"])
        observed = np.array(predictions).mean((0, 1))
        contributions[law] = []
        for stage, model in enumerate(models):
            source_mean = devx[:, :, stage].mean((0, 1))
            target_mean = xx[:, :, stage].mean((0, 1))
            pieces = {}
            for j, index in SENSOR_MODEL_INDEX.items():
                assert np.isclose(source_mean[j], model["mean"][index], rtol=0, atol=1e-10)
                delta = (target_mean[j]-source_mean[j]) * model["beta"][index] / model["sd"][index]
                pieces[(FEATURES+EXTRA)[j]] = delta
            total = float(observed[stage] - model["intercept"])
            contributions[law].append(dict(predicted_mean_change=total, sensory=sum(pieces.values()),
                                           remaining_posterior=total-sum(pieces.values()), per_input=pieces))
    data = dict(source=source, target=target, stages=STAGES, feature_order=FEATURES+EXTRA, ridge_mean_shift_decomposition=contributions,
                source_q05_q95=limits.transpose(1, 2, 0).tolist(),
                population="all environments including evaluation-only four-probe prefixes after policy lock",
                gain="paired pre-minus-post batch-quota Brier (13/128); identical source-law common decoder",
                designation="exploratory univariate support audit; descriptive, no policy uses these thresholds")
    (RESULTS / "shift_audit.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    rows = ["### 실제 probe response의 source 분포 이탈", "",
            "Ridge scorer의 입력에 포함되는 probe velocity 두 성분의 prefix 평균을 원래 단위로 확인했다. v2는 scorer가 probe_tan_mean이라 부르는 두 번째 좌표다. Local surface normal로 회전시킨 새로운 변수는 만들지 않았다. 비교 기준은 같은 횟수의 Coulomb training 환경에서 계산한 5·95 percentile이다. 아래 백분율은 해당 구간을 벗어난 환경 비율이다.", "",
            "| Law | Prefix | 환경 수 | v2 평균 [source 5%, 95%] | v2 이탈 | v1 이탈 | 같은 물체의 factual v1 이탈 | 실제 평균 Brier gain |",
            "|---|---|---:|---|---:|---:|---:|---:|"]
    for law, stats in [("Coulomb (source)", source), *target.items()]:
        for stage, summary in enumerate(stats):
            features = summary["features"]
            v2 = features[FEATURES[0]]["mean"]
            lo, hi = limits[:, stage, 0]
            percent = [f"{100 * features[key]['fraction_outside']:.1f}%" for key in FEATURES]
            rows.append(f"| {law} | {STAGES[stage].replace('to','→')} | {summary['environments']} | "
                        f"{v2:.4f} [{lo:.4f}, {hi:.4f}] | {' | '.join(percent)} | {summary['mean_gain']:+.5f} |")
    rows += ["", "Source의 각 192개 환경, 두 target law 각각의 128개 환경을 모두 사용했다. 4회 prefix는 중단한 곳에도 가상으로 공개한 사후 평가다. 실제 gain은 같은 128개 worlds에서 두 prefix의 common-decoder quota Brier를 빼 계산했다. 과거와 현재의 평균 차이에는 contact law와 새로 추출한 유한 표본의 차이가 함께 들어간다. 이탈은 training support 밖이라는 판정이 아니라 source 중앙 90%를 벗어났다는 뜻이다. 각 변수의 주변분포를 따로 보므로 이 표로 joint shift의 크기, 성능 하락의 원인, 배포용 alarm의 정확도를 추정할 수는 없다. [감사 수치](results/shift_audit.json)"]
    rows += ["", "Frozen linear scorer에서 source 평균 예측과 target 평균 예측의 차이는 계수 × 입력 평균 변화의 합으로 정확히 나뉜다. Probe velocity 평균·표준편차와 factual/probe covariance 등 다섯 sensor features의 합, 나머지 posterior features의 합을 원래 Brier gain 단위로 표시했다. 서로 상관된 feature가 함께 달라지므로 이 분해는 계수에 따른 산술적 기여다.",
             "", "| Law | Prefix | 평균 예측 변화 | Sensor features 기여 | Posterior features 기여 |",
             "|---|---|---:|---:|---:|"]
    for law, values in contributions.items():
        for stage, item in zip(STAGES, values):
            rows.append(f"| {law} | {stage.replace('to','→')} | {item['predicted_mean_change']:+.5f} | "
                        f"{item['sensory']:+.5f} | {item['remaining_posterior']:+.5f} |")
    (ROOT / "SHIFT_AUDIT_KO.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
