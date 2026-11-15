"""CPU-only aggregation of interleaved timings recorded by the v30 GPU runner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("learned", "fixed4", "uncertainty")
LAWS = ("split_static", "speed_weak")
TIMED_BLOCKS = 8
WORLDS = 128
N_BOOTSTRAP = 10000
SEED = 30913288


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_samples(results):
    batch, stream, setups, cohorts = [], [], [], []
    components = {"fit_ms": [], "score_ms": [], "engine_ms": []}
    model_checksum = sha(results / "model_lock.json")
    for law in LAWS:
        law_batch, law_stream, law_setups = [], [], []
        for block in range(TIMED_BLOCKS):
            folder = results / law / f"block_{block:03d}"
            meta = json.loads((folder / "metrics.json").read_text())
            artifact = folder / "metrics.npz"
            if meta["metrics_sha256"] != sha(artifact) or meta["model_sha256"] != model_checksum:
                raise ValueError(f"Invalid hash in {folder}")
            if meta["law"] != law or meta["block"] != block:
                raise ValueError(f"Invalid cohort key {folder}")
            with np.load(artifact, allow_pickle=False) as z:
                env = int(z["env_timed"])
                assert tuple(z["methods"]) == METHODS and 0 <= env < 8 and env == meta["timed_env"]
                assert z["batch_ms"].shape == (3, 3) and z["stream_ms"].shape == (12, 2, 3)
                assert len(set(z["timing_worlds"].tolist())) == 12
                assert np.all((z["timing_worlds"] >= 0) & (z["timing_worlds"] < WORLDS))
                assert z["setup_ms"].shape == (8,3)
                assert np.allclose(z["setup_ms"], sum(z[key] for key in components))
                for key in components:
                    components[key].append(z[key][env].copy())
                for key in ("batch_ms", "stream_ms", "setup_ms"):
                    assert np.all(np.isfinite(z[key]) & (z[key] > 0))
                law_batch.append(z["batch_ms"])
                law_stream.append(z["stream_ms"].reshape(-1, len(METHODS)))
                law_setups.append(z["setup_ms"][env])
                cohorts.append(dict(law=law, block=block, env=env,
                                    alpha=meta["alphas"][env], world_indices=z["timing_worlds"].tolist(),
                                    job=meta["job"], host=meta["host"], gpu=meta["gpu"]))
        batch.append(law_batch)
        stream.append(law_stream)
        setups.append(law_setups)
    return (np.asarray(batch), np.asarray(stream), np.asarray(setups),
            {key: np.mean(rows, axis=0).tolist() for key, rows in components.items()}, cohorts)


def compare(times, block_draws):
    # Every law has the same #independent blocks and #paired timings per block.
    # Preserve all three policies, all within-block worlds and repeated measures.
    values = times.reshape(-1, len(METHODS))
    p90 = np.quantile(values, 0.90, axis=0)
    ratio = p90[0] / p90[1:]
    law_ids = np.arange(times.shape[0])[None, :, None]
    replicates = []
    for start in range(0, len(block_draws), 100):
        sample = times[law_ids, block_draws[start:start + 100]]
        q = np.quantile(sample.reshape(len(sample), -1, len(METHODS)), 0.90, axis=1)
        replicates.append(q[:, :1] / q[:, 1:])
    return dict(n_per_method=len(values), p90_ms=p90.tolist(),
                learned_over_baseline=ratio.tolist(),
                ratio_pointwise_ci95=np.quantile(np.concatenate(replicates), [.025, .975], axis=0).T.tolist(),
                by_law_p90_ms={law: np.quantile(group.reshape(-1, len(METHODS)), .90, axis=0).tolist()
                               for law, group in zip(LAWS, times)})


def korean_table(summary):
    endpoints = summary["endpoints"]
    rows = [
        "## Common decoder의 GPU p90과 calibration 비용",
        "",
        "| 측정 구간 | 정책당 측정 횟수 | Learned (ms) | Fixed 4 (ms) | Uncertainty-only (ms) | Learned / Fixed 4 [95%] | Learned / Uncertainty [95%] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    titles = {"stream_warm": "Streaming, 1 world", "batch_warm": "Batch quota, 128 worlds 전체",
              "stream_amortized": "Streaming, setup을 128건에 균등 배분",
              "batch_with_setup": "Batch quota, setup 전액 포함"}
    for key, title in titles.items():
        item = endpoints[key]
        absolute = [f"{x:.3f}" for x in item["p90_ms"]]
        ratios = [f"{r:.3f} [{lo:.3f}, {hi:.3f}]" for r, (lo, hi) in
                  zip(item["learned_over_baseline"], item["ratio_pointwise_ci95"])]
        rows.append("| " + " | ".join([title, str(item["n_per_method"]), *absolute, *ratios]) + " |")
    rows.extend(["", "각 target law의 첫 8개 independent blocks에서 환경 한 곳씩 사전 추출했다. "
                 "각 환경에서 batch당 128개 world를 3회, streaming에서는 12개 서로 다른 world를 2회씩 "
                 "측정했다. 정책 실행 순서를 섞고 개별 실행 전후 CUDA synchronization을 수행했다. "
                 "측정한 것은 CPU·GPU 전송과 predictor 호출을 포함한 host-to-host wall time이다. "
                 "Runner에서 각 측정의 관측 선택과 첫 noise draw의 stick probability를 동결된 선택 및 "
                 "common decoder의 예측과 대조했다(확률의 절대오차 허용치 1e−8). "
                 "비율의 pointwise 95% interval은 law별로 block을 따로 10,000회 재표집했다. "
                 "같은 block에서 나온 세 정책의 측정과 모든 반복을 함께 유지했다.",
                 "", "환경별 setup은 factual·paid prefix posterior fit, CPU scoring과 정렬, selector bank 생성을 포함한다. "
                 "배포 전 한 번만 실행하는 detector 초기화 및 source model 학습과 별도로 산정했다. "
                 "동일한 common decoder 생성, 실물 sensing, 충돌·reset 대기 시간은 제외했다. "
                 "Streaming 상각값은 비용 배분값이고 한 건의 응답시간은 첫 행에서 읽는다.",
                 "", f"Source 데이터 로딩, nested cross-validation, 최종 ridge fit을 포함한 학습은 "
                 f"{summary['offline_seconds_source_fit_and_loading']:.2f}초였다. "
                 f"공유 detector 초기화는 {summary['shared_detector_static_ms']:.2f} ms였다. "
                 "Source target은 이전 v29 실행 기록을 재사용했으므로 과거 simulator 계산은 학습 시간에 포함되지 않았다."])
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    results = args.results
    completed = json.loads((results / "completion.json").read_text())
    trained = json.loads((results / "model_lock.json").read_text())
    assert completed["complete_blocks"] == 32 and completed["model_sha256"] == sha(results / "model_lock.json")
    batch, stream, setup, setup_components, cohorts = load_samples(results)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, TIMED_BLOCKS, size=(N_BOOTSTRAP, len(LAWS), TIMED_BLOCKS))
    endpoints = dict(stream_warm=stream, batch_warm=batch,
                     stream_amortized=stream + setup[:, :, None, :] / WORLDS,
                     batch_with_setup=batch + setup[:, :, None, :])
    summary = dict(
        methods=METHODS, baseline_order=METHODS[1:], laws=LAWS, gpu=completed["gpu"], job=completed["job"],
        independent_timed_cohorts=len(cohorts), worlds_per_batch=WORLDS, n_bootstrap=N_BOOTSTRAP,
        ci_method="Law-stratified paired block percentile bootstrap, whole timing cohorts with paired methods. Pointwise 95% intervals for p90 ratios; two baseline comparisons and warm/setup endpoints are descriptive.",
        unit="ms host-to-host wall-clock with CUDA synchronization; initial setup or marginally amortized setup specified by endpoint",
        validity="Runner rejects any timed action mismatch or abs(probability difference) >=1e-8 against the source-law common decoder and its first stored noise draw. Warm-ups excluded. No matched-law decoding or sensor I/O timed.",
        setup_scope="Path-specific full factual/prefix posterior fits + score/selection + selector bank. Includes paid-prefix processing; excludes sensing, physical reset, shared common decoder construction and shared static detector. Shared work was cached on the runner and charged back on each policy path. Streaming amortizes over 128 future worlds; batch charges setup once.",
        offline_seconds_source_fit_and_loading=trained["training_seconds"], shared_detector_static_ms=completed["detector_static_ms"],
        historical_source_label_creation="Reused the independent v29 source experiment. Training timer includes source file loading, validation, and fitting, excludes that earlier GPU campaign.",
        timed_cohort_component_means_ms=setup_components,
        endpoints={key: compare(times, draws) for key,times in endpoints.items()}, cohorts=cohorts)
    output = results / "timing_stats.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temp.replace(output)
    (results / "timing_summary_ko.md").write_text(korean_table(summary))
    print(json.dumps(summary["endpoints"]))


if __name__ == "__main__":
    main()
