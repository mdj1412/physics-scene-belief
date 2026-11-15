from selector import Selector, METHODS, SIZES, TASKS, SEEDS, action, full_prefix
from common import R, require, save, write, sha, freeze
from generate import generate
from model import t
import numpy as np
import torch
import time
from pathlib import Path
import os

BASELINES = ("v18_global", "v18_region", "v18_adaptive", "nested2048")
CANDIDATES = [f"{method}/{size}" for method in METHODS for size in SIZES]
NAMES = CANDIDATES + list(BASELINES) + ["query", "probe"]
CELLS = [(0, 0), (0, 1), (1, 1)]


def timing(selector, split, names, count, check_choices=None):
    public = np.load(R / "data" / split / "public.npz")
    # Read observations outside of selection, only the chosen action reaches the decoder.
    with np.load(R / "data" / split / "private.npz") as private:
        observations = {a: t(private[f"obs{a}"][:count, 0, 0]) for a in (1, 2, 3)}
    inputs = [(t(x), t(c)) for x, c in zip(public["factual"][:count, 0], public["context"][:count, 0])]
    elapsed = {name: [] for name in names}
    schedule = np.random.default_rng(98831)
    for repeat in range(4):
        for i, (x, ctx) in enumerate(inputs[:1] if repeat == 0 else inputs):
            for name in schedule.permutation(names):
                torch.cuda.synchronize()
                start = time.perf_counter()
                a, w, z = selector.decide(x, ctx, TASKS[1], name)
                pred = selector.inner.update(w, a, observations[a][i]) @ z
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                if check_choices is not None:
                    assert check_choices[i, NAMES.index(name)] == a, (name, i, a)
                if repeat:
                    elapsed[name].append(ms)
    return {name: dict(p50=float(np.median(x)), p90=float(np.quantile(x, .9)),
                       over8=float(np.mean(np.array(x) > 8)), n=len(x)) for name, x in elapsed.items()}


def collect(selector, public, condition, task, reference=None):
    records = []
    references = []
    for i, (factual, context) in enumerate(zip(public["factual"][:, condition], public["context"][:, condition])):
        x, ctx = t(factual), t(context)
        state = selector.fit(x, ctx, TASKS[task], True)
        risks = []
        for method in METHODS:
            prefix = selector.mixture_risks(state, method, 4096, prefixes=True)
            risks.extend(prefix.values())
        risks.extend(selector.baseline(state, name) for name in BASELINES)
        records.append(torch.stack(risks).cpu().numpy())
        if reference and i < 64:
            w, z, ow, oz = reference.fit(x, ctx, TASKS[task], True)
            ref = full_prefix(reference.inner, w, z, reference.outer, ow, oz, reference.plans, [1024])
            references.append(ref[0, :, 0].cpu().numpy())
        if (i + 1) % 128 == 0:
            print("policy", condition, task, i + 1, flush=True)
    risks = np.stack(records)
    choices = np.where(np.ptp(risks, axis=-1) <= 1e-12, 1, risks.argmin(-1) + 1)
    choices = np.column_stack((choices, np.ones(len(choices), int), np.full(len(choices), 3)))
    return risks, choices, np.array(references)


def evaluate(selector, public, private, condition, task):
    target_col = (4, 7)[task]
    outputs = []
    observations = {a: private[f"obs{a}"][:, condition] for a in (1, 2, 3)}
    truth = private["target"][:, condition, target_col] / np.sqrt(2)
    for i, (factual, context) in enumerate(zip(public["factual"][:, condition], public["context"][:, condition])):
        w, z, _, _ = selector.fit(t(factual), t(context), TASKS[task], False)
        pred = torch.stack([(selector.inner.update(w, a, t(observations[a][i])) @ z[:, 0]) / np.sqrt(2)
                            for a in (1, 2, 3)]).cpu().numpy()
        assert np.isfinite(pred).all() and pred.min() >= -1e-10 and pred.max() <= 1 + 1e-10
        outputs.append(((pred - truth[i]) ** 2).mean(-1))
    return np.stack(outputs)


def main():
    require()
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    files = [Path("PROTOCOL.md"), Path("selector.py"), Path("run.py"), Path("analyze.py")]
    inherited = Path("../adaptive_budget_v18")
    files.extend(inherited / name for name in ("model.py", "common.py", "generate.py", "budget.py", "budget_fast.py", "acquisition.py", "guard.py"))
    write(R / "sources.json", {str(p): sha(p) for p in files})
    generate(12, 8450011, "pilot")
    generate(512, 8750011, "test")
    primary, alternate = [Selector(seed) for seed in SEEDS]
    reference = Selector(283909, 293911, 303907)
    pilot = np.load(R / "data/pilot/public.npz")
    state = primary.fit(t(pilot["factual"][0, 0]), t(pilot["context"][0, 0]), TASKS[1], True)
    err, qmin, qmax = primary.mixture_risks(state, "conditional", 512, return_identity=True)
    assert err < 1e-13 and qmin >= -1e-12 and qmax <= 1 + 1e-12
    pilot_time = timing(primary, "pilot", CANDIDATES, 12)
    selected = {}
    for method in METHODS:
        feasible = [n for n in SIZES if pilot_time[f"{method}/{n}"]["p90"] <= 8]
        selected[method] = max(feasible, default=None)
    write(R / "pilot.json", dict(timing=pilot_time, selected=selected, identity_maxerr=err))
    print("pilot budgets", selected, flush=True)
    public = np.load(R / "data/test/public.npz")
    paths = []
    for condition, task in CELLS:
        for seed, selector in enumerate([primary, alternate] if condition == 0 else [primary]):
            risks, choices, ref = collect(selector, public, condition, task,
                                          reference if condition == seed == 0 else None)
            path = R / f"policy_{condition}_{task}_{seed}.npz"
            save(path, risks=risks, choices=choices, reference=ref)
            paths.append(path)
    freeze(R / "freeze.json", paths + [R / "pilot.json", R / "sources.json", R / "data/test/public.npz"],
           metadata="test outcomes unopened until choices fixed", methods=NAMES)
    print("policies frozen", flush=True)
    checked = np.load(R / "policy_0_1_0.npz")["choices"]
    write(R / "timing.json", timing(primary, "test", NAMES, 32, checked))
    with np.load(R / "data/test/private.npz") as private:
        for c, k in CELLS:
            save(R / f"loss_{c}_{k}.npz", world_action_loss=evaluate(primary, public, private, c, k))
    write(R / "device.json", dict(name=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
                                  node=os.environ["SLURMD_NODENAME"], torch=str(torch.__version__)))
    print("GPU experiment complete", flush=True)


if __name__ == "__main__":
    main()
