from run import METHODS, NAMES, CELLS
from common import R, read, write, sha, require
import numpy as np


def interval(x, quantiles=(.025, .975)):
    rng = np.random.default_rng(767119)
    draws = []
    for _ in range(100):
        idx = rng.integers(len(x), size=(200, len(x)))
        draws.extend(x[idx].mean(1))
    return [float(x.mean()), *np.quantile(draws, quantiles).tolist()]


def main():
    require()
    for file, digest in read(R / "sources.json").items():
        assert sha(file) == digest, file
    for file, digest in read(R / "freeze.json")["files"].items():
        assert sha(file) == digest, file
    summary = {}
    pilot = read(R / "pilot.json")["selected"]
    for c, k in CELLS:
        primary = np.load(R / f"policy_{c}_{k}_0.npz")
        actions, risk = primary["choices"], primary["risks"]
        assert actions.shape == (512, len(NAMES)) and np.isfinite(risk).all()
        assert np.isin(actions, [1, 2, 3]).all()
        action_loss = np.load(R / f"loss_{c}_{k}.npz")["world_action_loss"]
        assert action_loss.shape == (512, 3) and np.isfinite(action_loss).all()
        losses = np.take_along_axis(action_loss, actions - 1, axis=1)
        other = np.load(R / f"policy_{c}_{k}_1.npz") if c == 0 else None
        rows = {}
        for j, name in enumerate(NAMES):
            row = dict(loss=float(losses[:,j].mean()), choices=np.bincount(actions[:, j],minlength=4)[1:].tolist())
            if other is not None:
                row["agreement"] = float(np.mean(other["choices"][:,j] == actions[:,j]))
                if j < risk.shape[1]:
                    gap = risk[:,j,2] - risk[:,j,0]
                    alt = other["risks"][:,j,2] - other["risks"][:,j,0]
                    row["gap_change_rms"] = float(np.mean((gap-alt)**2)**.5)
                ref = primary["reference"]
                idx = actions[:len(ref), j] - 1
                row["reference_agreement"] = float(np.mean(idx == ref.argmin(1)))
                row["reference_regret"] = float(np.mean(ref[np.arange(len(ref)),idx] - ref.min(1)))
            rows[name] = row
        conditional = "conditional/4096"
        contrasts = {}
        pairs = [(conditional, b) for b in ("self_label/4096", "external/4096", "nested2048")]
        if pilot["conditional"] and pilot["external"]:
            pairs.append((f"conditional/{pilot['conditional']}", f"external/{pilot['external']}"))
        for a, b in pairs:
            delta = losses[:,NAMES.index(a)] - losses[:,NAMES.index(b)]
            contrasts[a + " minus " + b] = interval(delta, (.00625, .99375) if (c,k)==(0,1) else (.025,.975))
        agree_delta = None
        if other is not None:
            agreement = (other["choices"] == actions).astype(float)
            agree_delta = interval(agreement[:,NAMES.index(conditional)] - agreement[:,NAMES.index("self_label/4096")])
        summary[f"{c}_{k}"] = dict(rows=rows, contrasts=contrasts, agreement_delta=agree_delta)
    write(R / "summary.json", summary)
    print("Validated world-level paired contrasts, computed summary", flush=True)


if __name__ == "__main__":
    main()
