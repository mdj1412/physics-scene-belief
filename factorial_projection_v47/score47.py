"""Score the frozen routes with catalog-level pairing. Run on a CPU worker."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
POLICIES = ["always_original", "frozen_v46", "observation_only", "always_project"]


def interval(arr, comparisons=1):
    values = np.asarray(arr, dtype=float)
    mean = values.mean(axis=0)
    half = t.ppf(1 - .025 / comparisons, len(values) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return np.stack((mean, mean - half, mean + half), axis=-1)


def evaluate(panel):
    tag = f"panel_{panel:02d}.npz"
    with np.load(ROOT / "forecast" / tag) as pred, np.load(ROOT / "labels" / tag) as truth:
        prob = [np.asarray(pred[key], dtype=float) for key in ("p_original", "p_projected")]
        assert all(p.shape == (4, 2, 2, 2, 4, 128, 4) for p in prob)
        assert all(np.isfinite(p).all() and p.min() >= -1e-12 and p.max() <= 1 + 1e-12 for p in prob)
        paid = np.isfinite(pred["query_paid"]).all(-1)
        assert not (np.isfinite(pred["query_paid"]).any(-1) ^ paid).any()
        assert (paid == paid[:, :1, :, :, :, :, :1]).all()
        assert np.max(np.abs(prob[0][~paid] - prob[1][~paid]), initial=0.) < 1e-9
        y = truth["labels"]
        assert y.shape == (4, 2, 2, 2, 2, 3, 4, 128) and np.isin(y, [0, 1]).all()
        new = pred["gate_new"]
        old = pred["gate_v46"][:, None, ..., None]
        assert new.shape == (4, 2, 2, 2, 3, 4, 128, 4) and old.shape == (4, 1, 2, 2, 3, 4, 128, 1)
        assert np.isin(new, [0,1]).all() and np.isin(old, [0,1]).all()
        old = np.broadcast_to(old, new.shape).astype(bool)
        new = new.astype(bool)

        # risk arrays ordered family, alignment, height, variant, coupling, radius, offset, env, object, noise
        label = y[:, None, ..., None].astype(float)
        probabilities = [p[:, :, :, :, None, None, None, :, :, :] for p in prob]
        risks = [(p - label) ** 2 for p in probabilities]
        d = risks[1] - risks[0]
        include = np.broadcast_to(paid[:, :, :, :, None], new.shape)
        routes = [np.zeros_like(new), old, new, np.ones_like(new)]
        loss = []
        family = []
        for route in routes:
            cost = risks[0] + d * route[:, :, :, :, None, None]
            loss.append(cost.mean(axis=(0,2,3,5,6,7,8,9)))  # alignment, coupling
            family.append(cost.mean(axis=(1,2,3,4,5,6,7,8,9)))
        # Independent direct route score (before averaging, including every noise draw)
        direct = (np.where(new[:, :, :, :, None, None], probabilities[1], probabilities[0]) - label)**2
        np.testing.assert_allclose(direct.mean(), np.mean(loss[2]), atol=2e-13)
        add = d * (new & ~old)[:, :, :, :, None, None]
        remove = -d * (old & ~new)[:, :, :, :, None, None]
        np.testing.assert_allclose((add + remove).mean(), np.mean(loss[2])-np.mean(loss[1]), atol=1e-13)
        route_rates = [[gate.mean(), (gate & include).sum()/include.sum()] for gate in routes]
        # Shuffle within the same family, height, offset and purchased-query stratum.
        axes = (1,3,5,6,7)
        mix = (new & include).sum(axis=axes, keepdims=True) / include.sum(axis=axes, keepdims=True).clip(min=1)
        uniform = risks[0] + d * (include * mix)[:, :, :, :, None, None]
        cell_d = d.mean(axis=(3,5,7,8,9)) # family, alignment, height, coupling, offset
        cell_o = risks[0].mean(axis=(3,5,7,8,9))
        return dict(loss=loss, family=family, rates=route_rates, switch=(new != old).mean(),
                    switch_paid=((new != old) & include).sum()/include.sum(), paid=include.mean(),
                    new_minus_shuffled=direct.mean()-uniform.mean(), add=add.mean(), remove=remove.mean(),
                    flips=(y[:,:,0] != y[:,:,1]).mean(axis=(0,1,3,4,5,6)),
                    coupling_flips=(y[:,:,:,0] != y[:,:,:,1]).mean(),
                    new_rate_alignment=(new&include).sum(axis=(0,2,3,4,5,6,7))/include.sum(axis=(0,2,3,4,5,6,7)),
                    label_rate=y.mean((0,1,2,4,5,6,7)), cell_d=cell_d, cell_o=cell_o)


def main():
    panels = [evaluate(i) for i in range(12)]
    def get(k):
        return np.asarray([p[k] for p in panels])
    cell = get("loss")
    loss = cell.mean((2,3))
    comp = np.column_stack((loss[:,2]-loss[:,0], loss[:,2]-loss[:,1]))
    forced = cell[:,3] - cell[:,0]
    by_family = get("family")
    # Diagnose extra value of being told coupling to estimate a GROUP rule, scored on independent catalogs.
    d, orig = get("cell_d"), get("cell_o")
    assert d.shape == (12,4,2,2,2,3)
    oracle, disagrees = [], []
    for k in range(len(panels)):
        source_d = np.delete(d,k,axis=0).mean(axis=0)
        with_c, without_c = source_d < 0, source_d.mean(axis=3,keepdims=True) < 0
        oracle.append([(orig[k]+d[k]*route).mean() for route in (without_c,with_c)])
        disagrees.append(np.broadcast_to(without_c,with_c.shape).__ne__(with_c).mean())
    oracle = np.array(oracle)
    out = dict(methods=POLICIES, order="alignment low/high; coupling null/strong; intervals mean, lower, upper",
      catalog_loss=loss, absolute=interval(loss), primary=interval(comp,2), negative_catalogs=(comp<0).sum(0),
      absolute_by_factor=cell.mean(0), forced_by_factor=interval(forced,4),
      policy_vs_original_by_factor=interval(cell[:,2]-cell[:,0],4),
      forced_alignment_effect=interval(forced[:,1]-forced[:,0],2),
      forced_interaction=interval((forced[:,1,1]-forced[:,0,1])-(forced[:,1,0]-forced[:,0,0])),
      family_absolute=by_family.mean(0), family_policy_vs_original=interval(by_family[:,2]-by_family[:,0],4),
      projection_rates=get("rates").mean(0), switch_all_ci=interval(get("switch")),switch_paid_ci=interval(get("switch_paid")),
      paid_rate=get("paid").mean(), new_minus_shuffled_ci=interval(get("new_minus_shuffled")),
      v46_switch_decomposition=np.array([get("add").mean(), get("remove").mean()]),
      new_rate_low_high=get("new_rate_alignment").mean(0), variant_flip=interval(get("flips"),2),
      coupling_flip=interval(get("coupling_flips")), stick_prevalence=interval(get("label_rate"),2),
      informed_group_rule_abs=oracle.mean(0), informed_minus_uninformed_ci=interval(oracle[:,1]-oracle[:,0]),
      group_rule_disagree=np.mean(disagrees))
    (ROOT/"score47.json").write_text(json.dumps(out, default=lambda x:np.asarray(x).tolist(),allow_nan=False,indent=1)+"\n")
    print("held-out Brier",loss.mean(0),"paired differences",out["primary"])


if __name__ == "__main__":
    main()
