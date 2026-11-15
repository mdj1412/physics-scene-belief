"""Paired evaluation of within-calibration observation choice. Run on an allocated CPU."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np


def column_stats(samples, estimate):
    return {
        "mean": float(estimate),
        "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "upper95": float(np.quantile(samples, .95)),
        "lower95": float(np.quantile(samples, .05)),
    }


def summarize_ratio(samples, estimate):
    return {
        "ratio": float(estimate),
        "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "upper95": float(np.quantile(samples, .95)),
    }


def strata(cell):
    values = np.unique(cell)
    return [np.flatnonzero(cell == v) for v in values]


def sample_worlds(rng, groups, size):
    return np.concatenate([rng.choice(ids, (size, len(ids)), replace=True) for ids in groups], axis=1)


def crossed_means(values, cell, draws=10000):
    """B x W x M, paired resampling of banks and worlds within the physical cells."""
    banks, worlds, methods = values.shape
    by_world = values.transpose(1, 0, 2).reshape(worlds, banks * methods)
    p = np.full(banks, 1. / banks)
    rng = np.random.default_rng(252301)
    groups = strata(cell)
    out = np.empty((draws, methods))
    for start in range(0, draws, 50):
        size = min(50, draws - start)
        chosen = sample_worlds(rng, groups, size)
        counts = np.zeros((size, worlds))
        np.add.at(counts, (np.arange(size)[:, None], chosen), 1 / worlds)
        bank_weight = rng.multinomial(banks, p, size) / banks
        means = (counts @ by_world).reshape(size, banks, methods)
        out[start:start + size] = np.einsum("ib,ibm->im", bank_weight, means)
    return out


def paired_time_quantiles(times, cell, draws=5000):
    """B x timed-W x repeat-block x M. Each method keeps its matched timing block."""
    banks, worlds, blocks, methods = times.shape
    rng = np.random.default_rng(254919)
    groups = strata(cell)
    result = np.empty((draws, methods))
    for start in range(0, draws, 10):
        size = min(10, draws-start)
        b = rng.integers(banks, size=(size, banks, 1, 1))
        w = sample_worlds(rng, groups, size)[:, None, :, None]
        rep = rng.integers(blocks, size=(size, 1, 1, blocks))
        sample = times[b, w, rep].reshape(size, -1, methods)
        result[start:start+size] = np.quantile(sample, .9, axis=1)
    return result


def mean_if(x, condition):
    return float(x[condition].mean()) if condition.any() else None


def observation_switches(loss, cell, query, probe):
    """Two-action diagnostics. Loss shape B x W x action x independent noise replicate."""
    banks, worlds, _, repeats = loss.shape
    assert repeats % 2 == 0 and repeats >= 2
    first, second = np.split(loss, 2, axis=-1)
    first = first.mean(-1)
    second = second.mean(-1)
    cost = .5 * (first + second)

    # Positive values favor Q. Strict comparisons avoid counting exact ties as reliable.
    d1 = first[:, :, probe] - first[:, :, query]
    d2 = second[:, :, probe] - second[:, :, query]
    c1 = np.where(d1 >= 0, second[:, :, query], second[:, :, probe])
    c2 = np.where(d2 >= 0, first[:, :, query], first[:, :, probe])
    cost_cf = .5 * (c1 + c2)
    q_weight = .5 * ((d1 >= 0).astype(float) + (d2 >= 0))
    # Gap assigned to each winner by its held-out fold. Divide by winner count after resampling.
    q_gap = .5 * (np.where(d1 >= 0, d2, 0.) + np.where(d2 >= 0, d1, 0.))
    p_gap = .5 * (np.where(d1 < 0, -d2, 0.) + np.where(d2 < 0, -d1, 0.))
    stats = np.stack([cost_cf, cost[:, :, query]-cost_cf, cost[:, :, probe]-cost_cf,
                      q_weight, q_gap, p_gap], axis=-1)
    draws = crossed_means(stats, cell)
    means = stats.mean(axis=(0, 1))
    split_reliability = (np.sign(d1) == np.sign(d2)) & (d1*d2 != 0)
    stable = {}
    # Each physical world appears once. Averaging decoder banks does not provide extra worlds.
    mean1, mean2 = d1.mean(0), d2.mean(0)
    for margin in (0, .01, .02, .05):
        q = (mean1 > margin) & (mean2 > margin)
        p = (mean1 < -margin) & (mean2 < -margin)
        by_cell = [{"cell": int(v), "worlds": int((cell == v).sum()),
                    "query": int(q[cell == v].sum()), "probe": int(p[cell == v].sum())}
                   for v in np.unique(cell)]
        stable[f"{margin:.2f}"] = {
            "query_worlds": int(q.sum()), "probe_worlds": int(p.sum()),
            "ambiguous_worlds": int((~q & ~p).sum()),
            "query_fraction": float(q.mean()), "probe_fraction": float(p.mean()), "by_cell": by_cell,
        }

    def conditional_advantage(gap_col, denom, den_draws):
        mask = den_draws > 0
        if denom <= 0 or not mask.any():
            return None
        return column_stats(draws[mask, gap_col] / den_draws[mask], means[gap_col] / denom)

    # If a third view is available, count how often it also beats the nominal Q/P winner.
    best1 = first.argmin(-1)
    best2 = second.argmin(-1)
    threefold = .5*(np.take_along_axis(second,best1[...,None],2)[...,0] +
                    np.take_along_axis(first,best2[...,None],2)[...,0])
    return {
        "noise_per_half": repeats // 2, "unique_worlds": worlds, "decoder_banks": banks,
        "stable_unique_worlds_by_margin": stable,
        "unique_world_gate_declared": bool(min(stable["0.02"]["query_fraction"],
                                               stable["0.02"]["probe_fraction"]) >= .10),
        "bank_world_split_agreement": float(split_reliability.mean()),
        "bank_world_tie_half1": float((d1==0).mean()), "bank_world_tie_half2": float((d2==0).mean()),
        "q_winner_fraction": column_stats(draws[:,3], means[3]),
        "p_winner_fraction": column_stats(1-draws[:,3], 1-means[3]),
        "crossfit_oracle_qp_brier": column_stats(draws[:,0], means[0]),
        "always_query_minus_qp_oracle": column_stats(draws[:,1], means[1]),
        "always_probe_minus_qp_oracle": column_stats(draws[:,2], means[2]),
        "heldout_gain_for_q_winners": conditional_advantage(4,means[3],draws[:,3]),
        "heldout_gain_for_p_winners": conditional_advantage(5,1-means[3],1-draws[:,3]),
        "crossfit_all_actions_oracle_brier": float(threefold.mean()),
        "all_action_first_half_counts": np.bincount(best1.ravel(), minlength=loss.shape[2]).tolist(),
        "all_action_second_half_counts": np.bincount(best2.ravel(), minlength=loss.shape[2]).tolist(),
    }


def paired_effects(chosen, acts, names, cell, gate, comparisons):
    delta = chosen[:, :, gate, None] - chosen
    draws = crossed_means(delta, cell)
    mean_delta = delta.mean(axis=(0,1))
    out = {}
    for j, name in enumerate(names):
        flip = acts[:,:,gate] != acts[:,:,j]
        out[name] = {
            **column_stats(draws[:,j], mean_delta[j]),
            "different_choice_rate": float(flip.mean()),
            "different_choice_worlds": int(flip.any(0).sum()),
            "different_choice_bank_worlds": int(flip.sum()),
            "difference_when_choice_changes": mean_if(delta[:,:,j], flip),
            "per_bank_delta": delta[:,:,j].mean(1).tolist(),
        }
    decision = {names[j]: out[names[j]]["upper95"] < 0 for j in comparisons}
    return out, decision


def file_checks(results):
    for name in ("sources", "plan", "choices_freeze"):
        manifest = json.loads((results / (name + ".json")).read_text())
        for filename, expected in manifest.get("files", manifest).items():
            actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{name} digest mismatch for {filename}")


def method_bins(acts, actions):
    return {n: np.bincount(acts[:,:,i].ravel(), minlength=actions).tolist() for i,n in enumerate(METHOD_NAMES)}


METHOD_NAMES = ["gate", "k256", "query", "probe", "direct", "k1024", "tail_gate", "k128",
                "global", "k256_three", "v24_frozen"]
PRIMARY_NAMES = ["k256", "query"]


def switch_cell_contrasts(loss, cell):
    risk = loss.mean(-1)
    diff = risk[:,:,2] - risk[:,:,0]
    result = {}
    for c, budget in [(2, .03), (1, -.03)]:
        group = cell == c
        values = diff[:,group,None]
        boot = crossed_means(values, cell[group])[:,0]
        row = column_stats(boot, values.mean())
        row["threshold"] = budget
        row["pass"] = row["lower95"] > budget if budget > 0 else row["upper95"] < budget
        result[str(c)] = row
    return result


def summarize_masks(acts, chosen, cost, conf, transported, label, masks):
    # Accept both a world-only route (shared by banks) and a bank-specific confidence decision.
    out = {}
    gate, k256, query = [METHOD_NAMES.index(n) for n in ["gate","k256","query"]]
    for name, mask in masks.items():
        bank_mask = np.broadcast_to(mask, conf.shape)
        if not bank_mask.any():
            out[name] = {"worlds": 0, "bank_worlds": 0}
            continue
        avg = chosen[bank_mask].mean(0)
        focused = acts[bank_mask]
        rows = {
            "worlds": int(bank_mask.any(0).sum()), "bank_worlds": int(bank_mask.sum()),
            "label_rate": float(np.broadcast_to(label, bank_mask.shape)[bank_mask].mean()),
            "direct_rate": float(conf[bank_mask].mean()), "transported_high_rate": float(transported[bank_mask].mean()),
            "actionwise_brier_q_f_p": cost[bank_mask].mean(0).tolist(),
            "mean_loss": dict(zip(METHOD_NAMES, avg.tolist())),
            "action_counts_q_f_p": {n:np.bincount(focused[:,i],minlength=3).tolist() for i,n in enumerate(METHOD_NAMES)},
            "gate_minus_k256": float(avg[gate]-avg[k256]), "gate_minus_query": float(avg[gate]-avg[query])
        }
        out[name] = rows
    return out


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated compute node for this analysis.")
    results = Path("results")
    file_checks(results)
    methods_spec = json.loads((results/"methods.json").read_text())
    names, timed = methods_spec["names"], methods_spec["timed"]
    assert names == METHOD_NAMES and set(timed).issubset(names) and len(set(timed)) == len(timed)
    a, conf, transported, loss, ts = [], [], [], [], []
    ids = None
    for b in range(5):
        with np.load(results/f"choices_{b}.npz") as d:
            a.append(d["acts"].astype(int)-1); conf.append(d["confident"]); transported.append(d["transported_high"])
        with np.load(results/f"eval_{b}.npz") as d:
            loss.append(d["loss"]); ts.append(d["times"])
            if ids is None:
                ids = d["ids"]
            else:
                assert np.array_equal(ids, d["ids"])
    acts = np.stack(a); conf = np.stack(conf).astype(bool); transported = np.stack(transported).astype(bool)
    loss = np.stack(loss); times = np.stack(ts)
    with np.load(results/"test/private.npz") as d:
        cell = d["cell"].astype(int); label=d["label"].reshape(-1)
    assert acts.shape == (5,1280,len(names)) and loss.shape == (5,1280,3,32)
    assert times.shape == (5,80,4,len(timed)) and len(np.unique(ids)) == len(ids)
    assert np.array_equal(np.bincount(cell), [320]*4) and np.array_equal(np.bincount(cell[ids]), [20]*4)
    assert np.isin(label,[0,1]).all() and np.isin(acts,[0,1,2]).all()
    assert np.isfinite(loss).all() and loss.min()>=0 and loss.max()<=1+1e-10
    assert np.isfinite(times).all() and (times>0).all()
    gate, k256, query, probe, direct, k1024, tail_gate, k128, global_policy, k256_three, v24 = range(len(names))
    assert (acts[:,:,query]==0).all() and (acts[:,:,probe]==2).all()
    assert np.isin(np.delete(acts,k256_three,axis=2),[0,2]).all()
    assert np.array_equal(acts[:,:,gate],np.where(conf,acts[:,:,direct],acts[:,:,k1024]))
    assert np.array_equal(acts[:,:,tail_gate],np.where(conf,acts[:,:,k128],acts[:,:,k1024]))
    assert np.array_equal(acts[:,:,v24],np.where(transported,acts[:,:,k1024],acts[:,:,k128]))

    cost = loss.mean(-1)
    chosen = np.take_along_axis(cost, acts, axis=2)
    design = observation_switches(loss, cell, query=0, probe=2)
    design["physical_cell_contrasts"] = switch_cell_contrasts(loss,cell)
    design["design_pass"] = bool(design["unique_world_gate_declared"] and
                                 all(r["pass"] for r in design["physical_cell_contrasts"].values()))
    comparisons, decisions = paired_effects(chosen, acts, names, cell, gate, [k256,query])
    auxiliary = {}
    for name, col in (("tail_gate",tail_gate),("v24_frozen",v24)):
        ref = chosen[:,:,col,None]-chosen[:,:,[k256, query]]
        samples = crossed_means(ref, cell)
        auxiliary[name] = {
            n: column_stats(samples[:,j], ref[:,:,j].mean()) for j,n in enumerate(PRIMARY_NAMES)
        }
    # Exact policy value for a hypothetical routing coin independent of each world, within bank.
    rate = conf.mean(1,keepdims=True)
    blind = rate*chosen[:,:,direct]+(1-rate)*chosen[:,:,k1024]
    blind_delta = chosen[:,:,gate]-blind
    bd = crossed_means(blind_delta[...,None], cell)[:,0]
    blind_result = {"loss":float(blind.mean()), "gate_minus_blind":column_stats(bd,blind_delta.mean()),
                    "rate_matched_per_bank":rate.ravel().tolist()}

    quant = paired_time_quantiles(times, cell[ids])
    tflat = times.reshape(-1,len(timed))
    p50,p90,p95 = np.quantile(tflat,[.5,.9,.95],axis=0)
    timing = {n:{
        "p50_ms":float(p50[j]),"p90_ms":float(p90[j]),"p95_ms":float(p95[j]),
        "p90_ci95_ms":np.quantile(quant[:,j],[.025,.975]).tolist()
    } for j,n in enumerate(timed)}
    g = timed.index("gate")
    ratios = {n:summarize_ratio(quant[:,g]/quant[:,j],p90[g]/p90[j]) for j,n in enumerate(timed)}
    # Matched comparisons do not exchange or bootstrap intervention repeats as independent worlds.
    main_margins = {n:comparisons[n] for n in PRIMARY_NAMES}
    unique_worlds = len(cell)
    primary = {"gate_minus_baselines":main_margins, "passes":decisions,
               "superior_to_both":bool(all(decisions.values())), "worlds":unique_worlds,
               "banks":len(acts),"sensor_noise_replicates":loss.shape[-1]}
    out = {
        "methods":names,"timed_names":timed,"design":design,"primary":primary,
        "mean_loss":dict(zip(names,chosen.mean((0,1)).tolist())), "comparisons":comparisons,
        "support_only_comparisons":auxiliary,"timing":timing,"timing_ratios":ratios,
        "blind_routing":blind_result,"action_counts_q_f_p":method_bins(acts,3),
        "bank_loss":chosen.mean(1).tolist(),"confident_rate":float(conf.mean()),
        "transported_high_rate":float(transported.mean()),"mean_loss_by_action_q_f_p":cost.mean((0,1)).tolist(),
        "cell_rows":summarize_masks(acts,chosen,cost,conf,transported,label,{str(c):cell==c for c in np.unique(cell)}),
        "branches":summarize_masks(acts,chosen,cost,conf,transported,label,{"confident":conf,"uncertain":~conf}),
    }
    encoded = json.dumps(out, indent=2, allow_nan=False)
    tmp = results/"summary.tmp"
    tmp.write_text(encoded)
    tmp.replace(results/"summary.json")
    print("Design",design["design_pass"],"Brier",decisions,"p90 ms",timing["gate"]["p90_ms"],flush=True)


if __name__=="__main__":
    main()
