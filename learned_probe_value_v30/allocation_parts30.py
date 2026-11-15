"""Post-hoc paired allocation accounting for the source-decoder batch endpoint."""
import json
from pathlib import Path

import numpy as np

from analyze30 import LAWS, RESULTS, quota, selected


def main():
    rows = []
    action_diagnostics = []
    for law in LAWS:
        blocks = []
        action_blocks = []
        for path in sorted((RESULTS / law).glob("block_*/metrics.npz")):
            with np.load(path) as z:
                potentials = selected(z["cost"], quota(z["gap"]))[:, :, 0].mean(1)
                losses = potentials - potentials[:, 1:2]
                counts = z["counts"]
                # Give a zero contribution to unchanged environments; all three
                # columns are averaged over the same complete portfolio of eight.
                early = np.where(counts == 2, losses[:, [0]], 0.).mean(0)
                late = np.where(counts == 8, losses[:, [2]], 0.).mean(0)
                deployed = selected(z["cost"], z["quota_action"])[:, :, 0].mean((0, 1))
                np.testing.assert_allclose(early+late, deployed-deployed[1], atol=1.e-12)
                blocks.append([early, late])
                # These are paired, potential actions. They do not compete under the quota.
                action_blocks.append(z["cost"].mean((0,1)))
        assert len(blocks) == 16
        rows.append(blocks)
        action_diagnostics.append(np.mean(action_blocks,axis=0))
    x = np.asarray(rows)
    differences = x[..., [0]] - x[..., [1, 2]]
    rng = np.random.default_rng(30674165)
    draws = [rng.integers(16, size=(10000, 16)) for law in LAWS]
    paired = np.stack([d[i].mean(1) for d, i in zip(differences, draws)],axis=1)
    summary = dict(potential_action_brier=np.array(action_diagnostics).tolist(),
                   action_order=["query", "probe"], decoder_order=["source", "matched"],
                   stage=["stopped at 2 vs fixed 4", "continued to 8 vs fixed 4"],
                   comparators=["fixed4", "uncertainty"],
                   law_mean=differences.mean(1).tolist(),
                   pooled_mean=differences.mean((0, 1)).tolist(),
                   ci95=np.moveaxis(np.quantile(paired.mean(1), [.025,.975], axis=0), 0,-1).tolist(),
                   note="Descriptive exact accounting on the realized portfolios. A between-group observed difference does not isolate a causal stage effect.")
    (RESULTS / "allocation_parts.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
