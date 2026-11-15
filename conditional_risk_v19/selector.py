"""GPU acquisition using public factual evidence and candidate simulator predictions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adaptive_budget_v18"))
import torch
from model import Bank, DT, DEV, targets
from budget import plans, rep_plan, full_prefix, padded, select
from budget_fast import Engine

METHODS = ("conditional", "self_label", "external")
SIZES = (512, 2048, 4096)
TASKS = ("contact_event", "slip_event")
SEEDS = (213901, 243911)


def action(risk):
    if float(risk.max() - risk.min()) <= 1e-12:
        return 1
    return int(risk.argmin()) + 1


class Selector:
    def __init__(self, seed, inner_seed=183911, outer_seed=193913):
        self.inner = Bank(seed=inner_seed)
        self.outer = Bank(seed=outer_seed)
        self.plans = plans(seed)
        self.original_reps = rep_plan(seed)
        self.u = torch.stack([u for u, _ in self.plans])
        self.eps = torch.stack([eps for _, eps in self.plans])
        self.reps = torch.stack([
            torch.quasirandom.SobolEngine(1, scramble=True, seed=seed + offset)
            .draw(256, dtype=DT).to(DEV).flatten()
            for offset in (2000, 3000)
        ])

    def fit(self, factual, context, task, need_outer):
        w = self.inner.fit(factual)
        z = targets(self.inner.theta, context, task=task)
        if need_outer:
            return w, z, self.outer.fit(factual), targets(self.outer.theta, context, task=task)
        return w, z, None, None

    def mixture_risks(self, state, method, size, prefixes=False, return_identity=False):
        w, z, ow, oz = state
        indices = torch.searchsorted(w.cumsum(0), self.reps).clamp_max(len(w) - 1)
        labels = (z[indices, 0] > 0).to(DT)
        centers = torch.stack([padded(self.inner.white[a][indices]) for a in (1, 2, 3)], 1)
        m = size // 4
        u = self.u[:, :m]
        if method == "external":
            outer_index = torch.searchsorted(ow.cumsum(0), u.contiguous()).clamp_max(len(ow) - 1)
            signal = torch.stack([padded(self.outer.white[a][outer_index]) for a in (1, 2, 3)])[None]
            label = (oz[outer_index, 0] > 0).to(DT)[None, None]
        else:
            outer_index = (u * indices.shape[1]).long().clamp_max(indices.shape[1] - 1)
            signal = centers[:, :, outer_index, :]
            label = labels[:, None, outer_index]
        y = signal + self.eps[None, None, :, :m]
        log_weight = torch.matmul(y, centers[:, :, None].transpose(-1, -2))
        log_weight -= .5 * centers.square().sum(-1)[:, :, None, None]
        posterior = torch.softmax(log_weight, -1)
        q = torch.matmul(posterior, labels[:, None, None, :, None]).squeeze(-1)
        loss = q * (1 - q) if method == "conditional" else (label - q).square()
        if return_identity:
            exact = q * (1 - q).square() + (1 - q) * q.square()
            return float((exact - loss).abs().max()), float(q.min()), float(q.max())
        budgets = SIZES if prefixes else (size,)
        return {n: loss[..., :n // 4].mean((0, 2, 3)) for n in budgets if n <= size}

    def baseline(self, state, name):
        w, z, ow, oz = state
        if name == "nested2048":
            return full_prefix(self.inner, w, z, self.outer, ow, oz, self.plans, [512])[0, :, 0]
        engine = Engine(self.inner, w, z, self.outer, ow, oz, self.plans,
                        self.original_reps, 128, 1024, name == "v18_global")
        if name == "v18_adaptive":
            return select(engine)[1]
        return engine.fixed(128, 1024)

    def decide(self, factual, context, task, name):
        method, _, size = name.partition("/")
        state = self.fit(factual, context, task, method not in ("conditional", "self_label", "query", "probe"))
        if method in METHODS:
            risk = self.mixture_risks(state, method, int(size))[int(size)]
            selected = action(risk)
        elif method in ("query", "probe"):
            selected = 1 if method == "query" else 3
        else:
            selected = action(self.baseline(state, method))
        return selected, state[0], state[1]
