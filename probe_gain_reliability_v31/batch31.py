"""Exact two-action, many-world evaluation of v30's K256 future-risk functional."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'learned_probe_value_v30'))
from contact30 import t, torch
from quota_latency29 import risk_gap
from v21_policy import padded


def batch_gap(engine, x, chunk=4):
    # Each observation uses identical inner/outer Sobol draws, representative weights,
    # tail strata and K256/M2048 policy as v30. Only the unused factual view is omitted.
    answer = []
    for values in t(x).split(chunk):
        size = len(values)
        w = engine.inner.fit(values)
        ow = engine.outer.fit(values)
        z = (engine.inner.label[:, 0] > 0).to(w.dtype)
        oz = (engine.outer.label[:, 0] > 0).to(w.dtype)
        head_w, head = torch.topk(ow,192,dim=1,sorted=False)
        residual = ow.clone().scatter_(1, head, 0.)
        cdf = torch.stack([residual*(1-oz),residual*oz],dim=1).cumsum(-1)
        total = cdf[:,:,-1:]
        ix = torch.searchsorted(cdf.contiguous(), (total*engine.strata[256]).contiguous(),
                                right=True).clamp_max(ow.shape[-1]-1)
        idx = torch.cat([head, ix.flatten(1)],dim=1)
        wt = torch.cat([head_w, (total/32).expand(-1,-1,32).flatten(1)],dim=1)
        reps = engine.reps.flatten().expand(size,-1).contiguous()
        inner_ix = torch.searchsorted(w.cumsum(-1),reps).clamp_max(w.shape[-1]-1).reshape(size,2,256)
        uu = engine.u[:,:512].contiguous().flatten().expand(size,-1).contiguous()
        outer_ix = torch.searchsorted(ow.cumsum(-1),uu).clamp_max(ow.shape[-1]-1).reshape(size,4,512)
        action_risks = []
        for a in (1,3):
            centers = padded(engine.inner.white[a][inner_ix])
            # Flatten four equal-weight 512-draw outer banks for efficient GEMM.
            y = (padded(engine.outer.white[a][outer_ix])+engine.eps[:,:512]).flatten(1,2)
            query_logits = y[:,None] @ centers.transpose(-1,-2) - .5*centers.square().sum(-1)[:,:,None]
            q = (query_logits.softmax(-1) @ z[inner_ix][:,:,:,None]).squeeze(-1)
            mu = padded(engine.outer.white[a][idx])
            prior = wt.log() - .5*mu.square().sum(-1)
            logits = torch.bmm(y,mu.transpose(1,2)) + prior[:,None]
            p = torch.bmm(logits.softmax(-1), oz[idx][:,:,None]).clamp(0,1).transpose(1,2)
            action_risks.append((p*(1-q).square()+(1-p)*q.square()).mean((1,2)))
        answer.append((action_risks[0]-action_risks[1]).detach().cpu().numpy())
    return np.concatenate(answer)


def compare_reference(engine, x):
    ref = np.array([risk_gap(engine,v) for v in x])
    fast = batch_gap(engine, x)
    err = float(np.max(np.abs(ref-fast)))
    if err > 1e-8 or not np.array_equal(np.argsort(-ref,kind="stable")[:13],
                                        np.argsort(-fast,kind="stable")[:13]):
        raise ArithmeticError(f'K256 reference mismatch, max error {err:g}')
    return err
