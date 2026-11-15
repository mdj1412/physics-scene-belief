"""Observation policies. Only public factual evidence and precomputed model draws enter."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "conditional_risk_v19"))
from selector import Selector, action, padded, DT, DEV

CELLS = [f"{law}-{source}-{mode}" for law in ("C","E") for source in ("C","E") for mode in ("sample","mean")]
TASK = "slip_event"
VIEWS = ("query view", "factual view", "probe")


class Factorial(Selector):
    def __init__(self):
        super().__init__(651701, 637991, 643301)
        self.uniforms = torch.stack([
            torch.rand(128, generator=torch.Generator().manual_seed(673303 + j), dtype=DT).to(DEV) for j in range(4)
        ])
        self.u512 = self.u[:, :128]
        self.noise512 = self.eps[:, :128]

    def features(self, x, ctx, w, z):
        theta = self.inner.theta
        label = z[:,0] / np.sqrt(2.)
        p = w @ label
        law_mass = ((w[:,None] * (theta[:,1,None] == torch.arange(3, device=DEV))).sum(0))
        mean = w @ theta[:,2:]
        sd = (w @ (theta[:,2:] - mean).square()).clamp_min(0).sqrt()
        return torch.cat((x, ctx, p.view(1), (1/w.square().sum()).log().view(1),
                          law_mass, mean, sd)).cpu().numpy()

    def risks(self, state, requested):
        w, z, ow, oz = state
        indices = torch.searchsorted(w.cumsum(0), self.reps).clamp_max(len(w)-1)
        label = (z[indices,0] > 0).to(DT)
        centers = torch.stack([padded(self.inner.white[a][indices]) for a in (1,2,3)],1)
        norm = centers.square().sum(-1) * .5
        output = {}
        for law in ("C","E"):
            names = [name for name in requested if name.startswith(law+"-")]
            if not names:
                continue
            if law == "C":
                index = (self.u512 * 256).long().clamp_max(255)
                signal = centers[:,:,index,:]
            else:
                index = torch.searchsorted(ow.cumsum(0), self.u512.contiguous()).clamp_max(len(ow)-1)
                signal = torch.stack([padded(self.outer.white[a][index]) for a in (1,2,3)])[None]
            y = signal + self.noise512[None,None]
            scores = torch.matmul(y, centers[:,:,None].transpose(-1,-2)) - norm[:,:,None,None]
            q = torch.matmul(scores.softmax(-1), label[:,None,None,:,None]).squeeze(-1)
            probs = {"C": q}
            if any(name.split("-")[1] == "E" for name in names):
                actions = []
                log_prior = ow.clamp_min(1e-300).log()
                outer_label = (oz[:,0] > 0).to(DT)
                for a in (1,2,3):
                    mu = self.outer.white[a]
                    flat_y = y[:,a-1,:,:,:mu.shape[1]].reshape(-1,mu.shape[1])
                    prior_norm = log_prior - self.outer.norm[a]
                    chunks = [(chunk @ mu.T + prior_norm).softmax(-1) @ outer_label
                              for chunk in flat_y.split(128)]
                    actions.append(torch.cat(chunks).reshape(y.shape[0],4,128))
                probs["E"] = torch.stack(actions,1)
            for name in names:
                _, source, mode = name.split("-")
                p = probs[source].clamp(0,1)
                if mode == "mean":
                    loss = p*(1-q).square() + (1-p)*q.square()
                else:
                    sampled = (self.uniforms[None,None] < p).to(DT)
                    loss = (sampled-q).square()
                output[name] = loss.mean((0,2,3))
        return output


def design(features, model):
    x = (features-model["mean"])/model["scale"]
    x = np.clip(x,-8,8)
    return np.concatenate([x,x*x],axis=-1)


def fit_gate(features, loss):
    model = {"mean":features.mean(0),"scale":np.maximum(features.std(0),1e-9)}
    matrix = design(features,model)
    y = loss[:,2,None] - loss[:,:2]
    model.update(center=matrix.mean(0),intercept=y.mean(0))
    centered = matrix-model["center"]
    model["beta"] = np.linalg.solve(centered.T@centered + .05*len(y)*np.eye(centered.shape[1]),
                                    centered.T@(y-model["intercept"]))
    return model


def predict(features, model):
    return (design(features,model)-model["center"]) @ model["beta"] + model["intercept"]
