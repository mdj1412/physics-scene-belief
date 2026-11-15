"""Finite-population Bayes reference for the preregistered v38 contact panel.

This evaluator alone knows the held-out geometry, law family, alpha and both
prototype distributions. It integrates, never observes, indices, sign and tier.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REGIMES = ("strong", "weak", "null")
SIGNS = (1, -1)
BETA = .8289760825845313


def norm_kernel(data, mean, sigma):
    """Log Gaussian kernel, including all observation-dependent quadratic terms.

    Omitting the common normalizing constant is valid because every candidate
    state uses the same known diagonal noise. Centering limits cancellation.
    """
    center = mean[0]
    a = (data - center) / sigma
    b = (mean - center) / sigma
    return a @ b.T - .5 * a.square().sum(-1, keepdim=True) - .5 * b.square().sum(-1)


class Reference:
    def __init__(self, shape, surface, alpha):
        import physics38 as world
        self.w = world
        self.torch = world.torch
        self.sigma = [a.double() for a in world.sigmas()]
        self.early = {}
        self.last = {}
        self.dep = {}
        self.labels = {}
        for sign in SIGNS:
            law = (shape, surface, sign)
            cal = world.latent_support(alpha, law, 10)
            dep = world.latent_support(alpha, law, 20)
            assert len(cal) == len(dep) == 2048
            cx = world.factual_clean(cal, law).double()
            dx = world.factual_clean(dep, law).double()
            if sign == 1:
                self.cx, self.dx = cx, dx
            else:
                self.torch.testing.assert_close(cx, self.cx, atol=1.e-12, rtol=0)
                self.torch.testing.assert_close(dx, self.dx, atol=1.e-12, rtol=0)
            for d in SIGNS:
                self.early[sign, d] = world.impact(cal, 0., d*1.10, law, "strong")[0].double()
            self.dep[sign] = {a: m.double() for a, m in world.deployment_clean(dep, law, "strong").items()}
            labels = []
            for regime in REGIMES:
                self.last[sign, regime] = world.impact(cal, .023, 1.10, law, regime)[0].double()
                labels.append(world.impact(dep, .025, .425, law, regime)[1].double())
            self.labels[sign] = self.torch.stack(labels, 1)
        self.torch.testing.assert_close(self.dep[1][world.Q], self.dep[-1][world.Q],
                                        atol=1.e-12, rtol=0)

    def cast(self, x):
        return self.torch.as_tensor(x, device=self.cx.device, dtype=self.torch.float64)

    def calibration(self, xs, y3, dirs, y4):
        torch, w = self.torch, self.w
        kx = norm_kernel(self.cast(xs), self.cx, self.sigma[0])
        # Every observed prototype index was drawn independently with replacement.
        # p(x4) has the same marginal in all latent states, even before y4.
        logs3, logs4 = [], []
        for sign in SIGNS:
            early = self.cast(0.)
            for j, direction in enumerate(dirs):
                ky = norm_kernel(self.cast(y3[j:j+1]), self.early[sign, int(direction)], self.sigma[w.P])
                early = early + torch.logsumexp(kx[j] + ky[0], 0)
            for regime in REGIMES:
                ky = norm_kernel(self.cast(y4[None]), self.last[sign, regime], self.sigma[w.P])
                logs3.append(early)
                logs4.append(early + torch.logsumexp(kx[3] + ky[0], 0))
        return torch.stack([torch.stack(logs3), torch.stack(logs4)])

    def deployment(self, xx, actions, probe, query):
        """Integrate over deployment prototypes per action and *observed* sensor.

        Returns evidence by sign and the probability of stick in each of the six states.
        Other deployment objects' sensor responses never appear in a row's likelihood.
        """
        torch, w = self.torch, self.w
        logx = norm_kernel(self.cast(xx), self.dx, self.sigma[0])
        n, repeats = len(xx), probe.shape[1]
        evidence = self.cx.new_empty((n, repeats, 2))
        label = self.cx.new_empty((n, repeats, 2, 3))
        ess = self.cx.new_empty((n, repeats, 2))
        for a, readings in ((w.Q, query), (w.P, probe)):
            ids = np.flatnonzero(actions == a)
            if len(readings) != len(ids):
                raise ValueError("Compressed observations do not match paid actions")
            for offset in range(0, len(ids), 16):
                subset = ids[offset:offset+16]
                index = torch.as_tensor(subset, device=self.cx.device)
                sims = self.cast(readings[offset:offset+len(index)])
                for si, sign in enumerate(SIGNS):
                    if a == w.P or si == 0:
                        ks = norm_kernel(sims.reshape(-1, sims.shape[-1]),
                                         self.dep[sign][a], self.sigma[a])
                        jointly = ks + logx[index].repeat_interleave(repeats, 0)
                        logz = torch.logsumexp(jointly, 1)
                        weights = (jointly - logz[:, None]).exp()
                        eff = 1 / weights.square().sum(-1)
                    evidence[index, :, si] = logz.reshape(-1, repeats)
                    label[index, :, si] = (weights @ self.labels[sign]).reshape(-1, repeats, 3)
                    ess[index, :, si] = eff.reshape(-1, repeats)
        return evidence, label, ess

    def predict(self, cal, terms):
        evidence, label, ess = terms
        joint = cal.reshape(2, 1, 1, 2, 3) + evidence[None, :, :, :, None]
        weights = self.torch.softmax(joint.flatten(-2), dim=-1).reshape(joint.shape)
        p = (weights * label[None]).sum((-1, -2))
        effective = (weights[-1].sum(-1) * ess).sum(-1)
        return p.cpu().numpy(), effective.cpu().numpy()

    def brute(self, xs, y3, dirs, y4, x, reading, action):
        """Independent, explicit NumPy likelihood on one row for both prefixes."""
        def logkernel(y, means, sigma):
            y, means = np.asarray(y), means.cpu().numpy()
            return -.5 * (((y - means) / sigma.cpu().numpy()) ** 2).sum(axis=-1)
        def logtotal(z):
            m = z.max()
            return m + np.log(np.exp(z-m).sum())
        xcal = [logkernel(xj, self.cx, self.sigma[0]) for xj in xs]
        xdep = logkernel(x, self.dx, self.sigma[0])
        logs, sticks = [], []
        for sign in SIGNS:
            l3 = sum(logtotal(xcal[j] + logkernel(y3[j], self.early[sign,int(d)],
                                                  self.sigma[self.w.P])) for j,d in enumerate(dirs))
            dep = xdep + logkernel(reading, self.dep[sign][int(action)], self.sigma[int(action)])
            for j,r in enumerate(REGIMES):
                l4 = logtotal(xcal[3] + logkernel(y4,self.last[sign,r],self.sigma[self.w.P]))
                logs.append([l3 + logtotal(dep), l3 + l4 + logtotal(dep)])
                sticks.append(np.exp(dep-logtotal(dep)) @ self.labels[sign][:,j].cpu().numpy())
        logs = np.asarray(logs)
        z = np.exp(logs - logs.max(0))
        return (z * np.asarray(sticks)[:,None]).sum(0)/z.sum(0)


def corrected(base, z, weight):
    factor = np.exp(np.clip(BETA * z * weight, -30, 30))
    return base * factor / (1-base+base*factor)


def packed(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_block(src, outdir, cache, save_predictions=True):
    import physics38 as world
    prefix = src.parent.name
    shape, surface = prefix.split("_")
    dst = outdir / prefix / src.name
    dst.mkdir(parents=True, exist_ok=True)
    outfile = dst / "oracle_summary.npz"
    signature = filehash(src/"meta.json") + filehash(Path(__file__)) + filehash(Path(world.__file__))
    if outfile.exists():
        with np.load(outfile) as old:
            if str(old["fingerprint"]) != signature:
                raise RuntimeError("Stale oracle output: "+str(outfile))
        return
    start = time.monotonic()
    rows = {}
    for j, sign_name in enumerate(("plus", "minus")):
        shared = packed(src/f"shared_{sign_name}.npz")
        arms = [packed(src/f"{r}_{sign_name}.npz") for r in REGIMES]
        for e, alpha in enumerate(shared["alpha"]):
            key = (prefix, tuple(alpha))
            if key not in cache:
                cache[key] = Reference(shape, surface, alpha)
            ref = cache[key]
            xcal, y3, dirs = (shared[d][e] for d in ("far_x","first3_y","dirs"))
            xx, actions, pr, qu = (shared[d][e] for d in ("xx","actions","probe_obs","query_obs"))
            terms = ref.deployment(xx, actions, pr, qu)
            for ri, arm in enumerate(arms):
                y4 = arm["fourth_y"][e]
                cal = ref.calibration(xcal, y3, dirs, y4)
                predictions, eff = ref.predict(cal, terms)
                p3, p4 = predictions
                if e == 0 and src.name == "block_000":
                    for a, sensors in ((world.Q,qu), (world.P,pr)):
                        i = np.flatnonzero(actions==a)[0]
                        expected = ref.brute(xcal,y3,dirs,y4,xx[i],sensors[0,0],a)
                        np.testing.assert_allclose(expected,predictions[:,i,0],atol=2.e-8,rtol=0)
                base = shared["p_base"][e]
                truth = arm["labels"][e,:,None]
                weights = arm["weights"][:,:,e]
                policy = corrected(base[None,None], shared["z"][e], weights[:,:,None,None])
                risk = ((policy-truth)**2).mean((-1,-2))
                np.testing.assert_allclose(risk,arm["risk"][:,:,e],atol=1.e-7,rtol=0)
                implied = ((policy-p4)**2).mean((-1,-2))
                target_risk = ((predictions-truth)**2).mean((-1,-2))
                conditional = (p3-p4)**2
                case = dict(gain=conditional.mean(), oracle_risk=target_risk, risk=risk,
                            rb_gain=implied[0]-implied[1], rb_excess=implied,
                            ess=np.array([eff[actions==a].mean() for a in (world.Q,world.P)]),
                            calibration_residual=(predictions-truth).mean((-1,-2)),
                            conditional_gain=np.array([conditional[actions==a].mean() for a in (world.Q,world.P)]))
                for name, val in case.items():
                    if name not in rows:
                        rows[name] = np.empty((2,3,len(shared["alpha"])) + np.shape(val))
                    rows[name][j,ri,e] = val
                if save_predictions:
                    if "p" not in rows:
                        rows["p"] = np.empty((2,3,len(shared["alpha"]))+predictions.shape,dtype=np.float32)
                    rows["p"][j,ri,e] = predictions
    pp = rows.pop("p",None)
    if pp is not None:
        temp = dst/"posterior.tmp.npz"
        np.savez_compressed(temp,p=pp)
        os.replace(temp,dst/"posterior.npz")
    temp = dst/"summary.tmp.npz"
    np.savez_compressed(temp, **rows, signs=SIGNS, regimes=REGIMES, methods=np.array(["fixed","binary","ternary","continuous"]),
                        fingerprint=signature, elapsed=time.monotonic()-start)
    os.replace(temp,outfile)
    print("oracle",prefix,src.name,round(time.monotonic()-start,2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--output",type=Path,default=ROOT/"oracle_results")
    parser.add_argument("--cell", default="*")
    parser.add_argument("--block", default="block_*")
    parser.add_argument("--no-save-predictions",action="store_true")
    parser.add_argument("--follow",action="store_true",help="Watch until all 32 blocks complete (all cells only)")
    args=parser.parse_args()
    import physics38 as world
    world.torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
    cache={}
    done=set()
    with world.torch.inference_mode():
        while True:
            metas=sorted(args.results.glob(f"{args.cell}/{args.block}/meta.json"))
            for path in metas:
                if path not in done:
                    oracle_block(path.parent,args.output,cache,not args.no_save_predictions)
                    done.add(path)
            if not args.follow or len(done)==32:
                break
            time.sleep(4)
    if not done:
        raise RuntimeError("No committed policy block")


if __name__=="__main__":
    main()
