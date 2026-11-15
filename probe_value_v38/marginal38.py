"""Exact family-blind oracle sensitivity, matched to the first replicate of four blocks."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import physics38 as world
from oracle38 import Reference, REGIMES, ROOT, norm_kernel, packed, corrected

CELLS = tuple(f"{s}_{l}" for s in world.OBJECTS for l in world.SURFACES)


def x_density(ref, x, catalog):
    """Each observation is independent given the family/alpha, with latent iid prototype."""
    logs=[]
    for k in range(0,len(x),64):
        row=norm_kernel(ref.cast(x[k:k+64]),catalog,ref.sigma[0])
        logs.append(ref.torch.logsumexp(row,1))
    return ref.torch.cat(logs)


def factual_terms(candidates,cal_pool,xx,ids):
    result=[]
    for ref in candidates:
        cal=x_density(ref,cal_pool,ref.cx)
        dep=x_density(ref,xx,ref.dx)
        # Common catalog size and sigma constants cancel across families.
        pooled=cal.sum()+dep.sum()
        idx=world.torch.as_tensor(ids,device=cal.device)
        result.append((pooled,cal[idx],dep))
    return result


def pooled_predictions(candidates,known_index,cal_logs,dterms,fterms):
    """Condition on all factual rows ONCE, then use conditional paid likelihoods."""
    logstates=[]
    labelstates=[]
    for ref,mat,term,fx in zip(candidates,cal_logs,dterms,fterms):
        pooled,cal_x,dep_x=fx
        # mat included (x,y) for the paid calibration objects. Subtract those x's
        # because pooled already incorporated the complete public calibration pool.
        correction=world.torch.stack([cal_x[:3].sum(),cal_x.sum()])
        ll=mat-correction[:,None]
        # Likewise remove the query's factual marginal from its joint paid likelihood.
        observed=term[0]-dep_x[:,None,None]
        logstates.append(pooled+ll.reshape(2,1,1,2,3)+observed[None,:,:,:,None])
        labelstates.append(term[1])
    logstates=world.torch.stack(logstates,3)       # 2, object, noise, cell, sign, tier
    labels=world.torch.stack(labelstates,2)         # object, noise, cell, sign, tier
    weights=world.torch.softmax(logstates.flatten(3),-1).reshape_as(logstates)
    p=(weights*labels[None]).sum((-1,-2,-3))
    known,_=candidates[known_index].predict(cal_logs[known_index],dterms[known_index])
    return p.cpu().numpy(),known,weights.sum((-1,-2)).cpu().numpy()


def environment(src,known_index,e,refs,cache):
    meta=json.loads((src/"meta.json").read_text())
    shape,surface=CELLS[known_index].split("_")
    plus=packed(src/"shared_plus.npz")
    alpha=plus["alpha"][e]; cal_seed=meta["calibration_seeds"][e]
    # Family metadata is used to reconstitute public sensor recordings only.
    # All hypotheses in the evaluator below receive the same observed numbers.
    actual=world.draw_object(1024,cal_seed,10,alpha,(shape,surface,1))
    pool=world.factual(actual,cal_seed,11,(shape,surface,1))
    np.testing.assert_allclose(pool[plus["ids"][e]],plus["far_x"][e],rtol=0,atol=1.e-9)
    xx=plus["xx"][e]
    fterms=factual_terms(refs,pool,xx,plus["ids"][e])
    logs=world.torch.stack([a[0] for a in fterms])
    family_xprob=world.torch.softmax(logs,0).cpu().numpy()
    cases=[]
    for orient in ("plus","minus"):
        shared=plus if orient=="plus" else packed(src/"shared_minus.npz")
        np.testing.assert_array_equal(shared["xx"][e],xx)
        np.testing.assert_array_equal(shared["ids"][e],plus["ids"][e])
        np.testing.assert_array_equal(shared["far_x"][e],plus["far_x"][e])
        xs,y3,dirs,actions=(shared[d][e] for d in ("far_x","first3_y","dirs","actions"))
        pr=shared["probe_obs"][e,:,:1]; qu=shared["query_obs"][e,:,:1]
        dt=[ref.deployment(xx,actions,pr,qu) for ref in refs]
        for r in REGIMES:
            target=packed(src/f"{r}_{orient}.npz")
            cal=[ref.calibration(xs,y3,dirs,target["fourth_y"][e]) for ref in refs]
            m,k,cell=pooled_predictions(refs,known_index,cal,dt,fterms)
            # Conditional forecast with the true family should agree with the stored
            # primary oracle whenever the optional large posterior archive was saved.
            if cache is not None:
                saved=cache[("plus","minus").index(orient),REGIMES.index(r),e,:,:,:1]
                np.testing.assert_allclose(k,saved,rtol=0,atol=5.e-8)
            truth=target["labels"][e,:,None]
            probs=np.stack([k,m],0)              # known/hidden x prefix 3/4
            policy=corrected(shared["p_base"][e,:,:1][None,None],shared["z"][e],
                             target["weights"][:,:,e,None,None])
            rb=((policy[None]-probs[:,1:2,None])**2).mean((-1,-2))
            cases.append(dict(orientation=orient,regime=r,environment=e,
                known_gain=float(((k[1]-k[0])**2).mean()),
                hidden_gain=float(((m[1]-m[0])**2).mean()),
                family_factual_p=family_xprob.tolist(),
                family_paid_p=cell.mean((1,2)).tolist(),
                known_risk=((k-truth)**2).mean((1,2)).tolist(),
                hidden_risk=((m-truth)**2).mean((1,2)).tolist(),
                abs_probability_difference=np.abs(k-m).mean((1,2)).tolist(),
                rb_policy_gain=(rb[:,0]-rb[:,1]).tolist()))
    return cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--output",type=Path,default=ROOT/"marginal38.json")
    parser.add_argument("--oracle",type=Path,default=ROOT/"oracle_results")
    parser.add_argument("--block",default="block_000")
    args=parser.parse_args()
    refs={}
    world.torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
    results={}
    start=time.monotonic()
    with world.torch.inference_mode():
        for j,cell in enumerate(CELLS):
            src=args.results/cell/args.block
            with np.load(src/"shared_plus.npz") as sh: alphas=sh["alpha"].copy()
            archive=args.oracle/cell/args.block/"posterior.npz"
            saved=packed(archive)["p"] if archive.exists() else None
            results[cell]=[]
            for e,a in enumerate(alphas):
                aa=tuple(a)
                if aa not in refs:
                    refs[aa]=[Reference(*c.split("_"),a) for c in CELLS]
                results[cell].extend(environment(src,j,e,refs[aa],saved))
            print("marginal",cell,round(time.monotonic()-start,2),flush=True)
    obj=dict(cells=CELLS,block=args.block,replicate=0,alpha="known",
             prior="4 cells x 2 signs x 3 local regimes, independent and uniform",
             observations="all 1024 calibration factual rows, all 128 deployment factual rows, four/threes rows of paid calibration, focal paid deployment observation only",
             seconds=time.monotonic()-start,cases=results)
    tmp=args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n")
    os.replace(tmp,args.output)


if __name__=="__main__":
    main()
