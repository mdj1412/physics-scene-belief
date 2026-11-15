"""Post-fit interpretation and independent scoring check on allocated CPU."""
import numpy as np
import torch
from scipy.stats import rankdata
from policy import Factorial, TASK, CELLS, predict
from experiment import threshold, R, require, NAMES, public_rows
from common import read, write
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
analysis_spec = spec_from_file_location("v20_analysis", Path(__file__).with_name("analyze.py"))
analysis = module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)
ci = analysis.ci


def reference_check(engine):
    x,ctx=public_rows("validation",0)[1]
    w,z,ow,oz=engine.fit(x,ctx,TASK,True)
    ix=torch.searchsorted(w.cumsum(0),engine.reps)
    result=engine.risks((w,z,ow,oz),CELLS)
    manual={n:[] for n in CELLS}
    for ax in (1,2,3):
        by_rep={n:[] for n in CELLS}
        group=engine.u512.flatten()
        for rep in range(2):
            compact_indices=ix[rep]
            labels=(z[compact_indices,0]>0).to(z.dtype)
            for law in ("C","E"):
                if law=="C":
                    idx=compact_indices[(group*len(labels)).long()]
                    bank=engine.inner
                else:
                    idx=torch.searchsorted(ow.cumsum(0),group.contiguous())
                    bank=engine.outer
                from model import noise
                y=bank.mu[ax][idx]+engine.noise512.reshape(-1,7)[:,:bank.mu[ax].shape[1]]*noise(ax)
                # Use physical coordinates, explicit distances, and the public Bank.update implementation.
                d=(y[:,None]-engine.inner.mu[ax][compact_indices][None])/noise(ax)
                q=(-.5*d.square().sum(-1)).softmax(-1)@labels
                p_ext=engine.outer.update(ow,ax,y)@(oz[:,0]>0).to(oz.dtype)
                for src,p in (("C",q),("E",p_ext)):
                    for mode in ("sample","mean"):
                        val=(q-(engine.uniforms.flatten()<p).to(q.dtype)).square() if mode=="sample" else p*(1-q).square()+(1-p)*q.square()
                        by_rep[f"{law}-{src}-{mode}"].append(float(val.mean()))
        for n in CELLS:manual[n].append(np.mean(by_rep[n]))
    error=max(np.max(np.abs(np.array(manual[n])-result[n].numpy())) for n in CELLS)
    assert error<1e-10,error
    return float(error)


def describe():
    model=dict(np.load(R/"ridge.npz"))
    cfg=read(R/"gate.json")
    rows={}
    for c in (0,1):
        record=np.load(R/f"choices_{c}.npz")
        features=record["features"]
        actions=record["actions"]
        score=predict(features,model)
        assert np.allclose(score,record["gain"])
        g=score.max(1)
        row=np.arange(len(g))
        loss=np.load(R/f"loss_{c}.npz")["loss"]
        selected=actions[:,CELLS.index(cfg["routed"]["method"])]-1
        rate=loss[:,2]-loss[row,selected]
        view_benefit=loss[:,2]-loss[row,score.argmax(1)]
        valid=selected!=2
        on=g>threshold(cfg["routed"])
        label=view_benefit>0
        ranks=rankdata(g,method="average")
        pos=label.sum(); neg=len(g)-pos
        auc=float((ranks[label].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
        cuts=np.digitize(g,cfg["cuts"])
        grouped={}
        for mask,name in [(cuts==i,"Q"+str(i+1)) for i in range(4)]+[(on,"gate_on"),(~on,"gate_off")]:
            p=features[mask,13]
            grouped[name]=dict(n=int(mask.sum()),median_p=float(np.median(p)) if len(p) else None,
                              law_mass_mean=features[mask,15:18].mean(0).tolist() if len(p) else None,
                              view_benefit=ci(view_benefit[mask]),teacher_benefit=ci(rate[mask]),
                              realized_switch=int((on & valid & mask).sum()))
        high=view_benefit[cuts==3];low=view_benefit[cuts==0]
        rng=np.random.default_rng(277134)
        draws=np.concatenate([(high[rng.integers(len(high),size=(400,len(high)))].mean(1)-
                                low[rng.integers(len(low),size=(400,len(low)))].mean(1)) for _ in range(50)])
        rows[str(c)]=dict(auc=auc,positive_view_fraction=float(label.mean()),groups=grouped,
                           upper_lower_ci=[float(high.mean()-low.mean()),*np.quantile(draws,[.025,.975]).tolist()])
    return rows


if __name__=="__main__":
    require()
    torch.set_num_threads(2)
    write(R/"audit.json",dict(risk_max_abs_error=reference_check(Factorial()),groups=describe()))
    print("Direct factorial risks verified for all eight cells; subgroup diagnostics saved")
