"""Retrieve the identity of example atoms and an independent dense update in physical coordinates."""
from policy22 import RecoveryEngine, PAIRS, TASK, torch, t, DT
from common import read,write,R,require
from model import noise
import numpy as np

def run():
    require();torch.set_num_threads(2)
    summary=read(R/"summary.json");public=np.load(R/"test/public.npz");private=np.load(R/"test/private.npz")
    results=[]
    for x in summary["examples"]:
        engine=RecoveryEngine(PAIRS[x["bank"]])
        i,l,a,n=x["world"],["nominal","speed_threshold","directional_damping"].index(x["law"]),x["action"],x["repeat"]
        context=t(public["ctx"][i,l])
        _,_,ow,z=engine.fit(t(public["x"][i,l]),context,TASK,True)
        y=t(private[f"obs{a}"][i,l,n]);bank=engine.outer
        ix=ow.argsort(descending=True)
        residual=(y-bank.mu[a])/noise(a)
        loglike=-.5*residual.square().sum(-1)
        posterior=(ow.clamp_min(1e-300).log()+loglike).softmax(-1)
        tail=ix[512:];head=ix[:512]
        winner=tail[posterior[tail].argmax()]
        expected=x["metrics"];full_prob=float(posterior@(z[:,0]>0).to(DT))
        assert abs(float(posterior[tail].sum())-expected[1])<1e-5
        assert abs(full_prob-expected[6])<1e-5
        params=bank.theta[winner]
        rows=dict(law=x["law"],rank=int(((ow>ow[winner]).sum()+1).item()),label=int(z[winner,0]>0),
                  atom_before=float(ow[winner]),atom_after=float(posterior[winner]),theta=params.tolist(),
                  whitened_rms_winner=float(residual[winner].square().mean().sqrt()),
                  whitened_rms_factual_map=float(residual[ix[0]].square().mean().sqrt()),
                  label_error_top=expected[7]-expected[6],label_error_keep=expected[8]-expected[6],
                  mean_likelihood_ratio=float(((ow[tail].log()+loglike[tail]).logsumexp(0)-ow[tail].sum().log()-
                                              (ow[head].log()+loglike[head]).logsumexp(0)+ow[head].sum().log()).exp()))
        results.append(rows)
    write(R/"examples_checked.json",results)
if __name__=="__main__":run()
