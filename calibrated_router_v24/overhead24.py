"""Supplemental routing-rule profiling and error localization after model freeze."""
from pathlib import Path
import os,time
os.chdir(Path(__file__).resolve().parent)
from router24 import require,torch,R,Gate,public,np,BANKS,TEST_CALS,RouterEngine,Features,t,REFERENCE,save
from common import read
def probe():
    require();torch.set_num_threads(2)
    cfg=read(R/"selected.json");gate=Gate(cfg["model"]);xs=public("test")
    for b,pair in enumerate(BANKS):
        saved=np.load(R/f"choices_{b}.npz")
        ids=np.load(R/f"eval_{b}.npz")["ids"]
        rng=np.random.default_rng(34221)
        ts=np.empty((3,len(ids),8,2));signals=np.empty((3,len(ids),3))
        for j,cal in enumerate(TEST_CALS):
            eng=RouterEngine(pair,cal);ft=Features(cal)
            for k,i in enumerate(ids):
                x=xs[i,j];st=eng.fit(t(x));v=gate.score(ft(x))
                assert bool(v>=gate.cut)==bool(saved["fire"][i,j])
                full=eng.risk(st,f"keep{REFERENCE}m2048")
                for out,budget in enumerate((cfg["low"],cfg["high"])):
                    delta=eng.risk(st,f"keep{budget}m2048")-full
                    signals[j,k,out]=float((delta-delta.mean()).square().mean().sqrt())
                signals[j,k,2]=v
                for rep in range(-1,8):
                    for method in rng.permutation(2):
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        if method==0:decision=gate.high(ft(x))
                        else:decision=float(eng.score(st))>=0
                        if method==1:torch.cuda.synchronize()
                        us=(time.perf_counter()-start)*1e6
                        if rep>=0:ts[j,k,rep,method]=us
        save(R/f"routing_{b}.npz",times=ts,signals=signals,ids=ids)
if __name__=="__main__":probe()
