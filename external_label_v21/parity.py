import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
from run_v21 import StudyEngine,PAIRS,LAWS,public,private,online_times,action,TASK,R,save,write,read,freeze,require
import torch
import numpy as np

def run():
    require();torch.set_num_threads(2)
    path=R/"parity"
    path.mkdir(exist_ok=True)
    eng=StudyEngine((8803001,9903017))
    names=[f"top{k}m{m}" for k in (512,2048,4096) for m in (1024,2048,4096)]+["fixed"]
    ti=online_times(eng,public("development"),private("development"),names,[(i,0) for i in range(24)],3)
    q=np.quantile(ti.reshape(-1,len(names)),.9,axis=0)
    eligible=np.where(q[:-1]<=q[-1])[0]
    j=eligible[q[eligible].argmax()] if len(eligible) else abs(q[:-1]-q[-1]).argmin()
    chosen=names[j]
    write(path/"design.json",dict(options=names,p90=q.tolist(),selected=chosen,in_cap=bool(len(eligible))))
    freeze(path/"configuration.json",[path/"design.json",Path("parity.py"),Path("LATENCY_PROTOCOL.md")])
    print("Parity candidate frozen",chosen,q[j]/q[-1],flush=True)
    x,ctx=public("test");all_choices=[]
    for b,pair in enumerate(PAIRS):
        eng=StudyEngine(pair);choices=np.empty((len(x),3),int)
        for i in range(len(x)):
            for l in range(3):
                state=eng.fit(x[i,l],ctx[i,l],TASK,True)
                choices[i,l]=action(eng.risk(state,chosen))
        all_choices.append(choices);del eng
        print("Extra policy complete",b,flush=True)
    acts=np.stack(all_choices);save(path/"choices.npz",acts=acts)
    freeze(path/"freeze.json",[path/"choices.npz",path/"configuration.json"])
    obs=private("test")
    tm=[]
    for b,pair in enumerate(PAIRS):
        eng=StudyEngine(pair)
        fixed=np.load(R/f"policy_{b}.npz")["acts"][:,:,read(R/"methods.json").index("fixed")]
        exp=np.stack([acts[b],fixed],axis=-1)
        ti=online_times(eng,(x,ctx),obs,[chosen,"fixed"],[(i,l) for i in range(16) for l in range(3)],2,exp)
        tm.append(ti.reshape(16,3,2,2));del eng
    save(path/"times.npz",ms=tm)
if __name__=="__main__":run()
