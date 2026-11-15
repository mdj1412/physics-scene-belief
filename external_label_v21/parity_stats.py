from analyze_v21 import bootstrap,R,PAIRS,read,write,sha,require
import numpy as np

def main():
    require()
    path=R/"parity"
    for name in ("configuration","freeze"):
        for file,digest in read(path/(name+".json"))["files"].items():assert sha(file)==digest
    act=np.load(path/"choices.npz")["acts"]
    costs=np.load(path/"times.npz")["ms"]
    selected=np.stack([np.take_along_axis(np.load(R/f"eval_{i}.npz")["loss"],act[i,:,:,None]-1,axis=-1)[:,:,0] for i in range(len(PAIRS))])
    fixed=np.load(R/"paired_losses.npz")["loss"][:,:,:,read(R/"methods.json").index("fixed")]
    d=selected-fixed
    contrasts,world=bootstrap(np.stack([d[:,:,0],d[:,:,1:].mean(2)],axis=-1),.025)
    per_law,_=bootstrap(d,.05)
    latency={}
    for label,idx in [("nominal",[0]),("unseen",[1,2]),("speed_threshold",[1]),("directional_damping",[2])]:
        times=costs.take(idx,axis=2).reshape(-1,2)
        latency[label]=dict(p50=np.median(times,axis=0).tolist(),p90=np.quantile(times,.9,axis=0).tolist())
    # fraction of compact risk ordering preserved: subtract action-common offsets before RMS.
    names=read(R/"methods.json");control=read(R/"design.json")["selected"].split("m")[0]+"m512"
    sr=np.stack([np.load(R/f"policy_{i}.npz")["risks"] for i in range(len(PAIRS))])
    ar=np.stack([np.load(R/f"policy_{i}.npz")["acts"] for i in range(len(PAIRS))])
    rg=sr[:,:,:,names.index(control)]-sr[:,:,:,names.index("E-E-mean")]
    centered=rg-rg.mean(-1,keepdims=True)
    agreement=(ar[:,:,:,names.index(control)]==ar[:,:,:,names.index("E-E-mean")]).mean((0,1))
    write(path/"summary.json",dict(loss=selected.mean((0,1)).tolist(),fixed=fixed.mean((0,1)).tolist(),
         ci=contrasts,conditional_ci=world,per_law=per_law,bank_delta=d.mean(1).tolist(),timing=latency,
         full_control_agreement=agreement.tolist(),full_control_centered_rms=np.sqrt((centered**2).mean((0,1,3))).tolist()))
if __name__=="__main__":main()
