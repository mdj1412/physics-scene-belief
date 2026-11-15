from aligned import POLICIES,TEACHERS,limit
from common import R,read,write,require,sha
import numpy as np
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
src=spec_from_file_location("v20analysis",Path(__file__).with_name("analyze.py"))
mod=module_from_spec(src);src.loader.exec_module(mod)
require()
s=R/"aligned"
for name in ["choices_freeze","gate_freeze","source"]:
    for path,h in read(s/(name+".json"))["files"].items():assert sha(path)==h
data=np.load(s/"actions.npz");loss=np.load(s/"loss.npz")["loss"]
a=data["choices"];chosen=np.take_along_axis(loss,a-1,axis=1)
gates=read(s/"gate.json")
out={}
for j,n in enumerate(POLICIES):
    out[n]=dict(loss=float(chosen[:,j].mean()),switch=int((a[:,j]!=3).sum()))
    if n.startswith("aligned_"):
        teacher=n.replace("aligned_","")
        v=data["gain"][:,TEACHERS.index(teacher)]
        gain=chosen[:,0]-chosen[:,POLICIES.index(teacher)]
        cuts=np.digitize(v,gates[teacher]["cuts"])
        out[n].update(called=int((v>limit(gates[teacher])).sum()),contrasts={k:mod.ci(chosen[:,j]-chosen[:,POLICIES.index(k)],.0125) for k in ["probe","fixed_global"]},
                      strata=[mod.ci(gain[cuts==q]) for q in range(4)])
write(s/"summary.json",out)
