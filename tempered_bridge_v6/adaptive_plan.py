"""Choose outer refinement from numerical uncertainty only, without outcome data."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze import load,stability,diagnostics

p=argparse.ArgumentParser();p.add_argument('--previous',default='results/main');p.add_argument('--next-h',type=int,default=128)
a=p.parse_args();root=Path(a.previous);rows=[];selected=[]
for folder in sorted(root.glob('case_*')):
    if not (folder/'complete.json').exists():continue
    case=int(folder.name.split('_')[-1]);bad=False;qs={}
    for q in ['high','pixel_only']:
        ds={m:load(folder,q,m) for m in ['bridge','full']};st={m:stability(dd,90000+case) for m,dd in ds.items()}
        dg=diagnostics(ds['bridge'],ds['full']);unstable={}
        for m in ds:
            active=np.array(st[m]['active']);passes=np.array(st[m]['practical_confidence'])
            unstable[m]=np.where(active&~passes)[0].tolist();bad|=bool(unstable[m])
        gain=np.array(dg['full_gain']);regret=np.array(dg['regret']);disagreement=np.where(regret>np.maximum(1e-4,.05*gain.max(1)))[0].tolist()
        bad|=bool(disagreement);qs[q]=dict(stability=st,method_disagreement=disagreement,unstable=unstable)
    rows.append(dict(case=case,refine=bad,quality=qs))
    if bad:selected.append(case)
out=Path('results/adaptive_plans');out.mkdir(exist_ok=True)
(out/f'outer{a.next_h}.json').write_text(json.dumps(dict(previous=a.previous,next_h=a.next_h,cases=selected,rows=rows),indent=2))
print(','.join(map(str,selected)))
