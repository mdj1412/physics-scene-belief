"""Historical check of the law-transfer approximation by holding out one v39 law."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'probe_transfer_v39'
LAWS=('weakening','strengthening')
BETA=.8289760825845313
def historic_amplitude(law, v=.425):
    if law=='weakening':
        return .19*(-np.expm1(-v/.47))
    t=v/.75
    return .192*2*t/(1+t*t)
def main():
    frozen=json.loads((ROOT/'forecast40.json').read_text())
    source=json.loads((SOURCE/'stats39.json').read_text())
    eps=[]
    rows=[]
    for family in ('shelf','ridge'):
        for old,new in (LAWS,LAWS[::-1]):
            k=f'{family}_{old}'
            target=source['by_cell'][f'{family}_{new}']['mean']
            original=source['by_cell'][k]['mean']
            rho=float(np.clip(historic_amplitude(new)/historic_amplitude(old),-1,1))
            block=[]
            for path in sorted((SOURCE/'results'/k).glob('catalog_*/block_*.npz')):
                with np.load(path) as f:
                    p=f['p_base'][:,None];z=f['z'][:,None,:,None,None];w=f['weights'][:,:,:,None,None]
                    label=f['labels'][...,None]
                    def prob(w):
                        v=np.exp(BETA*z*w)
                        return p*v/(1-p+p*v)
                    a,b=prob(w),prob(.325)
                    d=(a-b)*(a+b-2*label)
                    np.testing.assert_allclose(d.mean(),(f['risk'][:,:,1]-f['risk'][:,:,0]).mean(),atol=1e-14)
                    y=(1+rho)*.5*label+(1-rho)*.5*label[::-1]
                    block.append(((a-b)*(a+b-2*y)).mean())
            assert len(block)==24
            p=float(np.mean(block));eps.append((abs(p-target),abs(original-target)))
            rows.append(dict(shape=family,source_law=old,target_law=new,retention=rho,transferred=p,target=target,copy=original))
    out=dict(cells=rows,mae_signed_transfer_vs_copy=np.mean(eps,axis=0).tolist(),
             historical_source=str(SOURCE.name),v39_completion_hash=frozen['historical_runtime_sha256'])
    (ROOT/'source_law_check40.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
