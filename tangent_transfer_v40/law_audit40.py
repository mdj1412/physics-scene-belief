"""Describe paired contact-law shifts on the realized benchmark after the freeze."""
import json
import os
from pathlib import Path
import numpy as np
import physics40 as w
from audit_physics40 import geom, rng

ROOT=Path(__file__).resolve().parent


def tangential(theta,height,speed,family):
    slope=theta[:,2]+theta[:,3]*(height/.025)+geom(height,family,'base')[0]
    return (speed+slope)/np.hypot(1,slope)


def main():
    if not (ROOT/'freeze40.sha256').exists() or os.environ.get('BELIEF_DEVICE')!='cpu':
        raise RuntimeError('Use CPU after freeze')
    w.torch.set_num_threads(1)
    output={}
    for family in ('shelf','ridge'):
        fields={k:[] for k in ('fourth_speed','query_speed','fourth_saturating',
                 'fourth_crossover','fourth_disagree','query_saturating','query_crossover','query_disagree')}
        for k in range(12):
            for b in range(2):
                seed=400000000+k*100000+b*10000
                case={}
                for law in ('saturating','crossover'):
                    path=ROOT/'results'/f'{family}_base_{law}'/f'catalog_{k:02d}'/f'block_{b:02d}.npz'
                    with np.load(path) as a:
                        case[law]={n:a[n] for n in ('alpha','fourth_stick','labels')}
                s,c=case.values()
                for name,arch in (('fourth','fourth_stick'),('query','labels')):
                    fields[name+'_saturating'].append(s[arch].ravel())
                    fields[name+'_crossover'].append(c[arch].ravel())
                    fields[name+'_disagree'].append((s[arch]!=c[arch]).ravel())
                for e,al in enumerate(s['alpha']):
                    sc,sd=seed+100+e,seed+200+e
                    ids=rng(sc,98).permutation(1024)[:4]
                    ci=rng(sc,4010).integers(2048,size=1024)[ids[3:4]]
                    di=rng(sd,4020).integers(2048,size=128)
                    setup=(family,'base','saturating',1,k)
                    tc=w.latent_support(al,setup,10).cpu().numpy()[ci]
                    td=w.latent_support(al,setup,20).cpu().numpy()[di]
                    fields['fourth_speed'].append(tangential(tc,.023,1.10,family))
                    fields['query_speed'].append(tangential(td,.025,.425,family))
        fields={k:np.concatenate(v) for k,v in fields.items()}
        summary={}
        for name,vec in fields.items():
            if name.endswith('speed'):
                summary[name]=dict(n=len(vec),minimum=float(vec.min()),maximum=float(vec.max()),
                                   mean=float(vec.mean()),fraction_abs_below_crossover=float((abs(vec)<.70).mean()),
                                   fraction_negative=float((vec<0).mean()))
            else:
                summary[name]=float(vec.mean())
        output[family]=summary
    (ROOT/'law_shift40.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
