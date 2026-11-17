"""Make neutral-prior source training cases from v42 worlds only. No v43 target import."""
import itertools
import os
from pathlib import Path

import numpy as np

import inference43 as infer
import model43 as model

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'height_offset_v42'/'results'
r = infer.r


def main():
    r.torch.set_num_threads(2)
    if not r.torch.cuda.is_available() or not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('A guarded source GPU job is required')
    engine = infer.make_bank()
    templates = {}
    result = {key:[] for key in ('feature','privileged','label','cue','weight','panel','shape','object_id')}
    for panel, (family,hkey,variant,law) in itertools.product(range(12), r.CONDITIONS):
        f = r.world.FAMILIES.index(family)
        h = r.hvalue(hkey)
        si = panel%2
        sign = r.SIGNS[si]
        cat = 256 + panel
        seed = 420_000_000 + panel*100_000
        context = (family, variant, h, law, sign, cat)
        record = SOURCE/'_'.join((family,hkey,variant,law))/f'catalog_{panel:02d}.npz'
        with np.load(record) as saved:
            alpha = saved['alpha']
            # Latin rotation samples four population pairs across cell strata within panel.
            i0 = (panel + 2*r.HEIGHTS.index(hkey) + 4*r.world.SURFACES.index(law)) % 8
            ids = [i0, (i0+1)%8]
            deployment = [r.world.deployment(seed+200+e,alpha[e],context,r.REGIMES[0],1) for e in ids]
            xx,nom,_,sensors = zip(*deployment)
            xx = np.stack(xx)
            actions = infer.actions43(xx, engine)
            sensors = infer.paid_sensors(sensors,actions)
            snap = saved['labels'][:,si][:,:,ids,:]
            np.testing.assert_array_equal(snap[4,0],np.stack(nom))
            label = np.stack((snap[4].astype(float),np.delete(snap,4,axis=0).mean(0)),axis=-1)
            # [regime,env,n,stratum] -> [env,n,regime,stratum].
            label = label.transpose(1,2,0,3)
            if (family,h) not in templates:
                u = r.world.oracle_direction(family,h)
                templates[family,h] = (infer.make_template(engine,h),infer.make_template(engine,h,u))
            base,priv = [infer.decode43(engine,item,xx,actions,sensors) for item in templates[family,h]]
            feat = model.features(base,xx,actions,sensors,h)
            n = len(ids)*128
            cue = saved['z'][si][ids]
            weight = saved['weights'][si][:,ids].T
            obj_id = np.stack([r.world.draw_indices(128,seed+200+e,20) + e*10000000 for e in ids])
        for key,arr in [('feature',feat.reshape(n,-1)),('privileged',priv.ravel()),('label',label.reshape(n,3,2)),
                        ('cue',np.repeat(cue,128)),('weight',np.repeat(weight,128,axis=0)),
                        ('panel',np.full(n,panel)),('shape',np.full(n,f)),('object_id',obj_id.ravel())]:
            result[key].append(arr)
        if (family,hkey,variant,law) == r.CONDITIONS[-1]:print('source completed panel',panel,flush=True)
    r.v38.atomic_npz(ROOT/'source_matrix43.npz',**{k:np.concatenate(v) for k,v in result.items()})


if __name__ == '__main__':
    with r.torch.inference_mode():
        main()
