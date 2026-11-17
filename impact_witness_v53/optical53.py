"""Paired marker line scans at the sacrificial impact, without deployment labels at inference."""
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
V52 = ROOT.parent / "postswap_adaptation_v52"
sys.path.insert(0, str(V52))
from policy52 import V50, bank
from policy53 import witnesses53, mixed53
import world50
from score52 import weighted_cdf_inverse

SCALE = .10  # mm / pixel
THRESHOLD = .15  # full-load differential displacement, mm
JITTER = (.20, .85)
EVENT_P = np.mean([0., .25, .50, 1.])


def read_scans(images):
    """Centroid of two markers in two exposures; no synthetic state in this step."""
    x = np.arange(images.shape[-1])
    floor = np.median(images, axis=-1, keepdims=True)
    light = np.maximum(images.astype(float) - floor - 10., 0.)
    center = (light @ x) / light.sum(-1)
    motion = (center[..., 1, :] - center[..., 0, :]) * SCALE
    differential = motion[..., 1] - motion[..., 0]
    return np.where(abs(differential) >= THRESHOLD, np.sign(differential), 0).astype('int8')


def readings(panel):
    order = bank(panel)[2]
    with np.load(V50 / 'cal50' / f'panel_{panel:02d}.npz') as f:
        family = f['family'][:, order, 3]
        height = f['height'][:, order, 3]
    envs, repeats = family.shape
    x = np.arange(49)
    table = np.empty((3, len(JITTER), envs, repeats), dtype='int8')
    rng = np.random.default_rng([531101, panel])
    blur = rng.uniform(.85, 1.45, size=(*family.shape, 1, 1, 1))
    jitter = rng.normal(size=(*family.shape, 2, 2))
    phase = rng.uniform(-.5, .5, size=(*family.shape, 1, 1))
    translation = rng.normal(0, 1.3, size=(*family.shape, 2, 1))
    noise = rng.normal(0, 2.5, size=(*family.shape, 2, 2, len(x)))
    for state in (-1, 0, 1):
        heights = np.zeros((*family.shape, 2, 2))
        for f, name in enumerate(world50.FAMILIES):
            for h in world50.HEIGHTS:
                points = h + np.array([-1.5, 1.5])
                select = (family == f) & (height == h)
                heights[select, 0, :] = world50.camera_shape(points, name)[0]
                heights[select, 1, :] = world50.loaded_shape(points, h, name, state)[0]
        for j, sig in enumerate(JITTER):
            center = 24. + heights / SCALE + phase + translation + sig * jitter
            image = 18. + 195. * np.exp(-.5*((x-center[..., None])/blur)**2) + noise
            table[state+1,j] = read_scans(np.clip(np.rint(image), 0, 255).astype('uint8'))
    return table


def main():
    assert os.environ.get('SLURM_JOB_ID')
    methods = ('ideal', 'line-scan, 0.20 px', 'line-scan, 0.85 px')
    full = np.empty((12,3,2,5,4,256,3))
    costs = np.empty_like(full)
    counts = np.zeros((2,3,3), dtype=np.int64)
    env, rep = np.indices((4,256))
    for panel in range(12):
        table = readings(panel)
        with np.load(V52 / 'risk52' / f'panel_{panel:02d}.npz') as d:
            exact, realized = d['exact'][0],d['realized']
        for e,event,k in np.ndindex(3,2,5):
            new = 4-k if event else k
            ow, of, nw, nf, truth, _ = witnesses53(panel,k,new,e)
            sign = np.stack([truth,table[truth+1,0,env,rep],table[truth+1,1,env,rep]])
            for j, z in enumerate(sign):
                if j:
                    np.add.at(counts[j-1],(truth.ravel()+1,z.ravel()+1),1)
                alpha=mixed53(ow,of,nw[...,:3],nf[...,:3],nw[...,3],z,method='likelihood')['alpha_joint']
                for out, risk in [(costs,exact[e,event,k,:,None,:]),(full,realized[e,event,k])]:
                    lo,hi,_,gap=np.moveaxis(risk,-1,0)
                    out[panel,e,event,k,:,:,j]=alpha*lo+(1-alpha)*hi-alpha*(1-alpha)*gap
    mean=((1-EVENT_P)*costs[:,:,0]+EVENT_P*costs[:,:,1]).mean(axis=(1,2,3,4))
    p90=[weighted_cdf_inverse([np.sort(full[:,:,ev,...,j].ravel()) for ev in (0,1)],(1-EVENT_P,EVENT_P)) for j in range(3)]
    res=dict(methods=methods, mm_per_pixel=SCALE, dead_zone_mm=THRESHOLD, jitter_px=JITTER,
             mean=mean.mean(0).tolist(),catalog_mean=mean.tolist(),session_p90=p90,
             confusion_true_by_observed=counts.tolist())
    (ROOT/'optical53.json').write_text(json.dumps(res,indent=2))
    print(res['mean'],p90)
if __name__ == '__main__':
    main()
