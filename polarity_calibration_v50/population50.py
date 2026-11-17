"""Source-disjoint catalogs; calibration and deployment draws come from separate support IDs."""
import os
import sys
import numpy as np
from world50 import ROOT, FAMILIES, SURFACES, LAWS, install_world43
sys.path.insert(0,str(ROOT.parent/'contact_patch_v44'))
import runner44 as old


def draw_set(alpha,support,seed,lane):
    theta,ind=[],[]
    for e,row in enumerate(alpha):
        bank=old.world._catalog(support,tuple(map(float,row)),lane,True)
        idx=np.random.default_rng(np.random.SeedSequence([seed,e,lane])).choice(len(bank),128,replace=False)
        theta.append(bank[idx].cpu().numpy())
        ind.append(idx)
    return np.stack(theta),np.stack(ind)


def main():
    assert os.getenv('SLURM_JOB_ID') and not old.torch.cuda.is_available()
    old.BASE=500000000
    install_world43(old.world)
    (ROOT/'worlds').mkdir(exist_ok=True)
    for i in range(12):
        seed,alpha,sign,regimes=old.population(i)
        seed_cal=530000000+100000*i
        dep,idx=draw_set(alpha,900+i,seed,20)
        cal,cal_idx=draw_set(alpha,920+i,seed_cal,10)
        k=(np.arange(4)+i)%4
        sf=np.asarray(SURFACES)[k//2]
        law=np.asarray(LAWS)[k%2]
        old.v38.atomic_npz(ROOT/'worlds'/f'panel_{i:02d}.npz',theta=dep,object_index=idx,
                           cal_theta=cal,cal_object_index=cal_idx,alphas=alpha,signs=sign,regimes=regimes,
                           seed=seed,cal_seed=seed_cal,support_panel=900+i,cal_support_panel=920+i,
                           surface=sf,law=law)

if __name__=='__main__':
    main()
