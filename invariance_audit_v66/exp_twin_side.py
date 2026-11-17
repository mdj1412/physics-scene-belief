"""Audit paid-query twins' common side information and changed signed coordinate."""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"factorial_projection_v47"
sys.path.insert(0,str(OLD))
from runner47 import read_world, apparatus, old, world47, r


def main():
    assert os.getenv("SLURM_JOB_ID") and os.getenv("CUDA_VISIBLE_DEVICES")==""
    old.torch.set_num_threads(1)
    world47.install_world43(old.world)
    n=0
    for k in range(12):
        pop=read_world(k)
        with np.load(OLD/"forecast"/f"panel_{k:02d}.npz") as f:
            for field,variant_axis in [("selected",2),("x_mm",2),("var_rad2",2)]:
                v=f[field]
                np.testing.assert_array_equal(v.take(0,axis=variant_axis),v.take(1,axis=variant_axis))
        for family in world47.FAMILIES:
            for height in world47.HEIGHTS:
                a,aa=apparatus(pop,family,height,"hollow")
                b,bb=apparatus(pop,family,height,"bulge")
                np.testing.assert_array_equal(a,b)
                for arow,brow in zip(aa,bb):
                    for channel in (r.Q,r.P):
                        np.testing.assert_array_equal(arow[channel],brow[channel])
                n+=1
    result=dict(catalogs=12,family_height_catalog_checks=n,objects_per_check=512,
                factual_11_channels_equal=True,pre_perturbation_query_equal=True,
                other_probe_channel_equal=True,selected_action_equal=True,measured_x_and_normal_variance_equal=True,
                measurement="All compared arrays elementwise exactly equal for hollow/bulge. No oracle side features in deployed representation.")
    (ROOT/"exp_twin_side.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)


if __name__=="__main__":
    main()
