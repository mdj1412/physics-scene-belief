"""Post-hoc all-object replication in lossless blocks of 8 per environment."""
import argparse
import fcntl
import itertools
from pathlib import Path
import os
import numpy as np
import bullet45 as ext
from bullet45 import b, base, slope

ROOT=ext.ROOT
PARTS=ROOT/"bullet45_parts"
OUT=ROOT/"labels45_bullet_full"


def group(panel, chunk, shape, var, h0, theta, signs, regimes):
    paths={(sf,law):PARTS/f"panel_{panel:02d}"/f"chunk_{chunk:02d}"/f"{shape}_{sf}_{law}_{var}_h{h0}.npz"
           for sf,law in itertools.product(base.SURFACES,base.LAWS)}
    if all(p.exists() for p in paths.values()):return
    data={key:{} for key in paths}
    for offset in base.OFFSETS:
        h=h0+offset
        surface=base.Surface(theta,h,shape,var,base.DT,base.GRID_MM)
        g=theta[:,2]+theta[:,3]*(h/25.)+slope(h,shape,var)
        for sf,law in paths:
            mu=base.material_mu(theta,h,np.zeros_like(g),g,sf,law,signs,regimes)
            obs=ext.simulate(surface,g,mu,theta[:,5])
            for k,v in obs.items():data[sf,law].setdefault(k,[]).append(v.reshape(4,8))
    for key,path in paths.items():
        a={k:np.stack(v) for k,v in data[key].items()}
        a["labels"]=(a["slip_speed"]<=base.THRESHOLD)&a["valid"]
        a["desired_centers_mm"]=h0+np.array(base.OFFSETS)
        a["first_contact_x_mm"]=a["desired_centers_mm"][:,None,None]+a["first_site_delta_mm"]
        if chunk==0:
            with np.load(ROOT/"labels45_bullet"/f"panel_{panel:02d}"/path.name) as orig:
                for name in a:np.testing.assert_array_equal(orig[name],a[name])
        path.parent.mkdir(exist_ok=True,parents=True)
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,**a)
        os.replace(tmp,path)


def worker():
    PARTS.mkdir(parents=True,exist_ok=True)
    b.connect(b.DIRECT)
    for chunk,panel in itertools.product(range(16),range(12)):
        dst=PARTS/f"panel_{panel:02d}"/f"chunk_{chunk:02d}"
        dst.mkdir(parents=True,exist_ok=True)
        done=dst/"DONE"
        if done.exists():continue
        with (dst/"claim").open("w") as lock:
            try:fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if done.exists():continue
            with np.load(ROOT/"results"/"worlds"/f"panel_{panel:02d}.npz") as f:
                theta=f["theta"][:,8*chunk:8*(chunk+1)].reshape(-1,6)
                signs=np.repeat(f["signs"],8);regimes=np.repeat(f["regimes"],8)
            for shape,var,h0 in itertools.product(base.SHAPES,base.VARIANTS,(19,31)):
                group(panel,chunk,shape,var,h0,theta,signs,regimes)
            done.write_text("complete\n")
            print("done",chunk,panel,flush=True)
    b.disconnect()


def combine():
    for p in range(12):
        panel=f"panel_{p:02d}"
        chunks=[PARTS/panel/f"chunk_{k:02d}" for k in range(16)]
        assert all((c/"DONE").exists() for c in chunks)
        names=sorted(q.name for q in chunks[0].glob("*.npz"))
        assert len(names)==48
        for name in names:
            rows=[]
            for d in chunks:
                with np.load(d/name) as a:rows.append({k:a[k] for k in a.files})
            data={k:(rows[0][k] if k=="desired_centers_mm" else np.concatenate([r[k] for r in rows],axis=-1))
                  for k in rows[0]}
            assert data["labels"].shape==(3,4,128)
            path=OUT/panel/name
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,**data);os.replace(tmp,path)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--combine",action="store_true")
    args=p.parse_args()
    if args.combine:combine()
    else:worker()
