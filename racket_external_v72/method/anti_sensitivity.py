"""Generate optical-uncertainty variants without reference to predictions or target residuals."""
import csv
from pathlib import Path
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'audit'/'preston'
def load(name):
    with (DATA/name).open() as f:return {r['id']:r for r in csv.DictReader(f)}
def store(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
    tracks=load('anti_eligible.csv');bounds=load('sign_sensitivity.csv')
    root=HERE/'anti_audited';root.mkdir(exist_ok=True)
    # A margin of zero still requires strict consistency over all validated optical perturbations.
    for variant,margin,ratio,screen in [
        ('interval',0,1,False),('margin_010',.1,1,False),('steady_normal',0,1,True),
        ('post_half',0,.5,False),('post_double',0,2,False)]:
        run=root/variant;run.mkdir(exist_ok=True);rows=[]
        for key,track in tracks.items():
            b=bounds[key]
            if screen and not all(.8<=float(b['dx_half_ratio_'+d])<=1.25 for d in ('in','out')):continue
            track=dict(track)
            for component in ('vx_out','vz_out','omega_out'):
                track[component]=float(track[component])*ratio
            track['sign_resolved']=int(float(b['min_out'])>margin or float(b['max_out'])< -margin)
            rows.append(track)
        dest=run/'observations.csv';store(dest,rows)
        if len(rows)>=9:
            subprocess.run([sys.executable,str(HERE/'anti.py'),str(dest),'--output',str(run)],check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable,str(HERE/'check_anti.py'),str(dest),str(run)],check=True,
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
if __name__=='__main__':main()
