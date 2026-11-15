"""Post-result numerical audit; never changes frozen acquisition choices."""
import os,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from model import *

def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2);r=Path('results');q=Quad(128);ref=Quad(192);f=json.loads((r/'test/freeze.json').read_text())
    out=r/'test/evaluation_refined';out.mkdir(exist_ok=True);maxdiff=0.;lossdiff=[]
    for name,hashvalue in f['policies'].items():
        p=r/'test/policies'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==hashvalue
        public=json.loads((r/'test/public'/name).read_text());filename=name.replace('.json','.npz')
        truth=np.load(r/'test/private'/filename);old=np.load(r/'test/evaluation'/filename)
        output={};recorddiff=0.
        for mode in ['boundary','coarse']:
            J,h,ep,eh=info(public,mode);pred=[];plugin=[]
            for a in ACTIONS:
                if a=='none':jj,hh,pp,ee=J,h,ep,eh
                else:jj,hh,pp,ee=update_batch(J,h,ep,eh,a,truth[a],mode)
                pv,am=q.predict(jj,hh,pp,ee,True)
                rr=ref.predict(jj,hh,pp,ee)
                recorddiff=max(recorddiff,float(abs(pv-rr).max()))
                pg=torch.cat([torch.zeros((len(am),1),device=DEV,dtype=DT),am],1)
                em=trunc_mean(pp,ee).expand(len(am));pl=predictions(pg,em)
                if a=='none':pv=pv.expand(16,-1,-1);pl=pl.expand(16,-1,-1)
                pred.append(pv.cpu().numpy());plugin.append(pl.cpu().numpy())
            pred=np.stack(pred);pl=np.stack(plugin)
            if mode=='boundary':output['prediction']=pred;output['losses']=.5*((pred-truth['target'])**2).sum(-1);output['plugin_losses']=.5*((pl-truth['target'])**2).sum(-1)
            else:output['coarse_losses']=.5*((pred-truth['target'])**2).sum(-1)
        for k in ['target','twin_target','factual_equal_max']:output[k]=old[k]
        output['quadrature_prediction_maxdiff']=recorddiff;maxdiff=max(maxdiff,recorddiff)
        lossdiff.append(float(abs(output['losses']-old['losses']).max()))
        np.savez_compressed(out/filename,**output)
    meta=dict(time=time.time(),worlds=128,order=128,reference_order=192,all_worlds_prediction_maxdiff=maxdiff,
              original64_to128_max_single_loss_difference=max(lossdiff),selection_unchanged=True)
    (r/'refinement.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
    assert maxdiff<1e-6,'Quadrature audit not converged'
if __name__=='__main__':main()
