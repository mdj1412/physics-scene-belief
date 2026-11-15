import os,json
import numpy as np
from analyze27 import RESULT,CFG,read_case,bootstrap,bands,sha
def main():
    assert os.environ.get('SLURM_JOB_ID')
    families=['neg_neg','neg_pos','pos_neg','pos_pos']
    loss=[];maes=[];fits=[]
    for f in families:
        block=[]
        for e in [e for e in CFG['environments'] if e['family']==f]:
            cur=read_case(e);folder=RESULT/'context27'/e['id']
            ext=json.loads((folder/'metrics.json').read_text())
            assert sha(folder/'metrics.npz')==ext['npz_sha256'] and sha(folder/'choice_lock.npz')==ext['choices_sha256']
            assert ext['main_npz_sha256']==cur['d']['npz_sha256']
            with np.load(folder/'metrics.npz') as v:
                co=v['common_known'];lo=v['loss']
                xx=cur['loss']
                block.append(np.column_stack([xx[:,0],lo,xx[:,5],xx[:,3],xx[:,6],co,xx[:,11],xx[:,9]]))
            maes.append([abs(cur['d']['estimate']['alpha'][0]-e['alpha'][0]),abs(ext['estimate']['alpha'][0]-e['alpha'][0])])
            fits.append(ext['setup_ms']['detector_fit'])
        loss.append(block)
    x=np.array(loss)
    indices=[(1,0),(1,3),(2,3),(1,2),(2,4),(6,5),(6,8),(7,8),(6,7),(7,9)]
    deltas=np.stack([x[...,a]-x[...,b] for a,b in indices],-1)
    pt=deltas.mean((0,1,2)); boot=bootstrap(deltas).mean(1)
    d=dict(order=['est1024','est16384','known_profile','neutral','known_both']*2,
        loss=x.mean((0,1,2)).tolist(),by_law=x.mean((1,2)).tolist(),families=families,contrasts=indices,
        deltas=pt.tolist(),sim95=bands(boot,pt).tolist(),maes=np.mean(maes,axis=0).tolist(),fit_ms=np.mean(fits))
    (RESULT/'extended.json').write_text(json.dumps(d,indent=2))
if __name__=='__main__':main()
