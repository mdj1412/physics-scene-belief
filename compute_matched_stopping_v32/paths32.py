"""Decompose full stopping loss relative to fixed four, and compute uniform-allocation risk."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_eval'
LAWS=('split_static','speed_weak','directional')
METHODS=('broad','repeated','fixed4','uncertainty')
def block(z):
    count=z['counts'];risk=z['oracle_stage_risk'];loss=z['brier']
    early=np.mean((count==2)*(risk[:,0]-risk[:,1])[:,None],axis=0)
    late=np.mean((count==8)*(risk[:,2]-risk[:,1])[:,None],axis=0)
    np.testing.assert_allclose(early+late,loss.mean(0)-loss[:,2].mean(),atol=1e-12)
    random=(risk@np.array([.5,.25,.25])).mean()
    b,r=count[:,:2].T
    scores=z['scores'];unc=z['uncertainty'];ties=[]
    for m in (0,1,3):
        v=scores[:,:,m] if m<2 else unc
        keep=count[:,m]>=4
        ranked_first=np.sort(v[:,0])[::-1]; ranked_last=np.sort(v[keep,1])[::-1]
        ties.append([ranked_first[3]==ranked_first[4],ranked_last[1]==ranked_last[2]])
    overlap=[np.sum((b>=4)&(r>=4))/4,np.sum((b==8)&(r==8))/2,np.mean(b!=r)]
    gap=(z['nominal_labels']!=z['labels']).mean()
    return np.stack([early,late,loss.mean(0)-random]),overlap,ties,random,gap


def main():
    values=[];overlap=[];ties=[];baseline=[];shift=[]
    for law in LAWS:
        records=[]
        for p in sorted((OUT/law).glob('block_*/metrics.npz')):
            with np.load(p) as z:records.append(block(z))
        assert len(records)==16
        for container,field in zip((values,overlap,ties,baseline,shift),zip(*records)):
            container.append(field)
    arr=np.array(values);boot=np.random.default_rng(3285931).integers(16,size=(10000,3,16))
    digest=dict(laws=LAWS,methods=METHODS,measure=['short_of_four','beyond_four','minus_uniform'],
        pooled=arr.mean((0,1)).tolist(),by_law=arr.mean(1).tolist(),
        pooled_pointwise95=np.quantile(arr[np.arange(3)[None,:,None],boot].mean((1,2)),[.025,.975],axis=0).transpose(1,2,0).tolist(),
        overlap_columns=['stage1_top4','final_top2','different_final_budget'],overlap=np.mean(overlap,axis=(0,1)).tolist(),
        ties_by_method=np.sum(ties,axis=(0,1)).astype(int).tolist(),uniform_by_law=np.mean(baseline,axis=1).tolist(),
        target_label_diff_from_coulomb=np.mean(shift,axis=1).tolist())
    (OUT/'paths32.json').write_text(json.dumps(digest,indent=2,allow_nan=False))
if __name__=='__main__':main()
