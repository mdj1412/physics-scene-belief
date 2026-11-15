"""Law-label ablation of the locked contour-specific forecasts."""
import json
import numpy as np
from stats40 import ROOT,CELLS,SAMPLES,interval

def main():
    pre=json.loads((ROOT/'forecast40.json').read_text())['cells']
    q=np.array([pre[c]['delta'] for c in CELLS])
    marginal=q.reshape(2,3,2).mean(axis=2,keepdims=True).repeat(2,axis=2).ravel()
    panel=np.array(json.loads((ROOT/'stats40.json').read_text())['panel_risk'])
    delta=panel[:,:,1]-panel[:,:,0]
    draws=delta[SAMPLES].mean(1)
    def loss(pred): return np.where(pred<0,panel[:,:,1],panel[:,:,0]).mean(1)
    diff=np.abs(draws-marginal).mean(1)-np.abs(draws-q).mean(1)
    routed=loss(marginal)
    d=dict(forecasts=marginal.tolist(), sign_hits=int((marginal*delta.mean(0)>0).sum()),
           mae=np.abs(marginal-delta.mean(0)).mean(),mae_minus_full_ci=np.quantile(diff,[.025,.975]),
           brier=routed.mean(),blind_minus_full=interval(routed-loss(q)),
           blind_minus_fixed=interval(routed-panel[:,:,0].mean(1)),blind_minus_optimizer=interval(routed-panel[:,:,1].mean(1)))
    (ROOT/'law_blind40.json').write_text(json.dumps(d,default=lambda x:x.tolist(),indent=2)+'\n')

if __name__=='__main__':main()
