"""Post hoc diagnostic: which errors are removable by changing the expert switch alone?"""
from replay import HERE, R, read, state
import json
import numpy as np


def score(z,y):
    d=z-y
    return dict(v_mae=np.linalg.norm(d[:,:2],axis=1).mean(),w_mae=abs(d[:,2]).mean()/R)


def main():
    folds=json.loads((HERE/'metrics.json').read_text())['folds']
    out={}
    for name,fold in folds.items():
        rows=sum([read('test',c) for c in fold['held']],[])
        x,y=state(rows,'in'),state(rows,'out'); ep=fold['empirical']
        a=np.column_stack([np.ones(len(x)),(x-ep['mean'])/ep['std']])
        gate=a@ep['logistic']>0
        actual=y[:,0]-y[:,2]>0
        minus,plus=a@ep['expert_minus'],a@ep['expert_plus']
        zp=np.where(gate[:,None],plus,minus)
        zy=np.where(actual[:,None],plus,minus)
        vm=np.minimum(np.linalg.norm((plus-y)[:,:2],axis=1), np.linalg.norm((minus-y)[:,:2],axis=1)).mean()
        wm=np.minimum(abs(plus[:,2]-y[:,2]),abs(minus[:,2]-y[:,2])).mean()/R
        margins={}
        for h in (.25,.5,1.):
            ids=abs(y[:,0]-y[:,2])>h
            margins[str(h)]=dict(n=int(sum(ids)),accuracy=float(np.mean(gate[ids]==actual[ids])) if ids.any() else None)
        out[name]=dict(margins=margins,n=len(x),accuracy=np.mean(gate==actual),pred_positive=np.mean(gate),
            true_positive_rate=np.mean(actual),gated=score(zp,y),oracle_true_sign=score(zy,y),
            oracle_best_each_endpoint=dict(v_mae=vm,w_mae=wm))
    (HERE/'gate_audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
