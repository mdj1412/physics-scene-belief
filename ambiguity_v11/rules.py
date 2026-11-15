"""Factual-evidence-only rules, fit to development Bayes risk labels."""
import json,time,os
from pathlib import Path
import numpy as np
from run import R,SPLITS,write
from model import FY,L

def data(split):
    x=[];y=[]
    for i in range(SPLITS[split][0]):
        p=json.loads((R/split/'public'/f'{i:04d}.json').read_text());a=json.loads((R/split/'policies'/f'{i:04d}.json').read_text())
        x.append(p['profile']+p['velocity']);y.append(a['joint_risk'])
    raw=np.array(x)
    # With equal Gaussian row noise, the boundary lives in span(y,y²,y⁴).
    # Orthogonal residuals carry no likelihood information for either candidate family.
    z=FY/L;Q,_=np.linalg.qr(np.column_stack([z,z*z,z**4]))
    features=np.concatenate([raw[:,:9]@Q,raw[:,9:]],1)
    return features,np.array(y)
def distance(x,y):return np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0)
def fit_tree(x,y,depth,minleaf):
    result={'mean':y.mean(0).tolist()}
    if depth==0 or len(x)<2*minleaf:return result
    best=np.inf;split=None
    for j in range(x.shape[1]):
        for v in np.unique(np.quantile(x[:,j],np.linspace(.1,.9,17))):
            m=x[:,j]<=v
            if m.sum()<minleaf or (~m).sum()<minleaf:continue
            # Decision-focused split criterion: lowest sum of action risks.
            score=y[m].sum(0).min()+y[~m].sum(0).min()
            if score<best:best=score;split=(j,float(v),m)
    if split is None:return result
    j,v,m=split;result.update(j=j,v=v,left=fit_tree(x[m],y[m],depth-1,minleaf),right=fit_tree(x[~m],y[~m],depth-1,minleaf));return result
def tree_predict(tree,x):
    if 'j' not in tree:return np.tile(tree['mean'],(len(x),1))
    out=np.empty((len(x),4));m=x[:,tree['j']]<=tree['v']
    out[m]=tree_predict(tree['left'],x[m]);out[~m]=tree_predict(tree['right'],x[~m]);return out
def fit():
    start=time.time();x,y=data('train');v,vy=data('validation');mean=x.mean(0);sd=x.std(0);x=(x-mean)/sd;v=(v-mean)/sd
    best=None;records=[];d=distance(x,x);dv=distance(v,x)
    for width in [1.,2.,4.,8.]:
        K=np.exp(-d/(2*width**2));Kv=np.exp(-dv/(2*width**2))
        for ridge in [.001,.01,.1,1.]:
            alpha=np.linalg.solve(K+ridge*np.eye(len(x)),y-y.mean(0));pred=Kv@alpha+y.mean(0);choice=pred.argmin(1)
            risk=float(vy[np.arange(len(vy)),choice].mean());records.append(dict(kind='rbf',width=width,ridge=ridge,risk=risk))
            if best is None or risk<best[0]:best=(risk,width,ridge,alpha)
    treebest=None
    for depth in [2,3,4,6,8]:
        for leaf in [4,8,16,32]:
            tree=fit_tree(x,y,depth,leaf);choice=tree_predict(tree,v).argmin(1);risk=float(vy[np.arange(len(vy)),choice].mean());records.append(dict(kind='tree',depth=depth,minleaf=leaf,risk=risk))
            if treebest is None or risk<treebest[0]:treebest=(risk,tree,depth,leaf)
    fixed=int(y.mean(0).argmin());fixedrisk=float(vy[:,fixed].mean());bestkind=min([('rbf',best[0]),('tree',treebest[0]),('fixed',fixedrisk)],key=lambda z:z[1])[0]
    np.savez(R/'rules.npz',x=x,mean=mean,sd=sd,alpha=best[3],width=best[1],ymean=y.mean(0),fixed=fixed)
    write(R/'tree.json',treebest[1]);write(R/'rule_fit.json',dict(selected=bestkind,rbf_risk=best[0],tree_risk=treebest[0],fixed_risk=fixedrisk,rbf_width=best[1],rbf_ridge=best[2],tree_depth=treebest[2],tree_minleaf=treebest[3],candidates=records,seconds=time.time()-start))
    print('rules',bestkind,best[0],treebest[0],fixedrisk,flush=True)
def apply(split):
    r=np.load(R/'rules.npz');tree=json.loads((R/'tree.json').read_text());fit=json.loads((R/'rule_fit.json').read_text())
    x,_=data(split);x=(x-r['mean'])/r['sd'];rbf=(np.exp(-distance(x,r['x'])/(2*float(r['width'])**2))@r['alpha']+r['ymean']).argmin(1)
    tr=tree_predict(tree,x).argmin(1);fixed=np.full(len(x),int(r['fixed']));best={'rbf':rbf,'tree':tr,'fixed':fixed}[fit['selected']]
    for i in range(len(x)):
        path=R/split/'policies'/f'{i:04d}.json';p=json.loads(path.read_text())
        p['actions'].update(rule_rbf=int(rbf[i]),rule_tree=int(tr[i]),rule_fixed=int(fixed[i]),rule_best=int(best[i]));write(path,p)
if __name__=='__main__':
    import sys
    assert os.environ.get('SLURM_JOB_ID')
    if sys.argv[1]=='fit':fit()
    else:apply(sys.argv[2])
