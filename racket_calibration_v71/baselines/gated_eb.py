"""Equal empirical Bayes hyperprior construction for two release-sign affine expert heads."""
import csv
import json
import numpy as np
import torch
from compare import HERE,V70,R,GROUPS,BUDGETS,old,xy,matrix


def feat(a,side):
    return torch.cat((a*(~side)[...,None],a*side[...,None]),-1)


def fit(data,groups):
    ids=sorted(c for g in groups for c in GROUPS[g])
    x,y=xy(data,'train',ids)
    _,model=old.empirical(x,y,x[:1])
    m=old.tensor(model['mean']);sd=old.tensor(model['std']); gate=old.tensor(model['logistic'])
    beta=old.tensor(model['expert_minus']+model['expert_plus'])
    d=[];noise=[]
    for g in groups:
        gx,gy=xy(data,'train',GROUPS[g]);ga=matrix((gx-m)/sd)
        f=feat(ga,gy[:,0]-gy[:,2]>0)
        theta=torch.linalg.lstsq(f,gy).solution
        d.append((theta-beta).flatten())
        err=gy-f@theta;noise.append(err.T@err/(len(err)-8))
    d=torch.stack(d); cov=d.T@d/len(groups)
    P=torch.linalg.inv(.85*cov+.15*torch.diag(cov.diag().clamp(min=.0001)))
    W=torch.linalg.inv(torch.stack(noise).mean(0)+torch.eye(3,device='cuda')*.0001)
    return m,sd,beta,gate,P,W


def run(prior,x,y,idx,tx,lambdas):
    m,sd,beta,gate,P,W=prior
    a=matrix((x[idx]-m)/sd); yy=y[idx]
    side=yy[...,0]-yy[...,2]>0
    switched=(a@gate>0)!=side
    cf=feat(a,side);res=yy-cf@beta
    ta=matrix((tx-m)/sd);ts=ta@gate>0
    for k in BUDGETS:
        flip=switched[:,:k].sum(1)>k/2
        tf=feat(ta.expand(len(idx),-1,-1),ts[None,:]^flip[:,None])
        if k==0:
            for h in lambdas:yield k,h,tf@beta
            continue
        b=cf[:,:k]
        gram=torch.einsum('bij,kl->bikjl',b.mT@b,W).reshape(-1,24,24)
        rhs=(b.mT@res[:,:k]@W).reshape(-1,24,1)
        for h in lambdas:
            update=torch.linalg.solve(gram+h*P,rhs).reshape(-1,8,3)
            yield k,h,tf@(beta+update)


def main():
    assert torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    data,_=old.load(V70/'data'/'upstream')
    primary=json.loads((V70/'solver'/'metrics.json').read_text())['folds']
    params={};result={};fields=['rubber','draw','budget','method','velocity','spin','sign_accuracy']
    source_prior={}
    def prior(g):
        key=tuple(sorted(g))
        if key not in source_prior:source_prior[key]=fit(data,key)
        return source_prior[key]
    with (HERE/'gated_eb_draw_metrics.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(fields)
        for target,held in GROUPS.items():
            source=[g for g in GROUPS if g!=target]
            vals={.1:0.,1.:0.,10.:0.}
            for pseudo in source:
                p=prior([g for g in source if g!=pseudo])
                x,y=xy(data,'train',GROUPS[pseudo])
                rng=np.random.default_rng(712000+min(GROUPS[pseudo]))
                perm=rng.permutation(len(x));val=perm[:192];pool=perm[192:]
                idx=np.stack([rng.choice(pool,32,replace=False) for _ in range(24)])
                for k,h,yp in run(p,x,y,idx,x[val],vals):
                    if k:vals[h]+=float((yp-y[val]).square().mean())/(6*5)
            h=min(vals,key=vals.get)
            x,y=xy(data,'train',held);tx,ty=xy(data,'test',held)
            idx=np.load(HERE/'draw_indices.npz')[target]
            params[target]=dict(selected=h,inner=vals);result[target]={}
            for k,_,yp in run(prior(source),x,y,idx,tx,[h]):
                d=yp-ty
                v=d[...,:2].norm(dim=-1).mean(1); s=d[...,2].abs().mean(1)/R
                acc=((yp[...,0]-yp[...,2]>0)==(ty[:,0]-ty[:,2]>0)).double().mean(1)
                if k==0:
                    ref=primary[target]['metrics']['two_regime']['test']
                    assert abs(float(v[0])-ref['velocity_mae'])<1e-8 and abs(float(s[0])-ref['spin_mae'])<1e-8
                result[target][k]=[float(v.mean()),float(s.mean())]
                for j,row in enumerate(torch.stack((v,s,acc),1).cpu().numpy()):
                    wr.writerow([target,j,k,'gated_expert_eb',*row])
    macro={k:np.mean([q[k] for q in result.values()],axis=0).tolist() for k in BUDGETS}
    (HERE/'gated_eb.json').write_text(json.dumps(dict(folds=result,macro=macro,fits=params),indent=2)+'\n')
    print(macro)
if __name__=='__main__':main()
