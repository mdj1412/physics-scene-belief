"""Secondary held-rubber comparisons with a fixed blade or balanced source exposure."""
import json
import torch
from benchmark import HERE,GROUPS,MUS,tensor,load,design,rotate,grid_predictions,records,empirical,mechanical,ols

def weighted_rigid(x,y,tx,w):
    mus=tensor(MUS); best=(float('inf'),)
    for deg in range(36):
        ni=rotate(x,deg)[1]
        if (ni>=0).any():continue
        nf=rotate(y,deg)[1]
        e=float((-(w*ni)@nf/((w*ni)@ni)).clamp(0,1))
        outs=grid_predictions(x,deg,mus,e)
        risks=(((outs[:,:,[0,2]]-y[:,[0,2]])**2).sum(-1)*w).mean(1)/2
        i=int(risks.argmin())
        best=min(best,(float(risks[i]),deg,MUS[i],e))
    risk,deg,mu,e=best
    return grid_predictions(tx,deg,tensor([mu]),e)[0],dict(deg=deg,mu=mu,e=e)

def weighted_emp(x,y,tx,w):
    wt=w/w.sum()
    m=wt@x; ss=(wt@((x-m)**2)).sqrt()
    a=design((x-m)/ss); ta=design((tx-m)/ss)
    side=y[:,0]-y[:,2]>0
    beta=torch.zeros(4,device=x.device,dtype=x.dtype)
    pen=torch.diag(tensor([0,1,1,1]))
    for _ in range(35):
        prob=torch.sigmoid((a@beta).clamp(-35,35))
        step=torch.linalg.solve(a.T@((w*prob*(1-prob))[:,None]*a)+pen,a.T@(w*(prob-side.double()))+pen@beta)
        beta-=step
        if float(step.norm())<1e-9:break
    sw=w.sqrt()[:,None]; aw,yw=a*sw,y*sw
    experts=[ols(aw[side==b],yw[side==b]) for b in (False,True)]
    return {'single_affine':ta@ols(aw,yw),'two_regime':torch.where((ta@beta>0)[:,None],ta@experts[1],ta@experts[0])}

def main():
    sets,_=load(HERE.parent/'data'/'upstream')
    out={}
    for setting in ['same_allround_blade','balanced_source_products']:
        eligible=list(range(5,11)) if setting=='same_allround_blade' else list(range(1,11))
        folds={}
        for name,allheld in GROUPS.items():
            held=[i for i in eligible if i in allheld]
            if not held:continue
            source=[i for i in eligible if i not in allheld]
            x=torch.cat([sets['train',i][0] for i in source]);y=torch.cat([sets['train',i][1] for i in source])
            tx=torch.cat([sets['test',i][0] for i in held]);ty=torch.cat([sets['test',i][1] for i in held])
            if setting=='same_allround_blade':
                pred,_=empirical(x,y,tx); phys,pp=mechanical(x,y,tx)
                pred['inclined']=phys['inclined']; pp=pp['inclined']
            else:
                def weight(i):
                    members=next(v for v in GROUPS.values() if i in v)
                    total=sum(len(sets['train',j][0]) for j in members)
                    return len(x)/((len(GROUPS)-1)*total)
                w=torch.cat([torch.ones_like(sets['train',i][0][:,0])*weight(i) for i in source])
                assert abs(float(w.sum())-len(x)) <1.e-8
                pred=weighted_emp(x,y,tx,w);pred['inclined'],pp=weighted_rigid(x,y,tx,w)
            folds[name]=dict(source=source,held=held,train=len(x),test=len(tx),inclined=pp,
                             metrics={m:records(tx,ty,yp)[1] for m,yp in pred.items()})
        out[setting]=folds
    (HERE/'diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
