from model import *
from adaptive import *
from baselines import prepare,native,nested
from acquisition import lmmse
from common import *
class Fake:pass
def run():
    require();out={};ph=phases(81531);u,eps=plan(8192,81533)
    for name,lab,w,good,bad in [('aliasing',[0,0,1,1],[.25]*4,[-.5,-.5,.5,.5],[-4,4,-4,4]),('higher_order',[0,0,1,1,1],[.25,.25,.0625,.375,.0625],[-5,5,-10,0,10],[0]*5)]:
        b=Fake();b.white={1:t(good)[:,None],2:t(bad)[:,None],3:torch.zeros((len(w),1),device=DEV,dtype=DT)};ww=t(w);z=t(lab)[:,None]*2**.5
        ix=torch.searchsorted(ww.cumsum(0),u.contiguous()).clamp_max(len(w)-1);sample=dict(truth=z[ix],y={a:b.white[a][ix]+eps[:,:1] for a in [1,2,3]})
        a,r,m,h=select(b,ww,z,sample,ph);assert a==1 and abs(float(r[1])-.25)<1e-12 and r[0]<r[1]
        out[name]=dict(action=a,risks=r.tolist(),meta=m.tolist())
        for K in LEVELS:
            for phase in ph:
                c=compact(b,ww,z,K,phase);assert abs(float(c['w'][c['z'][:,0]>0].sum())-.5)<1e-12
        # Zero predictive variation must be recognized as a numerical tie.
        _,_,mm,_=select(b,ww,torch.zeros_like(z),dict(truth=torch.zeros_like(sample['truth']),y=sample['y']),ph)
        assert int(mm[2])==2 and int(mm[0])==8
    pub=np.load(R/'data/pilot/public.npz');b=prepare(Bank());o=Bank(seed=2026111202);x=t(pub['factual'][0,0]);ctx=t(pub['context'][0,0]);w=b.fit(x);z=targets(b.theta,ctx)[:,[4,7]]
    assert torch.allclose(native(b,w,z,[1,1]),lmmse(b,w,z,[1,1]),atol=1e-9,rtol=1e-8)
    for k in range(2):
        zz=z[:,k:k+1];mass=float(w[zz[:,0]>0].sum())
        for K in LEVELS:
            c=compact(b,w,zz,K,ph[0]);assert abs(float(c['w'][c['z'][:,0]>0].sum())-mass)<2e-10
    out['label_mass_ties_optimized_native_verified']=True;write(R/'validation.json',out);print('Validation passed',flush=True)
if __name__=='__main__':run()
