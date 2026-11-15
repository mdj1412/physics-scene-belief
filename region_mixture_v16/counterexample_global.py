"""Specificity check: does equal-budget global mixture also pass the mechanism examples?"""
from validate import Fake
from model import *
from mixture import *
from common import *
require();eps=normal_plan();out={}
for name,label,w,good,bad in [('v15_aliasing',[0,0,1,1],[.25]*4,[-.5,-.5,.5,.5],[-4,4,-4,4]),('higher_order',[0,0,1,1,1],[.25,.25,.0625,.375,.0625],[-5,5,-10,0,10],[0]*5)]:
    b=Fake();b.white={1:t(good)[:,None],2:t(bad)[:,None],3:torch.zeros((len(w),1),device=DEV,dtype=DT)}
    c=compress(b,t(w),t(label)[:,None]*2**.5,True,global_mode=True);r=risks(c,eps)
    assert r[1]<r[2] and abs(float(r[0]-r[2]))<1e-10
    out[name]=dict(global_risks=r.cpu().tolist(),components=c['components'])
write(R/'counterexample_global.json',out);print(out)
