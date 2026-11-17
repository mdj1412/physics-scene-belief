"""Fold-frozen sign alignment and residual expert adaptation to a paid calibration prefix."""
import torch

def features(x, ep):
    def t(k):
        return x.new_tensor(ep[k])
    return torch.cat((torch.ones_like(x[..., :1]), (x-t("mean"))/t("std")),dim=-1)

def experts(a, branch, ep):
    minus=a @ a.new_tensor(ep["expert_minus"])
    plus=a @ a.new_tensor(ep["expert_plus"])
    return torch.where(branch[...,None],plus,minus)

def residual_design(a, branch):
    signed=2*branch.to(a.dtype)-1
    return torch.cat((a, a*signed.unsqueeze(-1)), dim=-1)

def fit_residual(a,y,route,ep,penalty):
    f=residual_design(a,route)
    residual=y-experts(a,route,ep)
    # Branch corrections shrink more strongly when a prefix samples only one branch.
    ridge=torch.diag(f.new_tensor([penalty]*4+[4*penalty]*4))
    return torch.linalg.solve(f.mT@f+ridge, f.mT @ residual)

def adapt(ep, cal_x, cal_y, test_x, penalty):
    a=features(cal_x,ep)
    t=features(test_x,ep)
    beta=t.new_tensor(ep["logistic"])
    d,k,_=cal_x.shape
    route=t@beta>0
    orig_a=a@beta>0
    true_a=cal_y[...,0]-cal_y[...,2]>0
    mismatch=(orig_a!=true_a).sum(1)
    flip=2*mismatch>k
    new_route=route.unsqueeze(0)^flip[:,None]
    t=t.unsqueeze(0).expand(d,-1,-1)
    orig=route.unsqueeze(0).expand(d,-1)
    f=experts(t,orig,ep)
    g=experts(t,new_route,ep)
    if k==0:
        return dict.fromkeys(["frozen_gate","gate_only","expert_only","gate_expert","routed_expert_only"],f),dict(flip=flip,mismatch=mismatch)
    # These two arms have exactly the same learned coefficients and differ only in routing.
    coef=fit_residual(a,cal_y,true_a,ep,penalty)
    coef_routed=fit_residual(a,cal_y,orig_a,ep,penalty)
    return {"frozen_gate":f,"gate_only":g,
            "expert_only":f+residual_design(t,orig)@coef,
            "gate_expert":g+residual_design(t,new_route)@coef,
            "routed_expert_only":f+residual_design(t,orig)@coef_routed},dict(flip=flip,mismatch=mismatch)
