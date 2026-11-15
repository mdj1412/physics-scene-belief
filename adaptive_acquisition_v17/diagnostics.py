"""Post-outcome explanatory audit; no policy changes or new primary hypotheses."""
import os
os.environ['BELIEF_DEVICE']='cpu'
import torch
from model import *
from common import *
from adaptive import compact,phases
from run_policy import EIDX,ETASKS
require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');p=np.load(R/'policies/matched_nominal.npz');aud=np.load(R/'audits/matched_nominal.npz');L=np.load(R/'world_losses.npz');b=Bank();phase=phases(81531);out={};mass=[];collapsed=[]
for i in range(512):
    w=b.fit(t(pub['factual'][i,0]));z=targets(b.theta,t(pub['context'][i,0]))[:,EIDX];mr=[];cr=[]
    for k in range(2):
        zz=z[:,k:k+1];prob=float(w[zz[:,0]>0].sum());mr.append(prob);K=int(p['adaptive_global_meta'][i,k,0])
        comps=[compact(b,w,zz,K,phase[j],True) for j in [0,1]]
        cr.append(all(len(torch.unique(c['z']))==1 for c in comps))
    mass.append(mr);collapsed.append(cr)
mass=np.array(mass);collapsed=np.array(collapsed)
for k,task in enumerate(ETASKS):
    gtie=p['adaptive_global_meta'][:,k,2]==2;both=(mass[:,k]>1e-10)&(mass[:,k]<1-1e-10);lost=gtie&both&collapsed[:,k]
    item=dict(global_risk_ties=int(gtie.sum()),ties_with_both_factual_regions=int((gtie&both).sum()),ties_with_both_replicas_single_label=int(lost.sum()),minority_mass_in_collapsed_ties_quantiles=np.quantile(np.minimum(mass[lost,k],1-mass[lost,k]),[0,.5,1]).tolist() if lost.any() else None)
    item['groups']={}
    for m in ['adaptive','adaptive_global']:
        for reason,label in [(0,'gap_separated'),(1,'cap_unresolved'),(2,'risk_tie')]:
            mask=p[m+'_meta'][:,k,2]==reason
            item['groups'][m+'/'+label]=dict(n=int(mask.sum()),seed2_agreement=float((aud[m+'_seed2'][mask,k]==p[m][mask,k]).mean()) if mask.any() else None,loss=float(L['matched_nominal__'+m][mask,k].mean()) if mask.any() else None,loss_minus_fixed128=float((L['matched_nominal__'+m][mask,k]-L['matched_nominal__fixed128'][mask,k]).mean()) if mask.any() else None)
    out[task]=item
write(R/'diagnostics.json',out);print(json.dumps(out,indent=2))
