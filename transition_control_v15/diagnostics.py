import os
os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
from common import *
from analyze import ci,contrast,by_action_loss,select_loss
from model import TASKS
def run():
    require();truth=np.load(R/'data/test/private.npz');loss=np.load(R/'world_losses.npz');acts=np.load(R/'decision_actions.npz');out={}
    for cell,cond in [('matched_nominal',0),('shifted_nominal',1)]:
        p=np.load(R/'policies'/f'{cell}.npz');small=np.load(R/'evaluations'/f'{cell}_16384.npz');large=np.load(R/'evaluations'/f'{cell}_65536.npz');entry=dict(subgroups={},decoder_swaps={},ess={})
        for k,j in [(2,4),(4,7)]:
            prob=small['pred'][:,0,0,j]/2**.5;var=prob*(1-prob);lin=p['native_scores'][:,1:,k].max(1);ratio=lin/np.maximum(var,1e-15)
            for label,mask in [('ambiguous',(prob>=.2)&(prob<=.8)),('low_linear_gain',(var>.05)&(ratio<.2))]:
                key=TASKS[k]+'/'+label;stats=dict(n=int(mask.sum()),linear_fraction_median=float(np.median(ratio[mask])) if mask.any() else None)
                if mask.sum()>=2:
                    stats['means']={m:float(loss[cell+'__'+m][mask,k].mean()) for m in ['native','pair','nested512','probe','query']}
                    stats['pair-native']=ci((loss[cell+'__pair'][:,k]-loss[cell+'__native'][:,k])[mask]);stats['nested512-native']=ci((loss[cell+'__nested512'][:,k]-loss[cell+'__native'][:,k])[mask])
                entry['subgroups'][key]=stats
        la=by_action_loss(large['pred'],truth['target'][:,cond])
        for m in ['native','pair','nested512','probe','query']:
            a=acts[cell+'__'+m];swap=select_loss(la,a);entry['decoder_swaps'][m]={task:dict(original_mean=float(loss[cell+'__'+m][:,k].mean()),large_decoder_mean=float(swap[:,k].mean()),small_minus_large=ci(loss[cell+'__'+m][:,k]-swap[:,k])) for k,task in enumerate(TASKS)}
            entry['ess'][m]={}
            for tag,ev in [('qmc16',small),('qmc64',large)]:
                e=ev['ess'][np.arange(len(a))[:,None],a,:]
                entry['ess'][m][tag]=dict(median_per_task=np.median(e,axis=(0,2)).tolist(),below20_fraction_per_task=(e<20).mean((0,2)).tolist())
        out[cell]=entry
    write(R/'secondary_diagnostics.json',out);print('Secondary diagnostics completed',flush=True)
if __name__=='__main__':run()
