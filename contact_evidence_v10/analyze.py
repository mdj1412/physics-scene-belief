"""Server-side post-freeze analysis, world-cluster bootstrap and figures."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import QUERIES,ACTIONS,basis,design,FY

R=Path('results');O=R/'analysis';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(x,alpha=.05):
    rng=np.random.default_rng(3911);boot=np.concatenate([x[rng.integers(len(x),size=(1000,len(x)))].mean(1) for _ in range(30)])
    return dict(mean=float(x.mean()),ci=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),world_sd=float(x.std(ddof=1)))
def main():
    assert os.environ.get('SLURM_JOB_ID'),'Analyze on compute node'
    assert (R/'complete.json').exists()
    evaldir=R/'test/evaluation_refined' if (R/'refinement.json').exists() else R/'test/evaluation'
    freeze=read(R/'test/freeze.json');methods={};native={};all_loss=[];twins=[];equals=[];num=[];times={};choices={};us={};sides=[];phys=[];audits=[]
    for name,checksum in freeze['policies'].items():
        p=R/'test/policies'/name;assert sha(p)==checksum
        pol=read(p);ev=np.load(evaldir/name.replace('.json','.npz'));ll=ev['losses'];all_loss.append(ll)
        for key,choice in pol['choices'].items():
            arr=ll[np.asarray(choice),:,np.arange(6)].T
            methods.setdefault(key,[]).append(arr.mean())
            choices.setdefault(key,[]).append(choice)
            if key=='coarse_cf8192':own=ev['coarse_losses'][np.asarray(choice),:,np.arange(6)].T
            elif key=='plugin_certainty':own=ev['plugin_losses'][np.asarray(choice),:,np.arange(6)].T
            else:own=arr
            native.setdefault(key,[]).append(own.mean())
        methods.setdefault('random_three_expected',[]).append(ll[1:].mean())
        native.setdefault('boundary_plugin_factual',[]).append(ev['plugin_losses'][0].mean())
        native.setdefault('coarse_factual',[]).append(ev['coarse_losses'][0].mean())
        for k in ['visibility_query_rule','boundary_cf32768']:
            c=np.asarray(pol['choices'][k]);native.setdefault('boundary_plugin_'+k,[]).append(ev['plugin_losses'][c,:,np.arange(6)].mean())
        for k,v in pol['utilities'].items():us.setdefault(k,[]).append(v)
        for k,v in pol['seconds'].items():times.setdefault(k,[]).append(v)
        twins.append(np.sqrt(((ev['target']-ev['twin_target'])**2).sum(1)))
        equals.append(float(ev['factual_equal_max']));num.append(float(ev['quadrature_prediction_maxdiff']))
        sides.append(read(R/'test/public'/name)['visible_side'])
        truth=np.load(R/'test/private'/name.replace('.json','.npz'));phys.append(truth['g'])
        if pol['audit']:audits.append((int(name[:3]),pol['audit']))
    loss=np.stack(all_loss) # world, action, repeat, query
    methods={k:np.array(v) for k,v in methods.items()};native={k:np.array(v) for k,v in native.items()}
    choices={k:np.array(v) for k,v in choices.items()};us={k:np.array(v) for k,v in us.items()}
    mainkey='boundary_cf32768';primary={}
    for key in ['none','query_type_rule','query_value_rule','boundary_quad64']:
        primary[key]=interval(methods[mainkey]-methods[key],.0125)
    desc={k:interval(methods[mainkey]-v) for k,v in methods.items() if k!=mainkey}
    twin=np.stack(twins);qutil=us['boundary_quad64'];choice=choices[mainkey]
    expected_regret=qutil.max(2)-np.take_along_axis(qutil,choice[:,:,None],2)[:,:,0]
    counts={k:np.bincount(v.ravel(),minlength=4).tolist() for k,v in choices.items()}
    agreement={k:float((v==choices['boundary_quad64']).mean()) for k,v in choices.items()}
    rank=[]
    for qi,y in enumerate(QUERIES):
        subset=[]
        for s in [0,1]:
            mask=np.array(sides)==s;subset.append(np.bincount(choice[mask,qi],minlength=4).tolist())
        rank.append(dict(query_mm=float(y*1000),counts_by_visible_side=subset))
    qdiff=[];halves=[]
    for i,a in audits:
        q96=np.asarray(a['quad96_same_outer']);q64=qutil[i]
        qdiff.append(abs(q96-q64).max());halves.extend((np.asarray(a['cf32_h512']).argmax(1)==choice[i]).tolist())
    # World-level regret relative to all-action, repeat-averaged retrospective oracle.
    # This uses test outcomes and is ONLY a diagnostic lower bound, not a baseline policy.
    retro=loss.mean(2).min(1).mean(1)
    hidden_gaps=np.concatenate([twin[i,[3,5] if s==0 else [0,2]] for i,s in enumerate(sides)])
    costs={}
    for cost in [0,.0001,.001,.01]:
        u=us[mainkey].copy();u[:,:,1:]-=cost;c=u.argmax(2)
        realized=np.stack([loss[i,c[i],:,np.arange(6)].mean(1) for i in range(len(loss))])
        costs[str(cost)]=dict(mse=float(realized.mean()),fraction_observe=float((c!=0).mean()),net_loss=float(realized.mean()+cost*(c!=0).mean()))
    summary=dict(worlds=len(loss),repeats=16,queries=6,decoder_directory=str(evaldir),means={k:float(v.mean()) for k,v in methods.items()},
                 native_means={k:float(v.mean()) for k,v in native.items()},primary=primary,descriptive=desc,
                 times={k:dict(mean=float(np.mean(v)),median=float(np.median(v))) for k,v in times.items()},
                 choices=counts,quad_agreement=agreement,ranking_by_query=rank,
                 quadrature_utility_regret=dict(mean=float(expected_regret.mean()),max=float(expected_regret.max())),
                 numerical=dict(decoder_order_audit_max=max(num),utility_quad64_96_max=max(qdiff),h256_h512_agreement=float(np.mean(halves)),
                                audited_worlds=len(audits)),
                 twins=dict(factual_image_mean_max=max(equals),velocity_gap_median=float(np.median(twin)),hidden_offcenter_gap_median=float(np.median(hidden_gaps)),hidden_offcenter_fraction_gap_gt_01=float((hidden_gaps>.1).mean()),
                            offcenter_gap_median=float(np.median(twin[:,[0,2,3,5]])),
                            offcenter_fraction_gap_gt_01=float((twin[:,[0,2,3,5]]>.1).mean()),
                            center_gap_max=float(twin[:,[1,4]].max()),per_query_mean=twin.mean(0).tolist()),
                 rules=read(R/'rules.json'),retrospective_noise_averaged_oracle=float(retro.mean()),
                 cf_gain_world_fraction=float((methods[mainkey]<methods['none']).mean()),cost_sensitivity=costs,
                 native_contrasts={k:interval(v-methods[mainkey]) for k,v in native.items() if k.startswith('boundary_plugin') or k=='coarse_cf8192'},time=time.time())
    (O/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(O/'world_metrics.npz',**methods)
    labels=['none','query_type_rule','query_value_rule','visibility_query_rule','boundary_standard8192','boundary_cf8192','boundary_cf32768','boundary_quad64','coarse_cf8192']
    fig,ax=plt.subplots(figsize=(10,5));vals=[methods[k].mean() for k in labels]
    ax.barh(labels,vals,color=['#7b8794']*4+['#e6a23c','#376ea6','#24613b','#00a087','#a579a6']);ax.invert_yaxis()
    ax.set_xlabel('Actual decision MSE (m²/s²), common boundary decoder');ax.set_title('Independent worlds: N=128, 16 noise repeats per world')
    fig.tight_layout();fig.savefig(O/'decision_loss.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for qi in [0,2,3,5]:axes[0].scatter(np.arange(len(twin)),twin[:,qi],s=7,alpha=.45,label=f'{QUERIES[qi]*1000:.0f} mm')
    axes[0].set(xlabel='World',ylabel='Twin response L2 gap (m/s)',title='Same factual likelihood, different interventions');axes[0].legend(ncol=2)
    axes[1].scatter(methods['none'],methods[mainkey],s=15,alpha=.7)
    lim=max(methods['none'].max(),methods[mainkey].max());axes[1].plot([0,lim],[0,lim],'k--',lw=1)
    axes[1].set(xlabel='No-observation MSE',ylabel='Boundary CF32768 MSE',title='Paired world-level loss');fig.tight_layout();fig.savefig(O/'ambiguity_and_gain.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for s,ax in enumerate(axes):
        mask=np.array(sides)==s;arr=np.array([np.bincount(choice[mask,q],minlength=4)/mask.sum() for q in range(6)]).T
        im=ax.imshow(arr,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(6),[f'{q*1000:.0f}' for q in QUERIES]);ax.set_yticks(range(4),ACTIONS)
        ax.set(xlabel='Query impact y (mm)',title=f'Factual visible lobe: {"left" if s==0 else "right"}')
    fig.subplots_adjust(left=.12,right=.82,bottom=.16,wspace=.2)
    cax=fig.add_axes([.85,.16,.025,.72]);fig.colorbar(im,cax=cax,label='Selection fraction')
    fig.savefig(O/'selection.png',dpi=180);plt.close(fig)
    # Illustrative pair selected by index before outcome analysis, not maximal-gap search.
    truth=np.load(R/'test/private/000.npz');pub=read(R/'test/public/000.json');ys=np.linspace(-.044,.044,500)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(design(ys)@truth['g']*1000,ys*1000,label='World')
    axes[0].plot(design(ys)@truth['twin']*1000,ys*1000,'--',label='Twin');axes[0].axhspan(-44,-6,alpha=.12,color='gray') if pub['visible_side']==1 else axes[0].axhspan(6,44,alpha=.12,color='gray')
    axes[0].set(xlabel='Boundary x offset (mm)',ylabel='Impact y (mm)',title='World 000; shaded region is occluded');axes[0].legend()
    for tag,col in [('target','#2166ac'),('twin_target','#b2182b')]:
        v=truth[tag];axes[1].quiver(np.zeros(6),QUERIES*1000,v[:,0],v[:,1],color=col,angles='uv',scale=5,label=tag)
    axes[1].set(xlim=(-1,1),ylim=(-44,44),ylabel='Impact y (mm)',title='Counterfactual velocity directions');axes[1].legend();fig.tight_layout();fig.savefig(O/'twin_example.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
