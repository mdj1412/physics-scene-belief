"""Fresh-environment posterior calibration under independent randomized sensor assignment."""
from pathlib import Path
import json
import os
import socket
import time
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import t as student
import physics38 as w

ROOT=Path(__file__).resolve().parent
PER_CELL_ALPHA=1024
SEED=384317229
TIERS=("strong","weak","null")
SIGNS=(1,-1)


def arr(x):
    return x.detach().cpu().numpy().astype(np.float64,copy=False)


def logs(y, m, sigma):
    c=m[0]
    a=(y-c)/sigma
    b=(m-c)/sigma
    return (a@b.T) - .5*(a*a).sum(-1)[:,None] - .5*(b*b).sum(-1)


def one_stratum(alpha,obj,sur,rng):
    laws=[(obj,sur,s) for s in SIGNS]
    cal=w.latent_support(alpha,laws[0],10)
    dep=w.latent_support(alpha,laws[0],20)
    n=PER_CELL_ALPHA;sigma=[arr(x) for x in w.sigmas()]
    calx=arr(w.factual_clean(cal,laws[0])); depx=arr(w.factual_clean(dep,laws[0]))
    early=np.array([[arr(w.impact(cal,0.,d*1.1,law,"strong")[0]) for d in SIGNS] for law in laws])
    last=np.array([[arr(w.impact(cal,.023,1.10,law,r)[0]) for r in TIERS] for law in laws])
    dep_mu={a:np.array([arr(w.deployment_clean(dep,law,"strong")[a]) for law in laws]) for a in (w.Q,w.P)}
    truths=np.array([[arr(w.impact(dep,.025,.425,law,r)[1]) for r in TIERS] for law in laws])
    # New independent draws for each environment, including the physical state and paid route.
    ci=rng.integers(len(cal),size=(n,4));di=rng.integers(len(dep),size=n)
    si=rng.integers(2,size=n);ri=rng.integers(3,size=n);first=rng.integers(2,size=n)
    direct=np.stack((first,1-first,first),axis=1)
    sensor=np.where(rng.random(n)<13/128,w.P,w.Q)
    cx=calx[ci]+rng.normal(size=(n,4,11))*sigma[0]
    dx=depx[di]+rng.normal(size=(n,11))*sigma[0]
    cy=early[si[:,None],direct,ci[:,:3]]+rng.normal(size=(n,3,2))*sigma[w.P]
    vy=last[si,ri,ci[:,3]]+rng.normal(size=(n,2))*sigma[w.P]
    sy={a:m[si,di]+rng.normal(size=m[si,di].shape)*sigma[a] for a,m in dep_mu.items()}
    label=truths[si,ri,di]
    pred=np.zeros((n,2))
    for begin in range(0,n,48):
        ii=np.arange(begin,min(begin+48,n))
        cal_log=[logs(cx[ii,j],calx,sigma[0]) for j in range(4)]
        dep_log=logs(dx[ii],depx,sigma[0])
        state_ll=np.empty((len(ii),2,2,3))
        state_event=np.empty((len(ii),2,3))
        for h in range(2):
            past=np.zeros(len(ii))
            for j in range(3):
                for d in range(2):
                    select=direct[ii,j]==d
                    if not select.any():continue
                    fresh=logs(cy[ii[select],j],early[h,d],sigma[w.P])
                    past[select]+=logsumexp(cal_log[j][select]+fresh,axis=1)
            dlog=np.empty(len(ii))
            for a in (w.Q,w.P):
                select=(sensor[ii]==a)
                if not select.any():continue
                weights=dep_log[select]+logs(sy[a][ii[select]],dep_mu[a][h],sigma[a])
                norm=logsumexp(weights,axis=1)
                state_event[select,h]=np.exp(weights-norm[:,None])@truths[h].T
                dlog[select]=norm
            state_ll[:,0,h,:]=(past+dlog)[:,None]
            for g in range(3):
                near=logs(vy[ii],last[h,g],sigma[w.P])
                state_ll[:,1,h,g]=past+dlog+logsumexp(cal_log[3]+near,axis=1)
        weights=softmax(state_ll.reshape(len(ii),2,6),axis=-1)
        pred[ii]=(weights*state_event.reshape(len(ii),1,6)).sum(axis=2)
    p3,p4=pred.T
    square=(p4-p3)**2
    oracle_realized=(p3-label)**2-(p4-label)**2
    return np.stack((p3-label,p4-label,p4-p3,oracle_realized-square,square,oracle_realized),axis=1),sensor


def main():
    t0=time.monotonic()
    w.torch.set_num_threads(2)
    metrics=("forecast3_minus_y","forecast4_minus_y","forecast4_minus_forecast3",
             "realized_minus_model_gain","ideal","realized")
    arrs=[];actions=[]
    for io,obj in enumerate(w.OBJECTS):
        for il,sur in enumerate(w.SURFACES):
            for ia,alpha in enumerate(w.ALPHAS):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,io,il,ia]))
                z,act=one_stratum(alpha,obj,sur,rng);arrs.append(z);actions.append(act)
            print(obj,sur,round(time.monotonic()-t0,1),flush=True)
    data=np.stack(arrs);a=np.stack(actions)
    mu=data.mean((0,1))
    variances=data.var(1,ddof=1)/(PER_CELL_ALPHA*len(arrs)**2)
    var=variances.sum(0)
    se=var**.5
    df=var**2 / (variances**2/(PER_CELL_ALPHA-1)).sum(0)
    intervals={}
    for j,name in enumerate(metrics):
        level=student.ppf(.975,df[j]); crit=student.ppf(1-.05/8,df[j])
        intervals[name]=dict(mean=mu[j],pointwise95=[mu[j]-level*se[j],mu[j]+level*se[j]],
                             se=se[j],df=df[j],p_two_sided=2*student.sf(abs(mu[j]/se[j]),df[j]))
        if j<4:
            intervals[name]["simultaneous95_bonferroni_4"]=[mu[j]-crit*se[j],mu[j]+crit*se[j]]
    byaction={str(x):dict(n=int((a==x).sum()),mean=list(data[a==x].mean(0))) for x in (w.Q,w.P)}
    result=dict(n=int(len(arrs)*PER_CELL_ALPHA),per_cell_alpha=PER_CELL_ALPHA,seed=SEED,labels_per_env=1,
                prefix3_first_direction="uniform",prior="uniform sign and local tier",sensor="independent random P with probability 13/128",
                metrics=metrics,results=intervals,descriptive_by_action=byaction,seconds=time.monotonic()-t0,
                host=socket.gethostname(),job=os.environ.get("SLURM_JOB_ID"),device=str(w.t(0).device))
    file=ROOT/"stress_oracle38.json"
    temp=file.with_suffix(".tmp")
    temp.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist())+"\n")
    temp.replace(file)

if __name__=="__main__":
    main()
