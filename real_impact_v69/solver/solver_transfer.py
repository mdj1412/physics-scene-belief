"""Sphere/plane calibration and paired whole-campaign transfer on actual measurements."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pybullet as b
import torch

HERE = Path(__file__).resolve().parent
RADIUS = .02135
THRESHOLDS = [.25, .50, 1.]
MUS = np.r_[np.arange(0, 1.0001, .025), 1.1, 1.25, 1.5, 2.0]
RATIOS = np.array([.03125, .0625, .125, .25, .5, .75, 1., 1.5, 2., 3., 4.])
MODELS = ('pybullet', 'compliant_coulomb', 'compliant_memory')


def tarray(x):
    return torch.as_tensor(x, dtype=torch.float64, device='cuda')


def scalar(x):
    return float(x.detach().cpu())


def normal(e, q):
    # Exact integration of the normal force over each interval of the bi-stiffness
    # oscillator, time scaled by loading frequency. Delta-v per unit incident vn.
    a = np.linspace(0, np.pi / 2, q + 1)
    loading = -np.diff(np.cos(a))
    unloading = e * np.diff(np.sin(a))
    dv = np.r_[loading, unloading]
    dt = np.repeat([np.pi / (2*q), np.pi/(2*q*e)], q)
    return dv, dt


def compliant(X, e, mu, ratio, q=80, inertia_ratio=.4):
    # mu and ratio are matching model rows. One state per candidate and real shot.
    # Return a COM + rim-velocity pair. Effective inverse mass is 1+1/(2/5)=3.5.
    # Solver time normalized by loading frequency, force normalized by mass.
    vx, vn, rim = X.T
    rim_factor = 1. / inertia_ratio
    inverse_effective_mass = 1. + rim_factor
    slip = (vx + rim).unsqueeze(0)
    normal_v = -vn.unsqueeze(0)
    mu, ratio = mu[:, None], ratio[:, None]
    elastic = torch.zeros_like(mu*slip)
    cum = torch.zeros_like(elastic)
    dv, dts = normal(e, q)
    for nj, dt in zip(dv, dts):
        cap = mu*normal_v*nj
        # For k_t>0, explicit tangential displacement and elastic return mapping.
        # For k_t=0, kinetic friction for sliding and static no-overshoot clamp.
        trial = -ratio*(elastic + slip*dt)*dt
        impulse = torch.where(ratio==0, -slip/inverse_effective_mass, trial)
        impulse = torch.clamp(impulse, min=-cap, max=cap)
        elastic = torch.where(ratio>0, -impulse/(ratio.clamp_min(1.e-15)*dt), 0.)
        cum = cum + impulse
        slip = slip + inverse_effective_mass*impulse
    return torch.stack((vx + cum, rim + rim_factor*cum), dim=-1)


def rigid(X, e, mus):
    vx, vn, rim = X.T
    cap = -mus[:, None] * (1 + e) * vn
    impulse = torch.clamp(-(vx + rim)/3.5, min=-cap, max=cap)
    return torch.stack((vx + impulse, rim + 2.5*impulse), dim=-1)


def bullet(X, e, mu, dt, inertia_ratio=.4):
    # Similarity units: R_engine=1, physical time unit=nominal ball radius [s].
    # Consequently the numeric engine COM speed & rim speed equal physical m/s.
    # Anticlockwise image spin corresponds to minus rotation about engine y (xz plane).
    b.connect(b.DIRECT)
    try:
        b.setGravity(0, 0, 0)
        b.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=100,
                                    restitutionVelocityThreshold=0, contactERP=0, contactSlop=0)
        plane = b.createMultiBody(baseMass=0, baseCollisionShapeIndex=b.createCollisionShape(b.GEOM_PLANE))
        ball = b.createMultiBody(baseMass=1, baseCollisionShapeIndex=b.createCollisionShape(b.GEOM_SPHERE, radius=1),
                                 basePosition=[0, 0, 1])
        for body, fric, rest in [(plane, 1., 1.), (ball, mu, e)]:
            b.changeDynamics(body,-1,lateralFriction=fric,restitution=rest,rollingFriction=0,
                             spinningFriction=0,linearDamping=0,angularDamping=0)
        if inertia_ratio != .4:
            b.changeDynamics(ball,-1,localInertiaDiagonal=[inertia_ratio]*3)
        pred = []
        for vx, vn, rim in X:
            # Ensures contact caches do not leak impulses between impact trials.
            b.resetBasePositionAndOrientation(ball,[0,0,3],[0,0,0,1])
            b.resetBaseVelocity(ball,[0,0,0],[0,0,0]); b.stepSimulation()
            b.resetBasePositionAndOrientation(ball,[0,0,1],[0,0,0,1])
            b.resetBaseVelocity(ball,[vx,0,vn],[0,-rim,0])
            for _ in range(4):
                b.stepSimulation()
            v, w = b.getBaseVelocity(ball)
            assert v[2] > 0, (vx,vn,rim,v)
            pred.append([v[0],-w[1],v[2]])
        return np.array(pred)
    finally:
        b.disconnect()


def probability(u_pred, errors, u_in):
    # Finite-sample Jeffreys smoothing of a scalar source residual CDF estimate.
    u_draw = u_pred[:,None] + errors[None,:]
    events = [(u_draw.abs()<=h) for h in THRESHOLDS] + [(u_in[:,None]*u_draw<0)]
    return [(event.sum(1) + .5) / (u_draw.shape[1]+1.) for event in events]


def campaign(source, target, writer):
    Xs, Ys = source[1][:,:3], source[1][:,3:]
    Xt, Yt = target[1][:,:3], target[1][:,3:]
    n = len(Xs)
    cal_np=(np.arange(n)//24)%4 == 1
    cal = torch.tensor(cal_np, device='cuda')
    X, Y = tarray(Xs), tarray(Ys)
    both = tarray(np.vstack([Xs, Xt]))
    fy = Y[~cal][:,[0,2]]
    # Normal data share no parameters with the tangential branch.
    fvn = X[~cal,1]
    e = scalar(-(fvn*Y[~cal,1]).sum()/fvn.square().sum())
    assert 0<e<=1
    mu_grid = tarray(MUS)
    rigid_risk = (rigid(X[~cal], e, mu_grid) - fy).square().mean((1,2))
    mu_rigid = MUS[int(rigid_risk.argmin().item())]
    mu_mem, ratio_mem = np.meshgrid(MUS,RATIOS,indexing='ij')
    param_mu, param_ratio = tarray(mu_mem.ravel()),tarray(ratio_mem.ravel())
    mem_train = compliant(X[~cal],e,param_mu,param_ratio)
    mem_risk = (mem_train-fy).square().mean((1,2))
    pos = int(mem_risk.argmin().item())
    best_mu,best_ratio = float(mu_mem.ravel()[pos]), float(ratio_mem.ravel()[pos])
    out = {'source':source[0], 'target':target[0], 'n_fit':int((~cal_np).sum()),
           'n_cal':int(cal_np.sum()),'n_test':len(Xt),'e':e,'mu_rigid':mu_rigid,
           'mu_memory':best_mu,'stiffness_ratio':best_ratio,'metrics':{}}
    comp = compliant(both,e,tarray([mu_rigid,best_mu]),tarray([0.,best_ratio]))
    finer = compliant(both,e,tarray([mu_rigid,best_mu]),tarray([0.,best_ratio]),160)
    # Matched rigid/point-compliant models have the same exact calibration objective.
    theoretical = rigid(both, e, tarray([mu_rigid]))[0]
    bullet_outputs = bullet(both.cpu().numpy(), e, mu_rigid, .00002)
    bullet_half = bullet(both.cpu().numpy(), e, mu_rigid, .00001)
    pb = tarray(bullet_outputs[:,:2])
    out['checks'] = {
        'pybullet_vs_rigid_max_m_s':scalar((pb-theoretical).abs().max()),
        'normal_restitution_max_m_s':float(abs(bullet_outputs[:,2]+both[:,1].cpu().numpy()*e).max()),
        'compliant_coulomb_vs_rigid_max_m_s':scalar((comp[0]-theoretical).abs().max()),
        'compliant_memory_80_vs_160_max_m_s':scalar((comp[1]-finer[1]).abs().max()),
        'compliant_memory_80_vs_160_mean_m_s':scalar((comp[1]-finer[1]).abs().mean()),
        'bullet_halfstep_max_m_s':float(abs(bullet_outputs-bullet_half).max())}
    # Save finer model as numeric sensitivity with same fitted parameters.
    all_y = tarray(np.vstack([Ys,Yt]))
    for method,pred in list(zip(MODELS,[pb,comp[0],comp[1]])) + [('compliant_memory_refined',finer[1])]:
        u = pred.sum(1)
        truth = all_y[:,0]+all_y[:,2]
        pr = probability(u, (truth[:n]-u[:n])[cal], both[:,0]+both[:,2])
        metric = {}
        for label, yp, yt in [('vx',pred[n:,0],all_y[n:,0]),('rim',pred[n:,1],all_y[n:,2]),('u',u[n:],truth[n:])]:
            metric['mae_'+label] = scalar((yp-yt).abs().mean())
            metric['rmse_'+label] = scalar((yp-yt).square().mean().sqrt())
        obs = [truth[n:].abs()<=h for h in THRESHOLDS] + [(both[n:,0]+both[n:,2])*truth[n:]<0]
        for ix, eventname in enumerate(['025','050','100','rev']):
            metric['brier_'+eventname] = scalar((pr[ix][n:]-obs[ix].double()).square().mean())
            metric['prevalence_'+eventname] = scalar(obs[ix].double().mean())
        out['metrics'][method] = metric
        rows = torch.column_stack([both,all_y,pred,u,torch.stack(pr,dim=1)]).cpu().numpy()
        for i,line in enumerate(rows):
            label = ('cal' if cal_np[i] else 'fit') if i<n else 'test'
            writer.writerow([source[0],target[0],source[0] if i<n else target[0],
                             i if i<n else i-n,label,method,*line.tolist()])
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=HERE.parent/'data69'/'ESM_2')
    args=parser.parse_args()
    assert torch.cuda.is_available()
    data={}
    for name in 'AB':
        path=args.data/f'data_campaign{name}.csv'
        raw=np.loadtxt(path,delimiter=',')
        assert raw.shape==({'A':330,'B':693}[name],6) and np.isfinite(raw).all()
        # Every CSV numerical component is R/s or rad/s. Multiplication by the same
        # radius converts rotation to rim speed and each translation to m/s.
        data[name]=(name,raw*RADIUS,hashlib.sha256(path.read_bytes()).hexdigest())
    start=perf_counter()
    with (HERE/'predictions.csv').open('w') as handle:
        writer=csv.writer(handle)
        writer.writerow(['source','target','campaign','row','split','method','vx_in','vn_in','rim_in',
                         'vx_out','vn_out','rim_out','vx_pred','rim_pred','u_pred',
                         'p025','p050','p100','p_rev'])
        studies=[campaign(data[a],data[b],writer) for a,b in [('A','B'),('B','A')]]
    report={'hostname':platform.node(),'slurm_job':os.environ.get('SLURM_JOB_ID'),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'bullet':b.getAPIVersion(),
            'data_hashes':{k:v[2] for k,v in data.items()},'radius_m':RADIUS,'studies':studies,
            'elapsed_seconds':perf_counter()-start}
    (HERE/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
