"""Choose residual complexity by source family holdout, fit T, lock simple alternatives."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit, logit

import model43 as model
import inference43 as infer

ROOT = Path(__file__).resolve().parent
WIDTHS = (8, 32)
EPOCHS = 45
SEED = 43031771


def save(name, d):
    def serial(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.integer,np.floating)): return v.item()
        raise TypeError(type(v))
    (ROOT/name).write_text(json.dumps(d, indent=2, allow_nan=False, default=serial)+'\n')


def fit_residual(raw, y, width):
    """Optimize sample-weighted Brier on future events, not error on parameter labels."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1.e-8] = 1.
    feats = torch.tensor(np.clip((raw-mean)/scale,-8.,8.),dtype=torch.float32)
    truth = torch.tensor(y,dtype=torch.float32)
    base = torch.tensor(raw[:,0],dtype=torch.float32)
    h = torch.nn.Linear(feats.shape[1],width)
    out = torch.nn.Linear(width,1,bias=False)
    skip = torch.nn.Linear(feats.shape[1],1)
    torch.nn.init.zeros_(skip.weight); torch.nn.init.zeros_(skip.bias); torch.nn.init.zeros_(out.weight)
    optim = torch.optim.AdamW([*h.parameters(),*out.parameters(),*skip.parameters()],lr=.006,weight_decay=.005)
    for epoch in range(EPOCHS):
        for idx in np.array_split(rng.permutation(len(raw)), max(1,(len(raw)+1023)//1024)):
            x = feats[idx]
            logodds = base[idx] + skip(x)[:,0] + out(torch.tanh(h(x)))[:,0]
            loss = (torch.sigmoid(logodds)-truth[idx]).square().mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*h.parameters(),*out.parameters(),*skip.parameters()],1.)
            optim.step()
    def array(tensor): return tensor.detach().numpy().astype(float)
    return dict(mean=mean,scale=scale,skip=array(skip.weight)[0],bias=float(skip.bias.detach().item()),
                b1=array(h.bias),w1=array(h.weight).T,w2=array(out.weight)[0],width=width)


def brier(p, y):
    # y is the exact average of binary events over predeclared strata.
    return float((np.square(p)-2*p*y+y).mean())


def main():
    torch.set_num_threads(2)
    data = np.load(ROOT/'source_matrix43.npz')
    x = data['feature']; panels = data['panel']; shapes = data['shape']
    target = data['label']     # [row, friction regime, nominal/shifted]
    nominal = target[:,:,0].mean(1)
    shifted = target[:,:,1].mean(1)
    fit_y = .5*nominal + .5*shifted
    train = panels < 8
    calibrate = (panels == 8)|(panels == 9)
    choose = panels >= 10
    table = []
    for width in WIDTHS:
        splits = []
        for held_family in (0,1):
            fit = train & (shapes != held_family)
            score = train & (shapes == held_family)
            residual = fit_residual(x[fit],fit_y[fit],width)
            p = model.robust_from_features(residual,x[score])
            splits.append(brier(p,fit_y[score]))
        table.append(splits)
    best = WIDTHS[np.argmin(np.mean(table,axis=1))]
    artifact = fit_residual(x[train],fit_y[train],best)
    # Same compute/predictor as augmented source training; use nominal labels only.
    erm = fit_residual(x[train],nominal[train],best)
    aug_p = model.robust_from_features(artifact,x)
    erm_p = model.robust_from_features(erm,x)
    # Features carry clipped base log odds to avoid extreme saturation in residual training.
    original = expit(x[:,0])
    preds = dict(original=original, robust=aug_p, privileged=data['privileged'])
    temps, bydec, simple = {},{},{}
    signs = data['cue']; size = (choose.sum(),3)
    for name,p in preds.items():
        log_odds = logit(np.clip(p,1.e-5,1-1.e-5))
        def grad(scale):
            return np.mean((expit(scale*log_odds[calibrate])-shifted[calibrate])*log_odds[calibrate])
        inverse_temp = brentq(grad,.005,100.) if grad(.005)*grad(100.) < 0 else (.005 if grad(.005)>0 else 100.)
        temp = 1/inverse_temp
        temps[name] = float(temp)
        q = lambda val: np.broadcast_to(val[:,None],size)
        pc = q(p[choose])
        zc = q(signs[choose])
        compare = {'zero':pc,
                   'temperature':q(expit(log_odds[choose]/temp)),
                   'fixed':infer.r.v38.apply(pc,zc*.325,infer.r.v38.BETA),
                   'optimizer':infer.r.v38.apply(pc,zc*data['weight'][choose],infer.r.v38.BETA)}
        bydec[name] = {k:dict(nominal=brier(v,target[choose,:,0]),shifted=brier(v,target[choose,:,1]))
                       for k,v in compare.items()}
        simple[name]=min(('zero','temperature','fixed'),key=lambda k:bydec[name][k]['shifted'])
    artifact.update(temperature=temps,source_best=simple,seed=SEED,feature_count=x.shape[-1],
                    source_protocol='v42 neutral-prior reconstructed features, source panels 0-7 fit; 8-9 T; 10-11 source-best; nominal/nonzero mixture .5/.5 for robustness, T fits nonzero NLL')
    save('frozen_model43.json',artifact)
    audit = dict(panel_split={'train':list(range(8)),'temp':[8,9],'choose':[10,11]},
                 rows=len(x), rows_per_panel=len(x)//12, families=['flare','roll'], validation_widths=WIDTHS,
                 leave_family_out_brier=table,chosen_width=best, temperatures=temps,
                 source_best=simple, simple_selection_scores=bydec,
                 source_shifted_robust_minus_erm=brier(aug_p[choose],shifted[choose])-brier(erm_p[choose],shifted[choose]),
                 source_nominal_robust_minus_erm=brier(aug_p[choose],nominal[choose])-brier(erm_p[choose],nominal[choose]),
                 probability_clip=1.e-5,
                 saturated_fraction_original=float(np.mean(np.abs(x[:,0])>11.5128)),
                 unique_prototype_panel_population_draws=int(np.unique(np.c_[panels,data['object_id']],axis=0).shape[0]),
                 feature_dim=x.shape[1],epochs=EPOCHS,beta=infer.r.v38.BETA)
    save('source_validation43.json',audit)
    print(json.dumps({k:audit[k] for k in ('rows','chosen_width','temperatures','source_best','source_shifted_robust_minus_erm','saturated_fraction_original')},default=lambda x:np.asarray(x).tolist()),flush=True)


if __name__=='__main__':main()
