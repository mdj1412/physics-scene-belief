"""Posterior expected Brier gain from three conventional and one near-query reading."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
BETA = .8289760825845313
FY = np.linspace(-.010, .010, 9)
MODES = ('strong', 'local_weak', 'local_null', 'global_null')
FAMILIES = ('rounded_fine','rounded_coarse','ridged_fine','ridged_coarse')
SIGNS = (-1, 1)

def load_artifact(path=None):
    with np.load(path or ROOT/'gate36_frozen.npz', allow_pickle=False) as saved:
        return {k:saved[k] for k in saved.files}

def observations(x,y):
    grid=FY/.010
    # Orthogonal projections of all nine equally noisy profile readings.
    linear=grid/np.sqrt(grid@grid)
    quadratic=grid**2-np.mean(grid**2)
    quadratic/=np.sqrt(quadratic@quadratic)
    proj=np.stack((np.ones(9)/3,linear,quadratic),axis=-1)
    return np.concatenate((x[...,:9]@proj/.0016,x[...,9:]/.095,y/.027),axis=-1)

def log_density(artifact, obs, action):
    # Mixture means/covariances marginalize independent latent objects per probe.
    means=artifact['means'][action]  # model, component, feature
    residual=obs[:,None,None,:]-means[None,...]
    energy=np.einsum('eski,skij,eskj->esk',residual,artifact['precision'][action],residual,optimize=False)
    return logsumexp(artifact['log_norm'][action][None,...]-.5*energy,axis=2)

def posterior(log_likelihood):
    return np.exp(log_likelihood-logsumexp(log_likelihood,axis=-1,keepdims=True))

def expected_gain(artifact, weights, z):
    grid=artifact['zgrid']
    coordinate=np.clip((z-grid[0])/(grid[1]-grid[0]),0,len(grid)-1)
    lo=coordinate.astype(int)
    hi=np.minimum(lo+1,len(grid)-1)
    frac=coordinate-lo
    values=artifact['gain'][:,lo].T*(1-frac[:,None])+artifact['gain'][:,hi].T*frac[:,None]
    return np.sum(weights*values,axis=1)

def predict(artifact,paid_x,paid_y,paid_dirs,matched_y,schedule):
    x=np.asarray(paid_x,float)
    y=np.asarray(paid_y,float)
    d=np.asarray(paid_dirs,int)
    n=len(x)
    if x.shape!=(n,4,11) or y.shape!=(n,3,2) or d.shape!=(n,3) or not np.isin(d,[-1,1]).all():
        raise ValueError('Expected four factual rows, three conventional responses and three signed actions')
    if schedule not in ('same','alternating'):
        raise ValueError('Unsupported motion schedule')
    if not np.all(d==d[:,:1]*([1,1,1] if schedule=='same' else [1,-1,1])):
        raise ValueError('Action labels disagree with the paid schedule')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite sensor reading')
    # Equal source priors over four regimes, four families and two orientations.
    likelihood=np.zeros((n,len(artifact['sign'])))
    for j in range(3):
        obs=observations(x[:,j],y[:,j])
        for action,sign in enumerate(SIGNS):
            selected=d[:,j]==sign
            if selected.any():
                likelihood[selected]+=log_density(artifact,obs[selected],action)
    weights=posterior(likelihood)
    # Degree of belief in far-field direction; global-null has zero signed value.
    z=weights@artifact['far_sign']
    score3=expected_gain(artifact,weights,z)
    out=dict(z=z,score3=score3,accept3=score3>0,
             near_strength3=weights@artifact['near_strength'])
    if matched_y is None:
        return out
    matched=np.asarray(matched_y,float)
    if matched.shape!=(n,2) or not np.isfinite(matched).all():
        raise ValueError('Expected one noisy near-query velocity per environment')
    weights4=posterior(likelihood+log_density(artifact,observations(x[:,3],matched),2))
    score4=expected_gain(artifact,weights4,z)
    out.update(score4=score4,accept4=score4>0,near_strength4=weights4@artifact['near_strength'])
    return out
