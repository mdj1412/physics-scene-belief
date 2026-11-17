"""Pair likelihood for the randomized sensor reset and independent camera intervention."""
from itertools import product

import numpy as np

Q = .29
MOMENTS = (.145, .0925)
LOCAL_LATCH = .26582278481012656
LOCAL_ERROR = 2 * .02 * .98
CAMERA_ERROR = 2 * .03 * .97
SOURCE_LOG_CUTOFF = np.log(8.417543970276335e-5 / 4.175119807325477e-5)
ARMS = ('r1c1', 'r0c1', 'r1c0', 'r0c0')
METHODS = tuple('map_' + arm for arm in ARMS) + tuple('risk_' + arm for arm in ARMS) + ('gate','fixed','oracle')


def encode(reference, impact, reset_sensor, camera=True):
    """Encode signed channels in chronological order; first local channel is the assigned target."""
    r, i = np.asarray(reference), np.asarray(impact)
    if r.shape != i.shape or r.shape[-2:] != (8,3):
        raise ValueError('Eight chronological slots and three instrument channels expected.')
    targets = np.asarray(reset_sensor)
    if targets.shape != r.shape[:-2] + (4,) or not np.isin(targets, [0,1]).all():
        raise ValueError('One randomized instrument ID per session is required.')
    active = (r != 0) & (i != 0)
    if not np.array_equal(active[...,0], active[...,1]):
        raise ValueError('Both contact sensors share the physical activity mask.')
    changes = r != i
    which = np.repeat(targets.astype(np.intp), 2, axis=-1)
    target = np.take_along_axis(changes[...,:2], which[...,None], axis=-1)[...,0]
    other = np.take_along_axis(changes[...,:2], (1-which)[...,None], axis=-1)[...,0]
    k = np.where(camera & active[...,2], 1+changes[...,2].astype(np.uint8), 0)
    symbol = np.where(active[...,0], 1+target.astype(np.uint8)+2*other.astype(np.uint8)+4*k, 0)
    return (13*symbol[...,::2] + symbol[...,1::2]).astype(np.uint8)


def binary_pair(moment):
    return np.array([[1-2*Q+moment, Q-moment],[Q-moment,moment]])


def upstream_joint(cause, regime):
    """P(true mismatch pair, common upstream apparent pair)."""
    moment = MOMENTS[regime]
    if cause == 0:
        return np.diag(binary_pair(moment).ravel())
    c = (moment-Q**2) / (Q*(1-Q))
    true = binary_pair(Q**2).ravel()
    return (1-c)*np.diag(true) + c*np.outer(true,[1-Q,0,0,Q])


def local_pair(u, intercept):
    # The target local instrument is cleaned just before the second read.
    u1,u2=divmod(u,2)
    out = np.zeros((2,2))
    for phantom,weight in [(None,1-LOCAL_LATCH),(0,LOCAL_LATCH*(1-Q)),(1,LOCAL_LATCH*Q)]:
        a = u1 if phantom is None else phantom
        b = u2 if phantom is None or intercept else phantom
        out[a,b] += weight
    e = np.array([[1-LOCAL_ERROR,LOCAL_ERROR],[LOCAL_ERROR,1-LOCAL_ERROR]])
    return e.T @ out @ e


def pair_table(reset, cause, regime):
    # Channels are instrument0 mismatch, instrument1 mismatch, camera mismatch.
    obs = np.zeros((8,8))
    channels = np.array([[(s>>i)&1 for i in range(3)] for s in range(8)])
    ix = channels[:,None,:]
    jx = channels[None,:,:]
    dist = upstream_joint(cause,regime)
    for t,u in product(range(4),repeat=2):
        if dist[t,u] == 0:
            continue
        t1,t2=divmod(t,2)
        cam = np.where(ix[...,2]==t1,1-CAMERA_ERROR,CAMERA_ERROR)*np.where(jx[...,2]==t2,1-CAMERA_ERROR,CAMERA_ERROR)
        sensor0 = local_pair(u,reset)[ix[...,0],jx[...,0]]
        sensor1 = local_pair(u,False)[ix[...,1],jx[...,1]]
        obs += dist[t,u]*cam*sensor0*sensor1
    assert np.isclose(obs.sum(),1)
    # Rows: inactive, active but no camera, active with camera 0/1.
    projection = np.vstack([np.ones((1,8)), np.tile(np.eye(4),(1,2)), np.eye(8)])
    complete = projection @ obs @ projection.T
    masked = np.array([0]+[1+(i%4) for i in range(12)])
    blind = complete[np.ix_(masked,masked)]
    return np.log(np.stack([blind.ravel(),complete.ravel()]))


TABLE = np.empty((2,2,2,2,169))
for _r,_k,_z in product(range(2),repeat=3):
    TABLE[_r,:,_k,_z] = pair_table(_r,_k,_z)


def log_odds(codes, reset, camera):
    """Log odds for A against B, and for physical against shared artifact, over the full pilot."""
    x = np.asarray(codes)
    if x.shape[-1:] != (4,):
        raise ValueError('Four observed two-object sessions required.')
    scores = TABLE[int(reset),int(camera)][:,:,x].sum(axis=-1)
    z = np.logaddexp(scores[0],scores[1])
    k = np.logaddexp(scores[:,0],scores[:,1])
    return z[0]-z[1], k[0]-k[1]


def route(codes, method):
    if method in ('gate','fixed'):
        return np.full(np.asarray(codes).shape[:-1], method=='fixed', dtype=bool)
    if method not in METHODS[:8]:
        raise ValueError('Expected a deployable MAP or source-risk selector.')
    loss, arm = method.split('_')
    odds,_ = log_odds(codes,int(arm[1]),int(arm[3]))
    return odds > (1.e-12 if loss=='map' else SOURCE_LOG_CUTOFF)
