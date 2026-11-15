"""Paired query-law stress test; consumes raw artifacts, independent NumPy physics."""
from pathlib import Path
import argparse, json
import numpy as np

BASE = Path(__file__).resolve().parent
ARMS = ('same','alternating')
CELLS = ('sinusoid_atan','sinusoid_algebraic','quartic_atan','quartic_algebraic')
S1 = np.array([abs(x['signed_amplitude']) for x in json.loads((BASE.parent/'directional_prefix_v33/physical33.json').read_text())['rows']])

def labels(theta, cell, sign, amplitude):
    obj, law = cell.split('_')
    _, _, slope, curve, mu, e = np.moveaxis(theta,-1,0)
    local = slope + curve + (-.045*np.sin(1.) if obj=='sinusoid' else .04)
    norm = np.sqrt(1.+local**2)
    vt = (.425+local)/norm
    jn = (1.+e)*np.maximum(0., (1.-.425*local)/norm)
    response = ((2/np.pi)*np.arctan(vt/.4) if law=='atan' else vt/np.sqrt(.3**2+vt**2))
    return abs(vt)<=mu*(1.+sign*amplitude*response)*jn

def moments(p0, p1, label, original_label):
    # p0,p1 = env x world x paired sensor realization; labels = env x world (or mean over surface conditions).
    ell = np.asarray(label,dtype=float); d = p1 - p0
    penalty = np.mean(d*d)
    alignment = 2*np.mean((d*p0).mean(-1)*(-1.)+d.mean(-1)*ell)
    orig = np.mean((p0*p0).mean(-1) + ell*(1.-2.*p0.mean(-1)))
    corr = np.mean((p1*p1).mean(-1) + ell*(1.-2.*p1.mean(-1)))
    assert abs(corr-orig-(penalty-alignment)) < 1e-10
    return np.array([orig,corr,corr-orig,penalty,alignment,np.mean(np.where(original_label,1-ell,ell))])

def summarize(x):
    # cell x block x mirrored orientation x schedule x outcome. Same initial signs within a strata.
    x = x.mean(2)
    rng = np.random.default_rng(351034)
    bd = np.empty((12000,2))
    for start in range(2):
        xs = x[:,start::2,:,2]
        ix = rng.integers(xs.shape[1],size=(12000,4,xs.shape[1]))
        draw = xs[np.arange(4)[None,:,None],ix].mean((1,2))
        if start==0: bd[:]=draw/2
        else: bd+=draw/2
    return {a: dict(zip(('original','corrected','delta','movement_squared','twice_alignment','changed_label_fraction'),x[:,:,i].mean((0,1)).tolist()), pointwise95=np.quantile(bd[:,i],[.025,.975]).tolist()) for i,a in enumerate(ARMS)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('rawdir',type=Path); args=ap.parse_args()
    grid = [(s, np.full((4,8,2,2,6),np.nan)) for s in ('full','half','zero','reverse','S1_abs_oriented')]
    for celli,cell in enumerate(CELLS):
        amp = .21 if cell.endswith('_atan') else .16
        for b in range(8):
            for si,sgn in enumerate([1,-1]):
                p0,p1,theta,y = read_case(args.rawdir,cell,b,sgn)
                assert np.array_equal(labels(theta,cell,sgn,amp),y), (cell,b,sgn,'failed exact label replay')
                for name,arr in grid:
                    if name=='S1_abs_oriented':
                        yy=np.mean([labels(theta,cell,sgn,a) for a in S1],axis=0)
                    else:
                        yy=labels(theta,cell,sgn,{'full':1,'half':.5,'zero':0,'reverse':-1}[name]*amp)
                    for s in range(2): arr[celli,b,si,s]=moments(p0[s],p1[s],yy,y)
    assert all(np.isfinite(x).all() for _,x in grid)
    out={n:summarize(x) for n,x in grid}
    (BASE/'transfer_stress35.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:[round(z['delta'],6) for z in v.values()] for k,v in out.items()}))

# read_case is defined below so that file naming has no effect on the numerical implementation.
def read_case(rawdir,cell,block,sign):
    folder=rawdir/(cell+('_plus' if sign==1 else '_minus'))/f'block_{block:03d}'
    rows=[]
    for sc in ARMS:
        with np.load(folder/(sc+'.npz'),allow_pickle=False) as f:
            rows.append(tuple(np.asarray(f[k],dtype=np.float64) for k in ('p0','p1','theta','labels')))
    np.testing.assert_array_equal(rows[0][2],rows[1][2])
    np.testing.assert_array_equal(rows[0][3],rows[1][3])
    return np.stack([x[0] for x in rows]),np.stack([x[1] for x in rows]),rows[0][2],rows[0][3].astype(bool)

if __name__=='__main__': main()
