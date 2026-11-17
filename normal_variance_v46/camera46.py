"""Shared raster optics, a pixel-only normal-variance estimator, and frozen lookups."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"image_pressure_v45"))
from camera45 import HEIGHT, WIDTH, XGRID, source_gate, infer_x

def shape_function(shape):
    if shape in ("fold","ripple"):
        sys.path.insert(0,str(ROOT.parent/"contact_patch_v44"))
        from forecast44 import shape_profile
    else:
        from geometry46 import shape_profile
    return shape_profile

def render_clip(theta, shape, variant, h_mm, offset_mm, seed, device='cpu', severity='primary'):
    """Orthographic approach; identical recordings under point, narrow and broad load.

    Camera generative parameters and object coordinates are intentionally not
    returned. A seed controls calibration, pose-independent nuisances, and shot noise.
    Rendering is offline; the calibrated fiducial strip and scene are in the image.
    """
    if severity != 'primary':
        raise ValueError('Use damage_clip for post-capture perturbations')
    profile = shape_function(shape)
    th = torch.as_tensor(theta).detach().cpu().numpy().reshape(-1,6)
    n = len(th)
    rng = np.random.default_rng(seed)
    def uniform(low, high):
        return rng.uniform(low, high, (n,1)).astype(np.float32)
    def noise(shape_, std):
        return rng.normal(0,std,shape_).astype(np.float32)
    abs_x = float(h_mm) + np.broadcast_to(np.asarray(torch.as_tensor(offset_mm).cpu(),dtype=np.float32),(n,)).reshape(n,1)
    pxmm = uniform(3.2,4.0)
    xzero = 47.5 + uniform(-2.3,2.3)
    yref = uniform(20.8,23.4)
    xpix = np.arange(WIDTH,dtype=np.float32)[None,:]
    ypix = np.arange(HEIGHT,dtype=np.float32)[None,:,None]
    xworld = 25. + (xpix-xzero)/pxmm
    def raised(x):
        # Vertical image coordinate increases into the material. Reference
        # each scene to nominal height, not to the hidden signed placement error.
        base = th[:,2:3]*(x-h_mm) + th[:,3:4]*(x*x-h_mm*h_mm)/50.
        f = profile(x,shape,variant)[0]
        fnom = profile(np.asarray(h_mm),shape,variant)[0]
        return (base+f-fnom).astype(np.float32)
    surface_y = yref + pxmm*raised(xworld)
    site_y = yref + pxmm*raised(abs_x)
    # Constant velocity and equal spacing between frames. All three frames
    # precede impact (positive vertical clearance, 0.425 horizontal / 1 vertical).
    gap3 = uniform(.52,1.55)
    frame_step = uniform(.55,1.1)
    jitter = noise((n,3,2),.07)
    bright = uniform(216,242)
    surf = uniform(144,173)
    shaft = uniform(32,61)
    width = uniform(2.6,4.2)
    beam = uniform(-7.,7.)
    ground = np.clip(ypix - surface_y[:,None,:]+.5,0.,1.)
    base = bright[:,None,:]+(surf-bright)[:,None,:]*ground
    base = base+beam[:,None,:]*(xpix[:,None,:]/WIDTH-.5)
    # Two printed fiducials on a fixed rail, at absolute abscissae 15, 35 mm.
    left = xzero-10.*pxmm
    right = xzero+10.*pxmm
    fid = np.maximum(np.clip(1.6-np.abs(xpix-left),0,1),np.clip(1.6-np.abs(xpix-right),0,1))
    occl = uniform(0,1)<.14
    occl_x = uniform(1,94)
    occl_w = uniform(4,10)
    clips = np.empty((n,3,HEIGHT,WIDTH),np.uint8)
    for t in range(3):
        clearance = gap3+(2-t)*frame_step
        x = xzero + pxmm*(abs_x-.425*clearance-25.) + jitter[:,t,:1]
        y = site_y-pxmm*clearance+jitter[:,t,1:2]
        dy = y[:,None,:]-ypix
        # Symmetric rigid shank ends in a triangular point. It has a finite
        # pre-contact extent but does not preload the future pressure footprint.
        tip_width = np.minimum(.75*np.maximum(dy,0.),width[:,None,:])
        tool = np.clip(tip_width+.45-np.abs(xpix[:,None,:]-x[:,None,:]),0,1)
        tool *= np.clip(dy+.5,0,1)*(ypix>=4)
        frame = base+(shaft[:,None,:]-base)*tool
        if t==2:
            blocked = occl[:,None,:] & (np.abs(xpix-occl_x)<occl_w)[:,None,:] & (ypix>=4)
            frame = np.where(blocked,186.,frame)
        frame[:,:3,:] = 226.-183.*fid[:,None,:]
        frame += noise((n,HEIGHT,WIDTH),4.)
        clips[:,t] = np.rint(np.clip(frame,0.,255.)).astype(np.uint8)
    return torch.from_numpy(clips).to(device)


def infer_normal_variance(frames, x_mm):
    """Population variance of seven finite-baseline surface normals, in radians squared."""
    # Air and material levels include unknown local exposure. Estimate the material
    # at the bottom of each column and the air from unobstructed pixels near the top.
    frame=frames[:,0].float()
    cols=torch.arange(WIDTH,device=frame.device,dtype=frame.dtype)
    rows=torch.arange(HEIGHT,device=frame.device,dtype=frame.dtype)
    rail=(175.-frames[:,:,:3,:].float().mean(dim=(1,2))).clamp_min(0)
    mid=WIDTH//2
    xl=(rail[:,:mid]*cols[:mid]).sum(-1)/rail[:,:mid].sum(-1).clamp_min(1.)
    xr=(rail[:,mid:]*cols[mid:]).sum(-1)/rail[:,mid:].sum(-1).clamp_min(1.)
    pix_per_mm=(xr-xl).clamp_min(5)/20.
    air=frame[:,4:11,:].amax(dim=1)
    # Interpolate the air level across shank columns via a wide median, then
    # compensate the upward bias of the maximum of seven noisy intensities.
    air=torch.nn.functional.pad(air[:,None,:],(8,8),mode="replicate").unfold(2,17,1).median(-1).values[:,0]-5.
    surface=frame[:,-4:,:].mean(1)
    midpoint=.5*(air+surface)
    # In the lower part, sustained gray pixels indicate material. The first frame
    # has a gap: a dark shaft alone cannot be accepted as the material run.
    plausible=(frame<midpoint[:,None,:]+5) & (frame>110.) & (rows[None,:,None]>=8)
    hit=plausible[:,:-2]&plausible[:,1:-1]&plausible[:,2:]
    idx=torch.where(hit,rows[:-2,None],float(HEIGHT-2)).amin(1).long()
    delta=torch.arange(-2,3,device=frame.device)[None,:,None]
    ii=(idx[:,None,:]+delta).clamp(0,HEIGHT-1)
    obs=frame.gather(1,ii)
    frac=((air[:,None,:]-obs)/(air-surface).clamp_min(20.)[:,None,:]).clamp(0,1)
    edge=idx+2.5-frac.sum(1)
    dx=torch.arange(-2,2.25,.5,device=frame.device,dtype=frame.dtype)
    xp=xl[:,None]+(x_mm[:,None]-15.+dx)*pix_per_mm[:,None]
    ix=xp.floor().long().clamp(0,WIDTH-2)
    weight=(xp-ix).clamp(0,1)
    y=(1.-weight)*edge.gather(1,ix)+weight*edge.gather(1,ix+1)
    angle=torch.atan((y[:,2:]-y[:,:-2])/pix_per_mm[:,None])
    return (angle-angle.mean(-1,keepdim=True)).square().mean(-1)

def true_stat(theta, shape, variant, true_x):
    """Evaluation reference uses same seven secant chords, centered at the true site."""
    dx=np.arange(-2.,2.25,.5)
    h=np.asarray(true_x).reshape(-1,1)+dx
    th=np.asarray(theta).reshape(-1,6)
    f=shape_function(shape)(h,shape,variant)[0]
    z=th[:,2:3]*h+th[:,3:4]*h*h/50.+f
    return np.arctan(z[:,2:]-z[:,:-2]).var(-1)

def load_lookup(device="cpu", kind="visual"):
    raw=json.loads((ROOT/"router46.json").read_text())[kind]
    return {k:torch.as_tensor(raw[k],device=device,dtype=torch.float64)
            for k in ("x_grid","thresholds","risk_x","risk_joint")}

def gate46(x_mm, variance, lookup):
    idx=(x_mm[...,None]-lookup["x_grid"]).abs().argmin(-1)
    if variance is None:
        return lookup["risk_x"][idx] < 0
    stratum=(variance[...,None]>lookup["thresholds"][idx]).sum(dim=-1).long()
    return lookup["risk_joint"][idx,stratum]<0
