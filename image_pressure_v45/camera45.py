"""Synthesize pre-contact side-view clips and estimate contact abscissa from pixels only.

The renderer owns the latent pose and shape. infer_x accepts a single uint8 video
tensor. The frozen router sees only an absolute coordinate in millimetres.
"""
from pathlib import Path
import sys

import numpy as np
import torch

WIDTH, HEIGHT = 96, 32
XGRID = (18.5,19.,19.5,30.5,31.,31.5)
HISTORICAL_D = (.00740230795167655, .00030392596767891847, .013675888628528466,
                -.0018944529104919646, -.0002800829062907069, .0035244049242156984)


def source_gate(x_mm):
    """Nearest predeclared historical cell, averaging over both shape families."""
    centers = x_mm.new_tensor(XGRID)
    risks = x_mm.new_tensor(HISTORICAL_D)
    return risks[(x_mm[...,None]-centers).abs().argmin(dim=-1)] < 0


def render_clip(theta, shape, variant, h_mm, offset_mm, seed, device='cpu', severity='primary'):
    """Orthographic approach; identical recordings under point, narrow and broad load.

    Camera generative parameters and object coordinates are intentionally not
    returned. A seed controls calibration, pose-independent nuisances, and shot noise.
    Rendering is offline; the calibrated fiducial strip and scene are in the image.
    """
    if severity != 'primary':
        raise ValueError('Use damage_clip for post-capture perturbations')
    # Reuse frozen contact profile from v44. This import is kept outside infer_x.
    proj = Path(__file__).resolve().parents[1]
    if str(proj/'contact_patch_v44') not in sys.path:
        sys.path.insert(0,str(proj/'contact_patch_v44'))
    import forecast44
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
        f = forecast44.shape_profile(x,shape,variant)[0]
        fnom = forecast44.shape_profile(np.asarray(h_mm),shape,variant)[0]
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


def infer_x(frames):
    """Track the tool and intersect its extrapolated path with the visible contour."""
    if frames.dtype != torch.uint8 or frames.ndim!=4 or frames.shape[1:]!=(3,HEIGHT,WIDTH):
        raise ValueError('Expected N×3×32×96 grayscale uint8 clips')
    b = frames.float()
    pix = torch.arange(WIDTH,device=b.device,dtype=b.dtype)
    rows = torch.arange(HEIGHT,device=b.device,dtype=b.dtype)
    rail = (175.-b[:,:,:3,:].mean(dim=(1,2))).clamp_min(0)
    lhs=rail[:,:WIDTH//2];rhs=rail[:,WIDTH//2:]
    xl=(lhs*pix[:WIDTH//2]).sum(-1)/lhs.sum(-1).clamp_min(1.)
    xr=(rhs*pix[WIDTH//2:]).sum(-1)/rhs.sum(-1).clamp_min(1.)
    dark = ((123.-b)/65.).clamp(0,1)*(rows[:,None]>=4)
    colmass = dark.sum(-2)
    mx = (colmass*pix).sum(-1)/colmass.sum(-1).clamp_min(1)
    rowmass = dark.sum(-1)
    # Last finite shank row plus a one-pixel allowance for antialiased apex.
    my = (rows*(rowmass>.30)).amax(-1)+1.
    valid3 = colmass[:,2].sum(-1)>.55*colmass[:,1].sum(-1)
    # A partially obscured shank may still have enough mass. Compare its
    # observed last centre with the constant-motion prediction as a consistency check.
    valid3 &= ((mx[:,2]-(2*mx[:,1]-mx[:,0])).abs()<1.3)
    xv = torch.where(valid3,mx[:,2],mx[:,1])
    yv = torch.where(valid3,my[:,2],my[:,1])
    dx = torch.where(valid3,(mx[:,2]-mx[:,0])*.5,mx[:,1]-mx[:,0])
    dy = torch.where(valid3,(my[:,2]-my[:,0])*.5,my[:,1]-my[:,0])
    slope = (dx/dy.clamp_min(1.)).clamp(-.25,1.1)
    # Ground is lighter than the tool. Its boundary is extracted from the
    # first (highest-clearance) frame, so the tip does not cover the contact site.
    ground = (b[:,0]>114.) & (b[:,0]<187.) & (rows[:,None]>5)
    interior = ground[:,:-2] & ground[:,1:-1] & ground[:,2:]
    edge = torch.where(interior,rows[:-2,None],float(HEIGHT)).amin(-2)
    pred = xv
    for _ in range(2):
        # Smooth adjacent columns to limit antialiasing/shot noise.
        idx = (pred.round().long()[:,None]+torch.tensor([-1,0,1],device=b.device)).clamp(0,WIDTH-1)
        ground_y=edge.gather(1,idx).median(-1).values-.5
        pred=xv+slope*(ground_y-yv).clamp(0,15)
    return 15.+20.*(pred-xl)/(xr-xl).clamp_min(5.)


def damage_clip(frames):
    """Post-capture final-frame loss, for sensitivity using exactly the same clips."""
    out=frames.clone()
    out[:,2,4:,:]=196
    return out
