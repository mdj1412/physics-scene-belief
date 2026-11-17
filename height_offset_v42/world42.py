"""Controlled hidden placement error: nominal sensors, actual-height impact only."""
from pathlib import Path
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT/'estimated_projection_v41'))
import world41 as world

OFFSETS_MM=(-1.,-.5,-.25,-.125,0.,.125,.25,.5,1.)
ZERO=OFFSETS_MM.index(0.)
PANEL_START=256


def catalog_panel(panel):
    if not 0 <= int(panel)<12 or int(panel)!=panel:
        raise ValueError('Expected a v42 panel in 0..11')
    return PANEL_START+int(panel)


def direction_pairs(panel,family,height,n=32):
    return world.direction_pairs(catalog_panel(panel),family,height,n=n)


def offset_labels(theta,context,regime):
    """Return [offset,object] binary events at fixed target speed. No observations are generated."""
    physical_objects=world.t(theta)
    if physical_objects.ndim!=2 or physical_objects.shape[-1]!=6:
        raise ValueError('Expected an [objects,6] latent array')
    center=context[2]
    if center not in world.HEIGHTS or context[-1] not in range(PANEL_START,PANEL_START+12):
        raise ValueError('Expected a nominal v41 query height and a held-out v42 catalog')
    states=[world.impact(physical_objects,center+offset*.001,world.QUERY_INCOMING,context,regime)[1]
            for offset in OFFSETS_MM]
    return world.torch.stack(states).cpu().numpy()
