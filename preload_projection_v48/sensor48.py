"""One scalar per weak contact from displacement across a 3-mm local gauge.

All nuisance draws and per-specimen accounts are simulator-private. Only scalar
displacement in mm and task-commanded metadata can enter the deployment router.
"""
from functools import lru_cache
import numpy as np
from world48 import HEIGHTS, STATES, STATE_WEIGHTS, camera_shape, loaded_shape

PRELOAD_N, REF_N = .4, 2.0
GAUGE_MM = 3.
TRANSFER = .60


@lru_cache(maxsize=128)
def _nuisance(seed, query_h, station):
    state_rng = np.random.default_rng(np.random.SeedSequence([seed,480713]))
    shared = state_rng.random((4,128))
    other = state_rng.choice(STATES, size=(4,128), p=STATE_WEIGHTS)
    rng = np.random.default_rng(np.random.SeedSequence([seed,480714,station]))
    gain = rng.uniform(.90,1.10,(4,128,1))*rng.uniform(.93,1.07)
    bias = rng.uniform(-.012,.012)+rng.normal(0.,.007,(4,128,1))
    noise = rng.normal(0.,.028,(4,128,4))
    return shared, other, gain, bias, noise


@lru_cache(maxsize=128)
def gauge_mm(family, site_mm, state):
    at = np.array([site_mm - GAUGE_MM/2, site_mm + GAUGE_MM/2])
    before = camera_shape(at,family)[0]
    after = loaded_shape(at,site_mm,family,state)[0]
    dz = (PRELOAD_N/REF_N)*(after-before)
    return float(dz[1]-dz[0])


def states_at(pop,query_height_mm,state,station_mm,transfer=TRANSFER):
    if query_height_mm not in HEIGHTS or station_mm not in HEIGHTS or state not in STATES or not 0<=transfer<=1:
        raise ValueError("invalid fixture")
    shared, other, *_ = _nuisance(int(pop["seed"]),query_height_mm,station_mm)
    if station_mm == query_height_mm:
        return np.full((4,128),state)
    return np.where(shared < transfer, state, other)


def readings(pop, family, query_height_mm, signed_state, station_mm, transfer=TRANSFER):
    state = states_at(pop,query_height_mm,signed_state,station_mm,transfer)
    _,_,gain,bias,noise = _nuisance(int(pop["seed"]),query_height_mm,station_mm)
    values = np.array([gauge_mm(family,station_mm,i) for i in (-1,0,1)])
    return (gain*values[state[...,None]+1]+bias+noise).astype(np.float32)
