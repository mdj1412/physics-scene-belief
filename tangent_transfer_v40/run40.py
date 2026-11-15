"""Evaluate the two frozen v38 policies on a paired panel of contour and law trials."""
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'probe_transfer_v39'))
import run39 as historical
import physics40 as world

previous = historical.previous
continuous = historical.continuous38
CONDITIONS = tuple(itertools.product(world.FAMILIES, world.CONTOURS, world.SURFACES))
CELLS = tuple('_'.join(condition) for condition in CONDITIONS)
SIGNS = (1, -1)
REGIMES = ('strong', 'weak', 'null')
METHODS = ('fixed', 'optimizer')
CATALOGS = 12
BLOCKS = 2
BASE = 400_000_000
OUT = ROOT / 'results'


def timing_stratum(cell, panel, block):
    if block == 0 and panel in (cell, (cell + 6) % CATALOGS):
        return (cell + 3 * int(panel != cell)) % 6
    return -1


def block_path(cell, catalog, index):
    return OUT / CELLS[cell] / f'catalog_{catalog:02d}' / f'block_{index:02d}.npz'


def already_complete(path, fingerprint):
    metadata = path.with_suffix('.json')
    if not metadata.exists():
        return False
    info = json.loads(metadata.read_text())
    if info['fingerprint'] != fingerprint or previous.digest(path) != info['sha256']:
        raise RuntimeError(f'Saved block failed provenance check: {path}')
    return True


def evidence_hash(records, factual):
    """Only reuse acquisitions when every array read by the acquisition agrees."""
    digest = hashlib.sha256()
    for x, ids, y, dirs in records:
        for obs in (x, ids[:3], y[:3], dirs[:3]):
            digest.update(np.ascontiguousarray(obs).tobytes())
    digest.update(np.ascontiguousarray(factual).tobytes())
    return digest.digest()


def block(detector, artifact, fingerprint, c, catalog, index, decoders, alpha, pair, seed, cache):
    filename = block_path(c, catalog, index)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if already_complete(filename, fingerprint):
        return
    start = (-1, 1)[index]
    time_ms = np.empty((0, 2))
    order = np.empty((0, 2), dtype=np.int8)
    stratum = timing_stratum(c, catalog, index)
    rows = {name: [] for name in (
        'risk', 'weights', 'p_base', 'labels', 'actions', 'z', 'cal_x',
        'ids', 'first3_y', 'fourth_y', 'fourth_stick', 'dirs')}
    for si, sign in enumerate(SIGNS):
        law = (*CONDITIONS[c], sign, catalog)
        risks, weights, labels, fourth_y, fourth_stick = [], [], [], [], []
        first = None
        for ri, regime in enumerate(REGIMES):
            cal = [world.calibration(seed+100+e, a, law, regime, start) for e, a in enumerate(alpha)]
            records = [item[0] for item in cal]
            x, y, dirs = previous.observations(records)
            deployment = [world.deployment(seed+200+e, a, law, regime, repeats=32)
                          for e, a in enumerate(alpha)]
            xx, target, _, obs = zip(*deployment)
            xx, target = np.stack(xx), np.stack(target)
            if first is None:
                key = evidence_hash(records, xx)
                if key not in cache:
                    cache[key] = previous.prefix_actions(detector, records, pair, xx)
                actions = cache[key]
                base = previous.paired_decode(decoders, xx, actions, obs)
                first = (x, y[:,:3], xx, obs)
                rows['actions'].append(actions)
                rows['p_base'].append(base)
                rows['cal_x'].append(x)
                rows['first3_y'].append(y[:,:3])
                rows['dirs'].append(dirs[:,:3])
                rows['ids'].append(np.stack([record[1] for record in records]))
            else:
                for actual, reference in zip((x, y[:,:3], xx), first[:3]):
                    np.testing.assert_array_equal(actual, reference)
                for actual, reference in zip(obs, first[3]):
                    for sensor in (previous.Q, previous.P):
                        np.testing.assert_array_equal(actual[sensor], reference[sensor])
            z, w = continuous.source_optimum(artifact, x, y[:,:3], dirs[:,:3], y[:,3])
            z_fixed, w_fixed = previous.policy(artifact, records, 'fixed', True)
            np.testing.assert_allclose(z, z_fixed, rtol=0, atol=1.e-12)
            np.testing.assert_array_equal(w_fixed, np.full(8, previous.SHRINK))
            if ri == 0:
                rows['z'].append(z)
            else:
                np.testing.assert_array_equal(z, rows['z'][-1])
            paired_loss = []
            for multiplier in (w_fixed, w):
                p = previous.apply(base, (z * multiplier)[:,None,None], previous.BETA)
                paired_loss.append(np.square(p - target[:,:,None]).mean((1, 2)))
            risks.append(np.stack(paired_loss))
            weights.append(w)
            labels.append(target)
            fourth_y.append(y[:,3])
            fourth_stick.append(np.asarray([item[3] for item in cal]))
            if stratum == si*len(REGIMES) + ri:
                time_ms, order = historical.timing(detector, artifact, records, pair, decoders,
                                                   xx, obs, actions, base, z, w, seed+699+c)
        for name, values in [('risk', risks), ('weights', weights), ('labels', labels),
                             ('fourth_y', fourth_y), ('fourth_stick', fourth_stick)]:
            rows[name].append(np.stack(values))
    previous.atomic_npz(filename, **{k:np.stack(v) for k,v in rows.items()}, alpha=alpha,
                        signs=SIGNS, regimes=REGIMES, methods=METHODS, full_ms=time_ms,
                        order=order, timing_stratum=stratum)
    previous.atomic_json(filename.with_suffix('.json'), dict(
        cell=CELLS[c], catalog=catalog, block=index, base_seed=seed, start=start, pair=pair,
        calibration_seeds=[seed+100+e for e in range(8)],
        deployment_seeds=[seed+200+e for e in range(8)],
        fingerprint=fingerprint, sha256=previous.digest(filename)))
    print('completed', CELLS[c], catalog, index, flush=True)


def main():
    if not (os.environ.get('SLURM_JOB_ID') and previous.torch.cuda.is_available()
            and previous.torch.cuda.device_count() == 1):
        raise RuntimeError('One assigned GPU is required')
    previous.torch.set_num_threads(2)
    locked = historical.source_hashes()
    # Include the reused v39 timing function and its imported world.
    old = json.loads((PROJECT/'probe_transfer_v39/results/completion.json').read_text())['hashes']
    for path in ('probe_transfer_v39/run39.py', 'probe_transfer_v39/physics39.py'):
        if previous.digest(PROJECT/path) != old[path]:
            raise RuntimeError(f'Historical dependency changed: {path}')
        locked[path] = old[path]
    for line in (ROOT/'freeze40.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if previous.digest(ROOT/name) != expected:
            raise RuntimeError(f'Prelaunch checksum mismatch: {name}')
        locked[str((ROOT/name).relative_to(PROJECT))] = expected
    fingerprint = hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    metadata = dict(fingerprint=fingerprint, hashes=locked, job=os.environ['SLURM_JOB_ID'],
                    gpu=previous.torch.cuda.get_device_name(), torch=str(previous.torch.__version__),
                    cuda=previous.torch.version.cuda, methods=METHODS, cells=CELLS, regimes=REGIMES,
                    signs=SIGNS, base=BASE, beta=previous.BETA, shrink=previous.SHRINK,
                    speed=previous.SPEED, height=previous.HEIGHT, catalogs=CATALOGS,
                    blocks_per_catalog=BLOCKS)
    previous.atomic_json(OUT/'run.json', metadata)
    with previous.torch.inference_mode():
        detector = previous.Detector()
        artifact = previous.load_artifacts()[previous.SPEED]
        for panel in range(CATALOGS):
            for b in range(BLOCKS):
                seed = BASE + 100_000*panel + 10_000*b
                pair = [seed+301, seed+302]
                alpha = previous.ALPHAS[np.random.default_rng(seed+401).permutation(8)]
                decoders = [previous.Engine(pair,*a) for a in alpha]
                cache = {}
                for c in np.random.default_rng(seed+707).permutation(len(CELLS)):
                    block(detector, artifact, fingerprint, int(c), panel, b, decoders,
                          alpha, pair, seed, cache)
    previous.atomic_json(OUT/'completion.json', dict(**metadata, blocks=288, timing_clusters=24))


if __name__ == '__main__':
    main()
