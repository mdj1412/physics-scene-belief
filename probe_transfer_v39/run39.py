"""Prospective multi-catalog comparison of the two frozen v38 policies."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PREVIOUS = PROJECT / 'probe_value_v38'
sys.path.insert(0, str(PREVIOUS))
import continuous38
import run38 as previous
import physics39 as world

CELLS = ('shelf_weakening', 'shelf_strengthening', 'ridge_weakening', 'ridge_strengthening')
SIGNS = (1, -1)
REGIMES = ('strong', 'weak', 'null')
METHODS = ('fixed', 'optimizer')
CATALOGS = 12
BLOCKS = 2
OUT = ROOT / 'results'
BASE = 390_000_000
OPTIMIZER_HASH = '9ba308386d5e3275cf22160d34a2d6b030f8fbc5e746a76aa87bfa6742e857e8'


def source_hashes():
    old = json.loads((PREVIOUS/'results/run.json').read_text())['hashes']
    locked = {(name if '/' in name else 'probe_value_v38/'+name): value for name, value in old.items()}
    locked['probe_value_v38/continuous38.py'] = OPTIMIZER_HASH
    for name, expected in locked.items():
        if previous.digest(PROJECT/name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    return locked


def timing(detector, artifact, records, pair, decoders, xx, obs, actions, base, z, weight, seed):
    sensors = [{a:item[a][:,0] for a in (previous.Q,previous.P)} for item in obs]
    rng = np.random.default_rng(seed)
    times = np.empty((3,2))
    orders = np.empty((3,2),dtype=np.int8)
    for rep in (-1,0,1,2):
        order = rng.permutation(2)
        for method in order:
            if method == 0:
                elapsed, selected, prob = previous.online(detector,artifact,'fixed',records,pair,decoders,xx,sensors)
                w = previous.SHRINK
            else:
                elapsed, selected, prob = continuous38.measure(detector,artifact,records,pair,decoders,xx,sensors)
                w = weight
            np.testing.assert_array_equal(selected,actions)
            expected = previous.apply(base[:,:,0],(z*w)[:,None],previous.BETA)
            np.testing.assert_allclose(prob,expected,rtol=0.,atol=1.e-8)
            if rep >= 0:
                times[rep,method] = elapsed
        if rep >= 0:
            orders[rep] = order
    return times, orders


def block(detector, artifact, fingerprint, cell_index, catalog, index):
    cell = CELLS[cell_index]
    shape, law = cell.split('_')
    dst = OUT/cell/f'catalog_{catalog:02d}'
    dst.mkdir(parents=True, exist_ok=True)
    filename = dst/f'block_{index:02d}.npz'
    meta = filename.with_suffix('.json')
    if meta.exists():
        info = json.loads(meta.read_text())
        if info['fingerprint'] != fingerprint or previous.digest(filename) != info['sha256']:
            raise RuntimeError(f'The saved block does not match the run manifest: {filename}')
        return
    seed = BASE + 10_000_000*cell_index + 100_000*catalog + 10_000*index
    pair = [seed+301,seed+302]
    start = (-1,1)[index]
    alpha = previous.ALPHAS[np.random.default_rng(seed+401).permutation(8)]
    decoders = [previous.Engine(pair,*a) for a in alpha]
    rows = {key:[] for key in ('risk','weights','p_base','labels','actions','z','cal_x',
                               'ids','first3_y','fourth_y','fourth_stick','dirs')}
    time_ms = np.empty((0,2))
    order = np.empty((0,2),dtype=np.int8)
    timing_stratum = -1
    if index == 0 and catalog % 2 == 0:
        timing_stratum = (catalog//2+cell_index)%6
    for sign_index, sign in enumerate(SIGNS):
        context = (shape,law,sign,catalog)
        risks, weights, labels, velocity, stick = [], [], [], [], []
        previous_rows = None
        for ri, regime in enumerate(REGIMES):
            cal = [world.calibration(seed+100+e,a,context,regime,start) for e,a in enumerate(alpha)]
            records = [entry[0] for entry in cal]
            x, y, dirs = previous.observations(records)
            xx, target, _, obs = zip(*[world.deployment(seed+200+e,a,context,regime,repeats=32)
                                       for e,a in enumerate(alpha)])
            xx, target = np.stack(xx), np.stack(target)
            if previous_rows is None:
                actions = previous.prefix_actions(detector,records,pair,xx)
                base = previous.paired_decode(decoders,xx,actions,obs)
                previous_rows = (x, y[:,:3],xx,obs)
                rows['actions'].append(actions)
                rows['p_base'].append(base)
                rows['cal_x'].append(x)
                rows['first3_y'].append(y[:,:3])
                rows['dirs'].append(dirs[:,:3])
                rows['ids'].append(np.stack([r[1] for r in records]))
            else:
                for actual,expected in zip((x,y[:,:3],xx),previous_rows[:3]):
                    np.testing.assert_array_equal(actual, expected)
                for actual,expected in zip(obs,previous_rows[3]):
                    for key in (previous.Q,previous.P):
                        np.testing.assert_array_equal(actual[key],expected[key])
            z,w = continuous38.source_optimum(artifact,x,y[:,:3],dirs[:,:3],y[:,3])
            zfixed,wfixed = previous.policy(artifact,records,'fixed',True)
            np.testing.assert_allclose(z,zfixed,rtol=0,atol=1.e-12)
            np.testing.assert_array_equal(wfixed,np.full(8,previous.SHRINK))
            if ri == 0:
                rows['z'].append(z)
            else:
                np.testing.assert_array_equal(z,rows['z'][-1])
            paired_risk = []
            for multiplier in (wfixed,w):
                prob = previous.apply(base,(z*multiplier)[:,None,None],previous.BETA)
                paired_risk.append(np.square(prob-target[:,:,None]).mean(axis=(1,2)))
            risks.append(np.stack(paired_risk))
            weights.append(w)
            labels.append(target)
            velocity.append(y[:,3])
            stick.append(np.asarray([entry[3] for entry in cal]))
            if timing_stratum == sign_index*len(REGIMES)+ri:
                time_ms, order = timing(detector,artifact,records,pair,decoders,xx,obs,actions,
                                        base,z,w,seed+699)
        for key, data in (('risk',risks),('weights',weights),('labels',labels),
                          ('fourth_y',velocity),('fourth_stick',stick)):
            rows[key].append(np.stack(data))
    previous.atomic_npz(filename,**{name:np.stack(value) for name,value in rows.items()},alpha=alpha,
                        signs=SIGNS,regimes=REGIMES,methods=METHODS,
                        full_ms=time_ms,order=order,timing_stratum=timing_stratum)
    previous.atomic_json(meta,dict(cell=cell,catalog=catalog,block=index,base_seed=seed,start=start,
                                   pair=pair,calibration_seeds=[seed+100+e for e in range(8)],
                                   deployment_seeds=[seed+200+e for e in range(8)],
                                   fingerprint=fingerprint,sha256=previous.digest(filename)))
    print('completed',cell,catalog,index,flush=True)


def main():
    if not (os.environ.get('SLURM_JOB_ID') and previous.torch.cuda.is_available()
            and previous.torch.cuda.device_count()==1):
        raise RuntimeError('One assigned GPU is required')
    previous.torch.set_num_threads(2)
    locked = source_hashes()
    with open(ROOT/'freeze39.sha256') as handle:
        for line in handle:
            expected, filename = line.strip().split(maxsplit=1)
            if previous.digest(ROOT/filename) != expected:
                raise RuntimeError(f'Prelaunch checksum mismatch: {filename}')
            locked[str((ROOT/filename).relative_to(PROJECT))] = expected
    fingerprint = hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    metadata = dict(fingerprint=fingerprint,hashes=locked,job=os.environ['SLURM_JOB_ID'],
                    gpu=previous.torch.cuda.get_device_name(),torch=str(previous.torch.__version__),
                    cuda=previous.torch.version.cuda,methods=METHODS,cells=CELLS,regimes=REGIMES,
                    signs=SIGNS,base=BASE,beta=previous.BETA,shrink=previous.SHRINK,speed=previous.SPEED,
                    height=previous.HEIGHT,catalogs=CATALOGS,blocks_per_catalog=BLOCKS)
    previous.atomic_json(OUT/'run.json',metadata)
    with previous.torch.inference_mode():
        detector = previous.Detector()
        artifact = previous.load_artifacts()[previous.SPEED]
        for catalog in range(CATALOGS):
            for c in range(len(CELLS)):
                for b in range(BLOCKS):
                    block(detector,artifact,fingerprint,c,catalog,b)
    previous.atomic_json(OUT/'completion.json',dict(**metadata,blocks=96,timing_clusters=24))


if __name__=='__main__':
    main()
