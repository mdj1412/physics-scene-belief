"""Paired v43 GPU inference under a common neutral bank, with evaluator-only labels."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

import inference43 as inference
import model43
import world43 as world

r = inference.r
v38 = r.v38
torch = r.torch
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / 'results'
DECODERS = ('original','robust','privileged')
POLICIES = ('zero','temperature','fixed','optimizer')
METHODS = tuple(f'{d}_{p}' for d in DECODERS for p in POLICIES)
PANELS, ENVS, REPEATS, TIME_REPS = 12, 4, 4, 3
CELLS = tuple(itertools.product(world.TARGET_SHAPES, world.TARGET_SURFACES,
                                world.TARGET_LAWS, world.VARIANTS, world.HEIGHTS))
BASE = 430000000


def cell_path(panel, cell):
    shape, surface, law, variant, height = cell
    name = f'{shape}_{surface}_{law}_{variant}_h{round(1000*height)}'
    return OUT / f'panel_{panel:02d}' / (name + '.npz')


def correction(base, d, p, source, artifact, records):
    return model43.correct(p, base, d, source, r, artifact, records)


def base_probability(name, source, engine, template, xx, selected, paid, height):
    p = inference.decode43(engine, template, xx, selected, paid)
    if name == 'robust':
        p = model43.robust_probability(source, p, xx, selected, paid, height)
    if not np.isfinite(p).all() or np.any((p < -1e-12) | (p > 1+1e-12)):
        raise ValueError('Invalid deployment probability')
    return p


def generate(panel, cell):
    shape, surface, law, variant, height = cell
    seed = BASE + 100000*panel
    perm = np.random.default_rng(seed + 401).permutation(len(v38.ALPHAS))
    alphas = v38.ALPHAS[perm[:ENVS]]
    signs = np.array([(-1,1)[(panel+e)%2] for e in range(ENVS)])
    regimes = [world.REGIMES[(panel+e)%3] for e in range(ENVS)]
    start = (-1,1)[panel%2]
    ctx = [(shape, variant, height, surface, law, int(signs[e]), panel) for e in range(ENVS)]
    dep = [world.deployment(seed+200+e, a, ctx[e], regimes[e], REPEATS) for e,a in enumerate(alphas)]
    obs = [rec[3] for rec in dep]
    xx = np.stack([rec[0] for rec in dep])
    records = [world.calibration(seed+100+e, a, ctx[e], regimes[e], start)[0]
               for e,a in enumerate(alphas)]
    evaluator = (ctx, dep, alphas, signs, regimes, seed, start)
    return xx, obs, records, evaluator


def timing(panel, engine, height, direction, xx, selected, obs, records, source, artifact, pred, fingerprint):
    path = OUT/'timing'/f'panel_{panel:02d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if r.old.already_complete(path, fingerprint):
        return
    one = [{a: row[a][:,:1] for a in (r.Q, r.P)} for row in obs]
    warm = np.empty((TIME_REPS, len(METHODS)))
    setup = np.empty_like(warm)
    order = np.empty(warm.shape, np.int16)
    rng = np.random.default_rng(BASE+panel*100000+4011)
    biggest = 0.
    for rep in range(-1,TIME_REPS):
        schedule = rng.permutation(len(METHODS))
        if rep >= 0:
            order[rep] = schedule
        for k in schedule:
            di, pi = divmod(int(k), len(POLICIES))
            d, p = DECODERS[di], POLICIES[pi]
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            template = inference.make_template(engine, height, direction if d == 'privileged' else None)
            torch.cuda.synchronize()
            ready = time.perf_counter_ns()
            actual = inference.actions43(xx, engine)
            paid = inference.paid_sensors(one, actual)
            base = base_probability(d, source, engine, template, xx, actual, paid, height)
            prediction = correction(base, d, p, source, artifact, records)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            np.testing.assert_array_equal(actual, selected)
            expected = pred[di,pi,...,:1]
            np.testing.assert_allclose(prediction, expected, atol=2e-9, rtol=0)
            biggest = max(biggest, float(np.max(np.abs(prediction-expected))))
            if rep >= 0:
                warm[rep,k] = (end-ready)/1e6
                setup[rep,k] = (ready-start)/1e6
    v38.atomic_npz(path, warm_ms=warm, setup_ms=setup, cold_ms=warm+setup,
                   order=order, methods=METHODS, max_error=biggest)
    v38.atomic_json(path.with_suffix('.json'), dict(sha256=v38.digest(path),fingerprint=fingerprint))


def block(panel,cell,engine,source,artifact,cache,templates,fingerprint,timer=False):
    path = cell_path(panel,cell)
    if not timer and r.old.already_complete(path,fingerprint):
        return
    shape,surface,law,variant,height = cell
    xx,obs,records,evaluator = generate(panel,cell)
    key = hashlib.sha256(xx.tobytes()).digest()
    if key not in cache:
        cache[key] = inference.actions43(xx,engine)
    selected = cache[key]
    paid = inference.paid_sensors(obs,selected)
    if cell == CELLS[0] and not timer:
        # Replace every unbought observation and attach decoy metadata before the
        # operating sensor adapter. Its output must be exactly the original input.
        poison = []
        for e, original in enumerate(obs):
            decoy = {'law': 'decoy', 'surface': 'decoy', 'object_family': 'decoy'}
            for a in (r.Q, r.P):
                decoy[a] = original[a].copy()
                decoy[a][selected[e] != a] = 999.
            poison.append(decoy)
        blinded = inference.paid_sensors(poison, selected)
        for before, after in zip(paid, blinded):
            assert set(before) == set(after) == {r.Q, r.P}
            for a in (r.Q, r.P):
                np.testing.assert_array_equal(before[a], after[a])
    direction = world.oracle_direction(shape,height)
    if (None,height) not in templates:
        templates[None,height] = inference.make_template(engine,height)
    if (shape,height) not in templates:
        templates[shape,height] = inference.make_template(engine,height,direction)
    if timer:
        with np.load(path) as raw:
            prediction = raw['pred']
        timing(panel,engine,height,direction,xx,selected,obs,records,source,artifact,prediction,fingerprint)
        return
    pred = []
    for d in DECODERS:
        template = templates[shape if d=='privileged' else None,height]
        p = base_probability(d,source,engine,template,xx,selected,paid,height)
        pred.append([correction(p,d,action,source,artifact,records) for action in POLICIES])
    pred = np.array(pred)
    # Score only after terminal predictions, including correction, are complete.
    ctx,dep,alphas,signs,regimes,seed,start = evaluator
    labels = np.stack([world.offset_labels(dep[e][2],ctx[e],regimes[e]) for e in range(ENVS)],axis=1)
    np.testing.assert_array_equal(labels[4],np.stack([rec[1] for rec in dep]))
    risk = np.mean((pred[None]-labels[:,None,None,:,:,None])**2,axis=(-1,-2))
    path.parent.mkdir(parents=True,exist_ok=True)
    x,y,dirs = v38.observations(records)
    z,weights = r.continuous.source_optimum(artifact,x,y[:,:3],dirs[:,:3],y[:,3])
    v38.atomic_npz(path,labels=labels,pred=pred,risk=risk,selected=selected,
                   xx=xx,query_paid=np.stack([d[r.Q] for d in paid]),probe_paid=np.stack([d[r.P] for d in paid]),
                   calibration_x=x,calibration_y=y,calibration_dirs=dirs,z=z,weights=weights,
                   oracle_direction=direction,seed=seed,alphas=alphas,signs=signs,regimes=regimes,start=start,
                   decoders=DECODERS,policies=POLICIES,offsets_mm=world.OFFSETS_MM,height=height)
    v38.atomic_json(path.with_suffix('.json'),dict(sha256=v38.digest(path),fingerprint=fingerprint))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('even','odd','timing'))
    args=parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('One allocated GPU required')
    torch.set_num_threads(1)
    prior = json.loads((PROJECT/'estimated_projection_v41/results/run.json').read_text())['hashes']
    locked = dict(prior)
    for line in (ROOT/'freeze43.sha256').read_text().splitlines():
        val,name=line.split(maxsplit=1)
        locked[ROOT.name+'/'+name]=val
    for name,val in locked.items():
        if v38.digest(PROJECT/name)!=val:
            raise ValueError('Dependency mismatch '+name)
    fingerprint=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    meta=dict(fingerprint=fingerprint,hashes=locked,job=os.getenv('SLURM_JOB_ID'),
              gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
              pair=inference.PAIR,alpha_prior=[0.,0.],base=BASE,panels=PANELS,repeats=REPEATS,
              cells=[list(c) for c in CELLS],environments=ENVS,decoders=DECODERS,policies=POLICIES,
              offsets_mm=world.OFFSETS_MM,mode=args.mode)
    v38.atomic_json(OUT/f'run_{args.mode}.json',meta)
    source=model43.load_model()
    artifact=v38.load_artifacts()[v38.SPEED]
    with torch.inference_mode():
        engine=inference.make_bank()
        templates={}
        for panel in range(PANELS):
            if args.mode in ('even','odd') and panel%2!=int(args.mode=='odd'):
                continue
            cache={}
            if args.mode=='timing':
                # Cover all 8 fam combinations and both heights in randomized crossovers.
                family=panel%8
                cell=(world.TARGET_SHAPES[family//4],world.TARGET_SURFACES[(family//2)%2],
                      world.TARGET_LAWS[family%2],world.VARIANTS[(panel//2)%3],world.HEIGHTS[panel%2])
                block(panel,cell,engine,source,artifact,cache,templates,fingerprint,True)
            else:
                for i in np.random.default_rng(BASE+panel*100000+707).permutation(len(CELLS)):
                    block(panel,CELLS[i],engine,source,artifact,cache,templates,fingerprint)
            print('completed',args.mode,'panel',panel,flush=True)
    v38.atomic_json(OUT/f'complete_{args.mode}.json',meta)


if __name__=='__main__':
    main()
