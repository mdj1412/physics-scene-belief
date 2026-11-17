"""Supplementary matched GPU timing of a decoder with no correction after the primary freeze."""
import json
import os
import time
import numpy as np
import runner41 as r
import stats41

v38=r.v38
METHODS=('original_uncorrected','original_fixed','estimated_fixed')


def main():
    if not (os.environ.get('SLURM_JOB_ID') and r.torch.cuda.is_available() and r.torch.cuda.device_count()==1):
        raise RuntimeError('One allocated GPU required')
    meta=json.loads((r.OUT/'completion.json').read_text())
    r.torch.set_num_threads(2)
    detector=v38.Detector()
    artifact=v38.load_artifacts()[v38.SPEED]
    warm=np.empty((12,3,3));setup=np.empty_like(warm);order=np.empty((12,3,3),int)
    maximum=0.
    for panel in range(12):
        c,block,stratum=r.schedule(panel)
        s,ri=divmod(stratum,3)
        path=r.rawpath(c,panel,block)
        with np.load(path) as raw:
            alpha=raw['alpha'];actions=raw['actions'][s];z=raw['z'][s]
            measured=raw['p_base'][:,s,:,:,0]
        m=json.loads(path.with_suffix('.json').read_text())
        family,hkey,variant,law=r.CONDITIONS[c]
        height=r.hvalue(hkey)
        context=(family,variant,height,law,r.SIGNS[s],panel)
        seed=m['base_seed']
        records=[r.world.calibration(seed+100+e,a,context,r.REGIMES[ri],m['start'])[0]
                 for e,a in enumerate(alpha)]
        deploy=[r.world.deployment(seed+200+e,a,context,r.REGIMES[ri],32) for e,a in enumerate(alpha)]
        xx=np.stack([row[0] for row in deploy])
        one=[{a:row[3][a][:,:1] for a in (r.Q,r.P)} for row in deploy]
        banks=[v38.Engine(m['pair'],*a).inner for a in alpha]
        with np.load(r.direction_file(panel,family,hkey)) as d:
            paired=d['pairs']
        rng=np.random.default_rng(seed+41092+c)
        for rep in (-1,0,1,2):
            shuffled=rng.permutation(3)
            for method in shuffled:
                r.torch.cuda.synchronize()
                start=time.perf_counter_ns()
                unit=r.world.estimate_direction(paired) if method==2 else None
                templates=r.templates(banks,height,unit)
                r.torch.cuda.synchronize()
                end_setup=time.perf_counter_ns()
                actual=v38.prefix_actions(detector,records,m['pair'],xx)
                prediction=r.decode(banks,templates,xx,actual,one)[:,:,0]
                if method:
                    # Same calibration/policy calls used by the primary fixed timing.
                    v38.observations(records)
                    zz,ww=v38.policy(artifact,records,'fixed',True)
                    prediction=v38.apply(prediction,(zz*ww)[:,None],v38.BETA)
                r.torch.cuda.synchronize()
                stop=time.perf_counter_ns()
                expected=measured[2 if method==2 else 0]
                if method:
                    expected=v38.apply(expected,(.325*z)[:,None],v38.BETA)
                maximum=max(maximum,float(np.abs(expected-prediction).max()))
                np.testing.assert_allclose(expected,prediction,rtol=0.,atol=2.e-10)
                np.testing.assert_array_equal(actions,actual)
                if rep>=0:
                    warm[panel,rep,method]=(stop-end_setup)/1e6
                    setup[panel,rep,method]=(end_setup-start)/1e6
            if rep>=0:order[panel,rep]=shuffled
        print('Completed timing crossover panel',panel,flush=True)
    cold=warm+setup
    v38.atomic_npz(r.ROOT/'latency_uncorrected41.npz',full_ms=warm,setup_ms=setup,cold_ms=cold,order=order,methods=METHODS)
    def evaluate(data):
        p90=np.quantile(data,.9,axis=(0,1))
        bootstrap=np.quantile(data[stats41.BOOT],.9,axis=(1,2))
        return dict(p90=p90.tolist(),ci=np.quantile(bootstrap,[.025,.975],axis=0).T.tolist(),
                    ratio_to_uncorrected=(p90/p90[0]).tolist(),
                    ratio_to_uncorrected_ci=np.quantile(bootstrap/bootstrap[:,[0]],[.025,.975],axis=0).T.tolist())
    v38.atomic_json(r.ROOT/'latency_uncorrected41.json',dict(full=evaluate(warm),cold=evaluate(cold),setup=evaluate(setup),
            job=os.environ['SLURM_JOB_ID'],gpu=r.torch.cuda.get_device_name(),max_error=maximum,
            main_fingerprint=meta['fingerprint'],methods=METHODS,panels=12,repeats=3))
    print('Completed supplementary GPU crossover',flush=True)


if __name__=='__main__':
    with r.torch.inference_mode():
        main()
