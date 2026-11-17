"""Label-free measurement QA against independently integrated deformation functions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t as student
import world41 as w

ROOT=Path(__file__).resolve().parent
PREFIXES=(4,8,16,32)


def truth(family,height):
    """Evaluate compact lobes in centered coordinates; independent of generator's path."""
    weights={'flare':(.80,-.85,.30,.95),'roll':(-.45,.55,.95,-.55)}[family]
    knots=np.array([11.,19.,23.,31.,39.])
    y=1000*height+np.array([-6.,-4.,-2.,0.,2.,4.,6.])
    shift=np.zeros(7)
    for a,b,k in zip(knots[:-1],knots[1:],weights):
        x=(2*y-a-b)/(b-a)
        shift+=.0009*k*np.where(np.abs(x)<1.,(1.-x*x)**2,0.)
    return shift


def reference_estimate(pairs):
    delta=(pairs[:,1]-pairs[:,0])/.00030
    _,vec=np.linalg.eigh(np.cov(delta,rowvar=False,ddof=1))
    return vec[:,-1]


def angle(est,actual):
    return float(np.rad2deg(np.arccos(np.clip(abs(est@actual),0.,1.))))


def summary(values):
    values=np.asarray(values)
    m=float(values.mean())
    err=float(student.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=m,pointwise95=[m-err,m+err],minimum=float(values.min()),maximum=float(values.max()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panels',type=int,required=True)
    ap.add_argument('--check-archives',action='store_true')
    args=ap.parse_args()
    if os.environ.get('BELIEF_DEVICE')!='cpu':
        raise RuntimeError('Compute the independent audit with CUDA disabled')
    w.torch.set_num_threads(1)
    cells={}
    streams=[]
    for f in w.FAMILIES:
        for h in w.HEIGHTS:
            key=f'{f}_{round(h*1000)}'
            actual=truth(f,h)
            truth_norm=float(np.linalg.norm(actual)/.0003)
            unit=actual/np.linalg.norm(actual)
            np.testing.assert_allclose(w.oracle_direction(f,h),unit,atol=2.e-13,rtol=0)
            records=[]
            for p in range(args.panels):
                streams.append(w.pair_seed(p,f,h))
                pairs=w.direction_pairs(p,f,h)
                est=w.estimate_direction(pairs)
                if args.check_archives:
                    path=ROOT/'results'/'directions'/f'panel_{p:02d}_{f}_h{round(1000*h)}.npz'
                    with np.load(path) as stored:
                        np.testing.assert_allclose(stored['pairs'],pairs,rtol=0.,atol=2.e-13)
                        if min(abs(stored['estimated']@est),abs(stored['oracle']@unit))<1.-1.e-10:
                            raise ValueError('Saved or estimated direction mismatch')
                check=reference_estimate(pairs)
                if abs(check@est)<1.-1.e-10:
                    raise ValueError('Independent covariance eigensolver disagrees')
                measures={}
                for n in PREFIXES:
                    subset=pairs[:n]
                    no_pair=subset.copy()
                    no_pair[:,1]=np.roll(subset[:,1],1,axis=0)
                    measured=angle(w.estimate_direction(subset),unit)
                    control=angle(w.estimate_direction(no_pair),unit)
                    measures[n]={'angle_deg':measured,'unpaired_angle_deg':control,
                                 'surviving_norm':float(np.sin(np.deg2rad(measured)))}
                records.append({'panel':p,'direction':est.tolist(),'prefix':measures})
            group={str(n):{k:summary([r['prefix'][n][k] for r in records])
                           for k in ('angle_deg','unpaired_angle_deg','surviving_norm')} for n in PREFIXES}
            cells[key]=dict(oracle=unit.tolist(),full_signal_sigma=truth_norm,panel=records,prefix=group)
    assert len(set(streams))==len(streams)
    data={'panels':args.panels,'passive_per_group':64,'stored_pairs_and_directions_verified':args.check_archives,'unique_passive_objects':len(streams)*32,
          'unique_profile_reads':len(streams)*64,'cells':cells}
    angles=data['cells']
    out=['# v41 방향 추정 검산','',
         '모든 값은 독립 passive calibration 물체에서 얻었다. Angle은 추정 방향과 oracle 방향의 부호를 무시한 각도이며 0°면 완전히 일치한다. 남는 변형 길이 비율은 sin(angle)이다. 같은 물체 pairing을 깨뜨린 control은 각 prefix에서 두 번째 profile만 한 행씩 옮겨 계산했다.','',
         '| Shape / height | 4 pairs | 8 pairs | 16 pairs | 32 pairs [panel 평균 95%] | 32 pairs 제거 후 남는 길이 | Unpaired 32 평균 각도 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,cell in angles.items():
        m=cell['prefix']
        u=m['32']['angle_deg']
        out.append('| '+name.replace('_',' / ')+' mm | '+' | '.join(f"{m[str(n)]['angle_deg']['mean']:.2f}°" for n in (4,8,16))
                   +f" | {u['mean']:.2f}° [{u['pointwise95'][0]:.2f}, {u['pointwise95'][1]:.2f}] | {m['32']['surviving_norm']['mean']:.1%} | {m['32']['unpaired_angle_deg']['mean']:.2f}° |")
    out+=['',f"매 group의 첫 32개 pair를 사용하는 기준에서 {len(streams)}개 독립 group calibration에 {data['unique_passive_objects']:,}개의 추가 물체, {data['unique_profile_reads']:,}회 profile readout이 필요했다. Calibrated group은 두 law 및 모든 variant·시작 block에 재사용했다. Angle을 서로 다른 eigensolver로 재검산했고 oracle 방향도 별도의 compact support 식으로 검산했다. [원자료](direction_audit41.json)"]
    (ROOT/'direction_audit41.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (ROOT/'DIRECTION_AUDIT_KO.md').write_text('\n'.join(out)+'\n')

if __name__=='__main__':main()
