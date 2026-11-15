"""Evaluate locked, direction-free calibrators on archived v35 fixed-4 predictions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent
FREEZE_HASH = 'd7da20831a220f3f3234d18189ba711002fd8ef207f4412a71a6cbffd5a39607'
SCHEDULES = ('same', 'alternating')
PRIMARY = ('sinusoid_atan', 'sinusoid_algebraic', 'quartic_atan', 'quartic_algebraic')
CONTROL = ('sinusoid_isotropic', 'quartic_isotropic')
METHODS = ('original', 'constant_offset', 'temperature', 'frozen_signed', 'temperature_then_frozen_signed')
ITERATIONS = 12000


def panel(root, cells, orientations, lock):
    risks = np.empty((len(cells), 8, len(orientations), 2, len(METHODS)))
    for f, family in enumerate(cells):
        for b in range(8):
            for s, sign in enumerate(orientations):
                base = root / (family+'_'+sign) / f'block_{b:03d}'
                meta = json.loads((base/'meta.json').read_text())
                for a, schedule in enumerate(SCHEDULES):
                    path = base/(schedule+'.npz')
                    assert sha256(path.read_bytes()).hexdigest() == meta['archives'][schedule]
                    with np.load(path, allow_pickle=False) as pack:
                        p = pack['p0']
                        obs = pack['labels'][...,None]
                        corrected = pack['p1']
                        assert p.shape == corrected.shape == (8, 128, 32)
                        assert obs.shape == (8, 128, 1) and np.isin(obs,[0,1]).all()
                        eps = lock['probability_clip']
                        eta = logit(np.clip(p,eps,1-eps))
                        intercept = expit(eta + lock['intercept'])
                        temperature = expit(eta * lock['inverse_temperature'])
                        assert float(pack['betas'][0]) == 0.8289760825845313
                        z = pack['z'][0, :, None, None]
                        assert z.shape == (8, 1, 1) and np.max(np.abs(z)) <= 1 + 1e-12
                        composed = expit(eta * lock['inverse_temperature'] + 0.8289760825845313*z)
                        risks[f,b,s,a] = [np.mean((pred-obs)**2)
                                           for pred in (p,intercept,temperature,corrected,composed)]
    return risks


def contrasts(risks):
    # ... × cell × schedule × method. Combine all cells with equal weight.
    all_cells = np.concatenate([risks.mean(axis=-3,keepdims=True),risks],axis=-3)
    return np.stack((all_cells[...,3]-all_cells[...,1],
                     all_cells[...,3]-all_cells[...,2],
                     all_cells[...,4]-all_cells[...,2]),axis=-1)


def summarize(risks, cells, seed):
    # Original 32 sensor draws, both mirrored orientations, same start sign and
    # both schedules stay paired inside each source-world block resample.
    blocks = risks.mean(2)
    cell_risk = blocks.mean(1)
    means = np.concatenate([cell_risk.mean(0,keepdims=True),cell_risk])
    offsets = contrasts(cell_risk)
    rng = np.random.default_rng(seed)
    n = len(cells)
    idx = (2*rng.integers(4,size=(ITERATIONS,n,4,2))+np.arange(2)).reshape(ITERATIONS,n,8)
    reps = contrasts(blocks[np.arange(n)[None,:,None],idx].mean(2))
    std = reps.std(axis=0,ddof=1)
    scaled = np.divide(np.abs(reps-offsets),std,out=np.zeros_like(reps),where=std>1e-14)
    cutoff = np.quantile(scaled.reshape(ITERATIONS,-1).max(axis=1),0.95)
    interval = np.stack([offsets-cutoff*std,offsets+cutoff*std],axis=-1)
    return dict(cells=('pooled',)+cells,schedules=SCHEDULES,methods=METHODS,absolute_brier=means.tolist(),
                contrasts=('frozen_signed minus constant_offset', 'frozen_signed minus temperature',
                           'temperature_then_frozen_signed minus temperature'),
                delta=offsets.tolist(),joint_ci95=interval.tolist(),max_standardized_cutoff=float(cutoff),
                simultaneous_comparisons=int(offsets.size),bootstrap_draws=ITERATIONS,bootstrap_seed=seed)


def table(data):
    lines = ['| Object × surface | Schedule | Original | Offset | Temperature | Signed | T + signed | Signed − offset [95%] | Signed − temperature [95%] | (T + signed) − T [95%] |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f,cell in enumerate(data['cells']):
        for a,schedule in enumerate(SCHEDULES):
            vals = data['absolute_brier'][f][a]
            fields = [cell,schedule]+[f'{x:.5f}' for x in vals]
            for delta,ci in zip(data['delta'][f][a],data['joint_ci95'][f][a]):
                fields.append(f'{delta:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]')
            lines.append('| '+' | '.join(fields)+' |')
    return '\n'.join(lines)


def report(target, results):
    p,c = results['directional'],results['isotropic']
    p_hi = np.asarray(p['joint_ci95']).max(axis=-1)[0]
    c_lo = np.asarray(c['joint_ci95']).min(axis=-1)[0]
    overview = []
    for i,sc in enumerate(SCHEDULES):
        for j,baseline in enumerate(('constant offset','temperature scaling')):
            delta = p['delta'][0][i][j]
            ci = p['joint_ci95'][0][i][j]
            if p_hi[i,j]<0:
                state = 'signed correction의 이득을 지지한다'
            elif ci[0]>0:
                state = 'score-only 보정의 이득을 지지한다'
            else:
                state = '두 방법의 순서는 이 자료에서 확정하기 어렵다'
            overview.append(f'- {sc}에서 signed correction과 {baseline}의 pooled Brier 차이는 {delta:+.5f}이고 simultaneous 95% CI는 [{ci[0]:+.5f}, {ci[1]:+.5f}]다. {state}.')
    for i,sc in enumerate(SCHEDULES):
        delta = p['delta'][0][i][2]
        lo,hi = p['joint_ci95'][0][i][2]
        phrase = '추가 이득을 지지한다' if hi < 0 else ('추가 손해를 지지한다' if lo > 0 else '추가 효과의 방향은 확정하기 어렵다')
        overview.append(f'- Temperature를 먼저 적용한 뒤 같은 βz를 더하면 {sc}에서 Brier 차이가 {delta:+.5f} [{lo:+.5f}, {hi:+.5f}]다. {phrase}.')
    bounds = np.asarray(p['joint_ci95'])[1:,...,1]
    both = np.all(bounds[:,:,:2] < 0,axis=-1).sum()
    stacked = np.sum(bounds[:,:,2] < 0)
    overview.append(f'- 개별 directional family × schedule 8개 중 두 score-only 기준을 모두 앞선 것은 {both}개다. Temperature 후 βz의 추가 이득은 {stacked}개에서 simultaneous CI 상한이 0보다 작았다.')
    mixed=[]
    for f,cell in enumerate(p['cells'][1:],1):
        if np.any(np.asarray(p['joint_ci95'])[f,:,:2,0]>0):
            mixed.append(cell)
    if mixed:
        overview.append('- Directional family 중 적어도 한 schedule과 score-only 기준에서 signed correction의 불리함이 검출된 조합은 '+', '.join(mixed)+'다.')
    if np.any(c_lo[:,:2]>0):
        for i,sc in enumerate(SCHEDULES):
            delta = c['delta'][0][i][2]
            lo,hi = c['joint_ci95'][0][i][2]
            overview.append(f'- Isotropic control에서 temperature 후 βz를 더한 pooled 차이는 {sc}에서 {delta:+.5f} [{lo:+.5f}, {hi:+.5f}]였다. 방향이 없는 target은 correction이 손해를 일으킬 수 있는 실험적 반례다.')
    text = f'''# v35에서 방향 정보 없이도 같은 개선을 얻을 수 있는가?

## Source에서 고정한 두 가지 대조군

V34의 fold 0에서는 estimated direction의 β를 **ridged_coarse**의 8개 block × 2개 orientation × 2개 schedule에서 공동으로 학습했다. 같은 32개 metrics archive에서 fixed 4의 같은 action을 선택했다. 각 stratum에서 8개 환경 × 128개 deployment worlds × 32개 sensor draws의 예측을 사용했다. 이 수는 1,048,576개 score이며, 같은 label과 prefix를 공유하는 반복을 포함한다. 두 source-only 보정은 target 자료 없이 CPU에서 학습·고정했다. [원래 fit](../bidirectional_probe_v34/anchored34.py) · [source-only fit 및 32개 입력 해시](source_cal35_frozen.json)

원 physics probability를 p, logit을 η라고 할 때 constant offset은 sigmoid(η + 0.01965140895), temperature scaling은 sigmoid(0.55849826698 η)다. T는 1.79051585140이다. 두 방법 모두 p를 [10⁻¹², 1−10⁻¹²]로 제한해 logit을 정의하고, binary log loss에 v34와 같은 0.005의 quadratic ridge를 더했다. Ridge의 기준은 identity에 해당하는 offset 0 및 inverse temperature 1이다. 원본 p가 경계일 때도 이 작은 clipping을 적용했다. Target family, schedule, sign에 관계없이 같은 계수를 사용했다. [학습 코드](fit_source_cal35.py) · [고정 계수와 입력 해시](source_cal35_frozen.json)

## 독립 target에서의 paired Brier

원래 fixed 4 action 및 관측은 다섯 방법에서 같다. T + signed는 먼저 temperature를 적용한 logit에 같은 v34 βz를 더해 sigmoid를 취한다. 음수는 각 contrast에서 왼쪽 방법의 Brier가 낮음을 뜻한다. [결과 확인 전 고정한 비교](SOURCE_CAL_PLAN_KO.md)

CI는 object × surface 내에서 block을 첫 probe 부호로 층화해 12,000회 재표집한 paired max-standardized interval이다. Mirrored orientation·두 schedule·동일 환경의 32개 sensor draws는 block 안에 함께 남겼다. Directional 표의 pooled 및 cell별 모든 30개 contrast에 대해 동시에 95%를 적용했다. Isotropic 표에서는 독립적으로 18개 contrast를 보정했다. 각 CI는 고정된 family의 block 변동에 관한 추정이다. [평가 코드](score_source_cal35.py) · [수치 원본](source_cal35.json)

{table(p)}

### Isotropic control

{table(c)}

## 해석

{chr(10).join(overview)}

다섯 방법은 같은 sensor action을 쓰므로 이 결과가 비교하는 양은 관측 뒤 확률의 품질이다. Score-only baseline에 이겨도 추정한 sign이 단일 물리적 원인임을 식별하지는 못한다. Surface의 다른 속성과 reader 출력이 함께 달라질 수 있다. 원 score의 전반적인 보정과 환경 간 판별력이 각각 얼마나 변했는지는 독립 reliability·resolution 분석과 within-environment sign intervention으로 평가할 수 있다. 이 부록의 두 대조군에는 추가 GPU timing을 측정하지 않았다. [Brier decomposition](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) · [physics residual의 선행 검증](https://journals.sagepub.com/doi/full/10.1177/0278364920954896)
'''
    target.write_text(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    args=parser.parse_args()
    raw=(ROOT/'source_cal35_frozen.json').read_bytes()
    assert sha256(raw).hexdigest()==FREEZE_HASH
    lock=json.loads(raw)
    d=summarize(panel(args.results,PRIMARY,('plus','minus'),lock),PRIMARY,350417)
    c=summarize(panel(args.results,CONTROL,('plus',),lock),CONTROL,350419)
    results=dict(source_model_sha256=FREEZE_HASH,directional=d,isotropic=c)
    (ROOT/'source_cal35.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    report(ROOT/'SOURCE_CAL_KO.md',results)
    print(np.array(d['delta'])[0], np.array(d['joint_ci95'])[0])


if __name__=='__main__':
    main()
