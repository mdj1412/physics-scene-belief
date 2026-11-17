"""Build report tables and a standalone figure from the scored held-out panels."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def ci(x, digits=6, scale=1):
    a, lo, hi = np.asarray(x)*scale
    return f'{a:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def write_tables(s):
    ds = ('Original', 'Source-trained robust', 'Privileged projection')
    pol = ('0%', 'Temperature', 'Fixed 32.5%', 'v38 optimizer')
    loss = np.asarray(s['loss'])
    times = np.asarray(s['timing']['p90'])
    tci = np.asarray(s['timing']['ci'])
    rows = ['## 12개 조합의 held-out Brier와 GPU p90',
            '', '512개 물체(4개 환경 × 128개)를 처리했다. 각 물체는 배정된 관측 하나를 사용했다. Brier는 낮을수록 좋다. Nominal은 실제 접촉점과 명목 높이가 일치한 경우, shifted는 여덟 비영 signed offset의 균등 평균이다. 각 조합은 동일한 물체·충돌 결과·관측을 공유한다.', '',
            '| Decoder | Correction | Nominal Brier | Shifted Brier | Warm GPU p90, ms [95% CI] |',
            '|---|---|---:|---:|---:|']
    for d,name in enumerate(ds):
        for c,policy in enumerate(pol):
            a,b=tci[d,c]
            rows.append(f'| {name} | {policy} | {loss[0,d,c,0]:.6f} | {loss[1,d,c,0]:.6f} | {times[d,c]:,.2f} [{a:,.2f}, {b:,.2f}] |')
    rows += ['', '### 사전 지정한 네 대조', '',
             '양수는 앞 방법의 Brier가 더 높다는 뜻이다. 12개 독립 catalog에 대해 paired t 구간을 구하고 아래 네 대조의 family-wise error rate를 5%로 보정했다. 반복 관측과 이동 높이는 같은 물체의 측정이다.', '',
             '| 대조 | 평균 차이 [동시 95% CI] |', '|---|---:|']
    labels = ('Robust에서 optimizer−temperature, shifted','Privileged−original, 보정 0%, nominal',
              'Privileged−original, 보정 0%, shifted','위 geometry 차이의 shifted−nominal interaction')
    rows += [f'| {lbl} | {ci(x)} |' for lbl,x in zip(labels,s['primary'])]
    rows += ['', '### 어떤 family와 어느 signed offset에서 재현됐는가', '',
             '여덟 family 조합에서 같은 panel끼리 비교한 점추정이다. 동일한 두 수준의 family를 재조합했으므로 여덟 개의 독립적인 재현으로 세지 않는다.', '',
             '| Object / surface / law | Robust optimizer−T | Geometry Δ nominal | Geometry Δ shifted | Interaction |',
             '|---|---:|---:|---:|---:|']
    for k,v in s['group'].items():
        rows.append('| '+k+' | '+' | '.join(f'{a[0]:+.6f}' for a in v)+' |')
    rows += ['', '| 숨은 offset (mm) | Contour 변경에 따른 stick/slip 전환 | Geometry Δ, 0% [9점 동시 95% CI] |',
             '|---:|---:|---:|']
    for off, flip, g in zip(s['offsets_mm'],s['contour_flip'],s['geometry_by_offset']):
        rows.append(f'| {off:+g} | {100*flip[0]:.2f}% | {ci(g)} |')
    rows += ['', '전환율은 같은 물체의 hollow·bulge와 base를 **같은 실제 높이**에서 비교했다. Geometry Δ는 base까지 포함한 전체 물체에서 privileged−original이다. [전체 수치와 panel별 결과](stats43.json)']
    (ROOT/'TABLES_KO.md').write_text('\n'.join(rows)+'\n')


def plot(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained',gridspec_kw={'width_ratios':[1,1.15,1]})
    col = ('#15536b','#008878','#bb5a28')
    dnames = ('Original','Robust','Privileged')
    offs = s['offsets_mm']
    loss = np.asarray(s['loss'])
    for i in range(3):
        ax[0].plot(np.arange(4),loss[1,i,:,0],marker=('o','s','^')[i],color=col[i],label=dnames[i])
    ax[0].set_xticks(np.arange(4),('Zero','Temp.','Fixed','Opt.'))
    ax[0].set_ylabel('Held-out Brier (shifted)')
    ax[0].legend(frameon=True,fontsize=8,loc='center left',bbox_to_anchor=(0.02,.37),ncol=1)
    g = np.asarray(s['geometry_by_offset'])*1000
    ax[1].fill_between(offs,g[:,1],g[:,2],color='#83afa5',alpha=.35)
    ax[1].plot(offs,g[:,0],'o-',color=col[0],ms=4)
    ax[1].axhline(0,color='0.4',lw=.9,ls='--')
    ax[1].set_xlabel('Actual minus nominal height (mm)')
    ax[1].set_ylabel('Projected minus original Brier (×1,000)')
    times=np.asarray(s['timing']['p90'])
    for i in range(3):
        ax[2].plot(np.arange(4),times[i],marker=('o','s','^')[i],color=col[i])
    ax[2].set_xticks(np.arange(4),('Zero','Temp.','Fixed','Opt.'))
    ax[2].set_ylabel('Warm GPU p90 (ms / 512 objects)')
    for a,title in zip(ax, ('A. Correction and decoder','B. Projection cost, zero correction','C. Online latency')):
        a.set_title(title,fontsize=12,loc='left')
        a.grid(axis='y',alpha=.2)
        a.spines[['right','top']].set_visible(False)
    fig.savefig(ROOT/'blind_ood_v43.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    data=json.loads((ROOT/'stats43.json').read_text())
    write_tables(data)
    plot(data)
