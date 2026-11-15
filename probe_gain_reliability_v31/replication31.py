"""Stage-specific paid-only and fresh-object rank/replay tables for the v31 report."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def j(path):
    return json.loads(path.read_text())


def pct(x):
    return f'{100*x:.1f}%'


def show(est, ci, decimals=5, scale=1):
    return f'{est*scale:.{decimals}f} [{ci[0]*scale:.{decimals}f}, {ci[1]*scale:.{decimals}f}]'


def tab(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)] + ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    primary=j(ROOT/'results/crossed_stats31.json')
    matched=j(ROOT/'results_pool/crossed_stats31.json')
    sections=['### 이미 관측한 물체에 접촉만 추가하면',
              '이 보충 실험은 v30과 같은 구입 절차를 사용한다. 위와 **같은 144개 cohort, paid prefix, numerical bank, 73,728개 deployment worlds, common decoder**를 다시 썼다. 각 future는 원래 1,024개 factual pool에서 그 stage에 이미 접촉한 물체를 제외하고 서로 다른 2개 또는 4개를 고른다. 보관된 실제 물체의 latent state에서 새로운 noisy contact response만 받는다. 새 물체의 factual 정보는 추가되지 않는다. 서로 다른 future가 같은 물체를 뽑을 수 있다. 독립성의 대상은 고정한 물체 집단에서의 무작위 선택과 sensor noise다. Fresh-object control을 본 뒤 이 검증을 추가했고, secondary 결과를 확인하기 전에 설계를 기록했다. [보충 검증 설계](MATCHED_PLAN_KO.md)']
    rows=[]
    rank_rows=[]
    replays=[]
    for panel,name,stats in [('results','새 물체 · X+Y',primary),('results_pool','기존 물체 · Y만',matched)]:
        pooled=stats['pooled']
        pp=j(ROOT/panel/'portfolio31.json')
        swaps=j(ROOT/panel/'swap31.json')['pooled']
        for st,stage in enumerate(['2→4','4→8']):
            g=pooled['gain']
            comps=np.array(pooled['components']['estimate'][st])*1.e6
            ranks=pooled['ranks']; uncertainty=stats['primary']['simultaneous95'][st]
            rv=ranks['estimate'][st]
            rows.append([name,stage, show(g['estimate'][st],g['pointwise95'][st]),
                         *[f'{v:.2f}' for v in comps], show(comps[1]-comps[0],np.array(uncertainty[0])*1e6,2)])
            rank_rows.append([name,stage,show(rv[2],np.array(uncertainty[1])+.5,1,100),
                              show(rv[5],ranks['pointwise95'][st][5],1,100),
                              f"{swaps[st][0]:.2f}",pct(swaps[st][1])])
            replays.append([name,stage,*[show(pp['pooled'][st][k],pp['pooled95'][st][k]) for k in [0,1,4]]])
    sections += [tab(['추가 관측','Stage','Signed Brier gain [95%]','Future ×10⁶','Batch ×10⁶','Interaction ×10⁶','Batch − future ×10⁶ [simultaneous 95%]'],rows),
                 tab(['추가 관측','Stage','독립 1×1 순서 일치 % [simultaneous 95%]','독립 2×2 순서 일치 % [pointwise 95%]','교체한 probe 수 / 13','교체 0건인 cell'],rank_rows),
                 '교체 수는 추가 calibration 후 128개 미래 물체 중 원래 probe를 받았던 13개에서 빠진 물체 수다. 같은 수의 다른 물체가 새로 probe를 받는다. 따라서 전체 batch Brier가 거의 변하지 않은 이유가 애초에 action이 같았기 때문인지 확인할 수 있다. 두 panel 모두 저장된 world별 두 action의 Brier 차이와 교체 명단으로 전체 gain을 복원했다. [Fresh-object action 검산](results/swap31.json) · [기존 물체 action 검산](results_pool/swap31.json)',
                 tab(['추가 관측','Stage','한 반복으로 선택·재평가','독립 반복에 평가','V30 frozen scorer'],replays),
                 '마지막 표는 여덟 환경 중 네 곳의 접촉 예산을 배정해 균등 무작위 배분보다 얻은 Brier gain이다. 두 panel이 같은 prefix와 배포 물체를 공유하므로 관측 절차 자체의 차이에 대한 검정은 block을 짝지어 계산했다. [보충 실험의 전체 추정치](results_pool/crossed_stats31.json) · [동일 비용 배분](results_pool/portfolio31.json)']
    levels=[]
    law_names={'coulomb':'Coulomb','split_static':'Static–kinetic split','speed_weak':'Speed weakening'}
    for law,display in law_names.items():
        for path,name in [('results','새 물체'),('results_pool','기존 물체')]:
            blocks=[]
            for folder in sorted((ROOT/path/law).glob('block_*')):
                with np.load(folder/'metrics.npz') as zz:
                    before=zz['before'].mean((1,2))
                    after=zz['after'].mean((1,2,3))
                    blocks.append(np.stack([before,after],axis=1))
            assert len(blocks)==6
            mean=np.mean(blocks,axis=0)
            levels.append([display,name,*[f'{x:.5f}' for x in mean.flatten()]])
    sections.extend([tab(['Contact law','추가 물체','2회 Brier','4회 Brier(2에서 계속)','4회 Brier(고정 prefix)','8회 Brier(4에서 계속)'],levels),
                     '표의 앞뒤 Brier는 같은 stage에서 같은 128개 물체와 decoder로 평가했다. 두 4회 열은 서로 다른 calibration history다. 첫 열의 4회는 2회 prefix에서 독립적으로 분기한 future를 평균했고, 다음 열의 4회는 원래 고정해 둔 실제 4회 prefix로 시작한다.'])
    with np.load(ROOT/'results_pool/coulomb/block_000/metrics.npz') as z:
        g=z['gain'][0,0]
        alpha=z['alpha'][0]
    def format_example(label, row):
        return [label] + [f'{v:+.5f}' for v in [*row,row.mean()]]
    example=[format_example(f'F{i+1}', row) for i,row in enumerate(g)]
    example.append(format_example('열 평균',g.mean(0)))
    sections.extend(['### 같은 paid prefix에서 나온 실제 4×4 표',
                     f'파일 순서상 첫 cohort(Coulomb, block 0, environment 0, coupling {alpha[0]:+.2f}·{alpha[1]:+.2f})의 2→4회 예다. 모든 행은 보유한 물체에 접촉만 추가한다. 같은 행에서 B1과 B2의 차이는 같은 접촉 결과가 다른 배포 집단을 만났을 때의 차이다. 같은 열에서 F1과 F2를 비교하면 어느 물체에 추가 접촉했는지만 달라진다.',
                     tab(['독립 calibration future','Batch 1','Batch 2','Batch 3','Batch 4','행 평균'],example),
                     '표의 단위는 signed Brier gain이다. 같은 행의 네 값이나 같은 열의 네 값은 하나의 축을 공유한다. [해당 cohort의 raw gain](results_pool/coulomb/block_000/metrics.npz)'])
    paired=j(ROOT/'results_pool/paired_pool31.json')
    rows=[]
    for st,stage in enumerate(['2→4','4→8']):
        for idx,name,sc,dec in [(0,'평균 Brier gain',1,5),(1,'Batch − future variance ×10⁶',1e6,2),
                                (2,'독립 1×1 순서 일치율(%p)',100,2),(3,'독립 2×2 순서 일치율(%p)',100,2)]:
            rows.append([stage,name,show(paired['paired_difference'][st][idx],paired['simultaneous95'][st][idx],dec,sc)])
    sections.extend(['### 두 구입 방식의 paired 차이',
                     tab(['Stage','비교 지표','기존 물체 − 새 물체 [8-contrast simultaneous 95%]'],rows),
                     '이 차이의 interval이 0을 포함하면 관측 방식의 차이를 검출하지 못했다는 뜻이다. 두 방식이 동등하다고 판정하려면 실용적 margin을 먼저 정하고 equivalence를 검정해야 한다. [분석 계획](POOL_STATS_PLAN_KO.md) · [Paired 추정 및 sensitivity](results_pool/paired_pool31.json)'])
    (ROOT/'MATCHED_RESULTS_KO.md').write_text('\n\n'.join(sections)+'\n')

if __name__=='__main__':
    main()
