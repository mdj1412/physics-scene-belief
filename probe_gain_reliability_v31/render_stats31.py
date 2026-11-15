"""Korean display tables for v31; estimates raw, rank and gain units kept distinct."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
LAWS={'coulomb':'Coulomb','split_static':'Static–kinetic split','speed_weak':'Speed weakening'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
                     +['| '+' | '.join(map(str,row))+' |' for row in rows])


def bounds(point,interval,scale=1,places=2):
    return f'{point*scale:.{places}f} [{interval[0]*scale:.{places}f}, {interval[1]*scale:.{places}f}]'


def main():
    j=json.loads((R/'crossed_stats31.json').read_text())
    z=j['pooled'];p=j['primary']
    def get(k):return np.asarray(z[k]['estimate']),np.asarray(z[k]['pointwise95'])
    comps,cis=get('components')
    shares,_=get('component_shares')
    rank,ri=get('ranks')
    gs,gb=get('gain')
    diff,df=get('abs_repeat_difference')
    out,oi=get('practical_discrepancy')
    grid,gc=get('grid_var')
    stages=['2→4','4→8']
    parts=['## 교차 반복 실험의 통계 결과',
           '각 stage에서는 각 환경·prefix 안에서 먼저 variance와 gain을 계산하고 세 contact laws에 같은 가중치를 주었다. Gain이 양수이면 추가 관측으로 Brier가 줄었다. Variance는 Brier² 단위이고 SD는 Brier 단위다.',
           '### 주 분석 · 어느 변동이 더 컸고 독립 순위가 재현됐나',
           table(['Stage','Batch−future variance ×10⁶ [simultaneous 95%]','양 축을 모두 바꾼 순위 일치율 [simultaneous 95%]','평균 signed gain [pointwise 95%]'],[
               [stage,bounds(p['estimate'][s][0],p['simultaneous95'][s][0],1e6),
                bounds(.5+p['estimate'][s][1],np.asarray(p['simultaneous95'][s][1])+.5,1,3),bounds(gs[s],gb[s],1,5)]
               for s,stage in enumerate(stages)]),
           '네 대비 전체를 동시에 포함하는 구간이다. Rank concordance의 우연 기준은 0.5다. 표본은 law당 independent blocks 6개이며 각 block에서 8개 환경을 비교했다. Interval은 미리 정한 세 law·여덟 population 조건과 현재 simulator에 조건부다.',
           '### 단일 Brier gain의 variance decomposition',
           table(['Stage','Component','Variance ×10⁶ [pointwise 95%]','단일 관측 분산의 비중','Component SD (Brier)'],
                 [[stages[s],name,bounds(comps[s,k],cis[s,k],1e6),f'{100*shares[s,k]:.1f}%',
                   f'{np.sqrt(comps[s,k]):.5f}' if comps[s,k]>=0 else '음수 MOM 추정']
                  for s in range(2) for k,name in enumerate(['Calibration future','Deployment batch','Future × batch'])]),
           '각 prefix의 F×B 표에서 row/column/grand mean을 구해 두 main effects와 interaction을 나누었다. 어떤 future로 posterior가 갱신됐는지에 따라 *특정 batch*의 action과 이득이 달라질 수 있다. 이 부분은 두 main effects 사이에 임의로 나눠 담지 않았다. Prefix를 고정하고 한 축만 새로 뽑을 때의 재측정 분산은 각각 V_future+V_interaction 및 V_batch+V_interaction이다. 양쪽 모두를 다시 뽑은 한 cell의 분산은 세 성분의 합이다. 별도 반복을 갖지 않는 계산 오차가 존재했다면 interaction과 따로 식별할 수 없다. Main effect의 추정값이 음수가 될 수 있는 unrestricted method-of-moments estimate를 원자료에 그대로 보존했다.',
           '### 같은 환경 여덟 곳의 순위 재현성',
           table(['독립적으로 바꾼 정보 · 두 번째 순위','2→4 concordance [95%]','4→8 concordance [95%]'],[
               [label,*[bounds(rank[s,i],ri[s,i],1,3) for s in range(2)]] for i,label in enumerate([
                   '같은 future, 새 batch','새 future, 같은 batch','새 future, 새 batch','둘 다 새로, 각 순위에 1 future × 2 batches',
                   '둘 다 새로, 각 순위에 2 futures × 1 batch','둘 다 새로, 각 순위에 2 futures × 2 batches'])]),
           '환경 쌍의 순서가 같으면 1점, 반대면 0점, 동점은 0.5점이다. 각 환경에서 가능한 disjoint replicate를 서로 독립적으로 열거했다. 같은 future 또는 같은 batch를 쓴 행은 공유한 정보가 있는 상황의 일치율이다. 평균을 내는 세 행에서는 두 순위가 어떤 future도, 어떤 batch도 공유하지 않는다. Averaging은 개별 환경의 population 불변성을 보장하지 않는다. 같은 환경에서 같은 prefix까지 조건을 맞춘 순위다.',
           '### 반복을 어디에 늘리면 환경 평균을 더 정밀하게 추정하는가',
           table(['반복당 futures × batches','조건부 gain 평균의 예상 SD · 2→4','조건부 gain 평균의 예상 SD · 4→8'],[
               [f'{f} × {b}',*[f'{np.sqrt(grid[s,k]):.5f}' if grid[s,k]>=0 else '음수 분산 추정' for s in range(2)]]
               for k,(f,b) in enumerate(j['grids'])]),
           '이 값은 V_future/F + V_batch/B + V_interaction/(F·B)에 해당하는 variance를 환경과 법칙에 걸쳐 평균한 뒤 square root를 취했다. 독립적으로 얻는 환경 한 곳의 prefix-conditional mean gain label의 sampling uncertainty이다. 전체 논문 평균에 대한 standard error와 다르다. 두 설계의 cell 수가 같아도 센서 구입·reset 비용과 동일 future를 재사용하는 GPU 비용은 다를 수 있다.',
           table(['Stage','독립 두 cell gain의 평균 절대 차이','절대 차이 >0.003인 비율 [95%]','환경 평균 관측 분산 ×10⁶','반복오차 보정 후 환경 간 분산 ×10⁶ [95%]'],
                 [[stages[s],f'{diff[s,2]:.5f}',bounds(out[s,2],oi[s,2],100,1)+'%',f"{1e6*z['between_observed']['estimate'][s]:.2f}",
                   bounds(z['between_corrected']['estimate'][s],z['between_corrected']['pointwise95'][s],1e6)]
                  for s in range(2)]),
           '같은 환경의 두 독립 gain 사이 절대 차이는 Brier 감소량 자체의 크기가 아니다. 마지막 열은 한 block 안 여덟 환경의 4×4 평균 사이 표본분산에서 평균 label의 sampling variance를 뺀 추정값이다. 환경당 관측한 서로 다른 paid prefix는 하나이므로 population 특성과 prefix noise를 이 열에서 더 분리할 수 없다.',
           '### 법칙별 결과',
           table(['Law','Stage','Future / batch / interaction variance ×10⁶','독립 concordance [pointwise 95%]','평균 signed gain [pointwise 95%]'],[
               [LAWS[law],stages[s],' / '.join(f'{v*1e6:.2f}' for v in j['per_law']['components']['estimate'][l][s]),
                bounds(j['per_law']['ranks']['estimate'][l][s][2],j['per_law']['ranks']['pointwise95'][l][s][2],1,3),
                bounds(j['per_law']['gain']['estimate'][l][s],j['per_law']['gain']['pointwise95'][l][s],1,5)]
               for l,law in enumerate(j['laws']) for s in range(2)]),
           'Estimator 식, 재표집 단위와 사전 판정 기준은 통계 계획에 기록했다. Additive known-effects, tie handling, estimator identity, 축과 index 교환에 대한 검산을 compute node의 CPU에서 통과했다. [사전 통계 계획](STATISTICAL_PLAN_KO.md) · [전체 수치](results/crossed_stats31.json) · [분석 코드](stats31.py) · [검산 코드](check_stats31.py)']
    extra_file=R/'practical31.json'
    if extra_file.is_file():
        extra=json.loads(extra_file.read_text())
        value=np.array(extra['pooled'])
        parts.extend([
            '### 관측된 개선의 부호와 크기도 재현되는가',
            table(['단일 관측의 개선 조건','Stage','임의 첫 cell에서 충족','첫 cell 충족 후 두 축이 독립인 다음 cell에서 다시 충족'],
                  [[k,stages[s],f'{100*value[s,i]:.1f}%',f'{100*value[s,i+1]:.1f}%']
                   for k,i in [('Gain > 0',0),('Gain > 0.003',2)] for s in range(2)]),
            '조건부 확률의 분모는 첫 번째 결과가 기준을 충족한 사례다. 여기에 같은 paid prefix를 유지하면서 future와 batch를 모두 독립적으로 뽑은 다음 cell을 대응했다. 유리한 단일 결과를 먼저 본 상황에서 그 판단이 다시 성립하는지 묻는 탐색적 진단이다. [실질 개선의 반복 확률과 구간](results/practical31.json)'])
    (ROOT/'STATS_KO.md').write_text('\n\n'.join(parts)+'\n')


if __name__=='__main__':
    main()
