"""Generate the Korean report and a scientific figure from the audited server artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
def read(name):
    return json.loads((RESULTS / name).read_text())

def table(head, rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'---|'*len(head)] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])

def interval(estimate, ci, decimals=5, scale=1):
    return f'{estimate*scale:.{decimals}f} [{ci[0]*scale:.{decimals}f}, {ci[1]*scale:.{decimals}f}]'

def p(x):
    return f'{100*x:.1f}%'

def main():
    data = read('crossed_stats31.json')
    replay = read('portfolio31.json')
    d = data['pooled']
    t = lambda k: np.array(d[k]['estimate'])
    ci = lambda k: np.array(d[k]['pointwise95'])
    stages = ['2→4회', '4→8회']
    laws = ['Coulomb', 'Static–kinetic split', 'Speed weakening']
    comp = t('components')
    bands = np.array(data['primary']['simultaneous95'])
    rank = t('ranks')
    rank_ci = ci('ranks')
    src=['한 future 고정, 새 batch','한 batch 고정, 새 future','두 축 모두 새로 추출 · 1×1',
         '독립 1×2 평균끼리','독립 2×1 평균끼리','독립 2×2 평균끼리']
    rows=[]
    for k,stage in enumerate(stages):
        for j,var in enumerate(['Calibration future','Deployment batch','Future × batch']):
            rows.append([stage,var,interval(comp[k,j], ci('components')[k,j],2,1e6),
                         p(t('component_shares')[k,j]),f'{np.sqrt(comp[k,j]):.5f}' if comp[k,j]>=0 else '—'])
    rank_rows=[]
    for j,label in enumerate(src):
        row=[label]
        for k in range(2):
            bounds=bands[k,1]+.5 if j==2 else rank_ci[k,j]
            row.extend([interval(rank[k,j],bounds,1,100),f"{t('abs_repeat_difference')[k,j]:.5f}",
                        p(t('practical_discrepancy')[k,j])])
        rank_rows.append(row)
    pval = np.asarray(replay['pooled'])
    p_ci = np.asarray(replay['pooled95'])
    selection_rows = [[label,*[interval(pval[k,j],p_ci[k,j]) for k in range(2)]] for j,label in enumerate([
        '1×1 · 선택에 사용한 결과에 재평가','1×1 · 양쪽이 독립인 결과에 평가','2×2 · 선택에 사용한 평균에 재평가',
        '2×2 · 양쪽이 독립인 평균에 평가','V30 frozen scorer · 4×4 평균으로 채점','Uncertainty-only · 4×4 평균으로 채점'])]
    perlaw=[]
    for i,law in enumerate(laws):
        for k,stage in enumerate(stages):
            v=data['per_law']['components']['estimate'][i][k]
            l=data['per_law']['ranks']['estimate'][i][k][2]
            b=data['per_law']['ranks']['pointwise95'][i][k][2]
            g=data['per_law']['gain']
            perlaw.append([law,stage,interval(g['estimate'][i][k],g['pointwise95'][i][k]),
                           *[f'{x*1e6:.2f}' for x in v],interval(l,b,1,100)])
    formula='Var(mean gain) = V_F/n_F + V_B/n_B + V_I/(n_F·n_B)'
    parts=['# v31 · 같은 관측에서 출발해도 추가 probe의 가치는 얼마나 재현되는가?',
           (ROOT/'FINDINGS_KO.md').read_text().strip(),
           '## 1. 같은 질문을 독립적으로 반복하는 방법',
           (ROOT/'DESIGN_KO.md').read_text().strip(),
           '## 2. 어느 축이 실제 Brier gain을 흔드는가?',
           '''각 paid prefix에서 4×4 gain matrix를 만들었다. 같은 행의 네 cell은 calibration 결과가 같고, 같은 열의 네 cell은 평가하는 물체·관측·정답이 같다. Row mean과 column mean으로 설명한 뒤 남는 교차 변동이 interaction이다. 항을 먼저 각 prefix 안에서 추정한 뒤 여덟 환경, 독립 blocks, 세 law에 동일 가중치를 주었다. 환경 자체의 물성 분포 차이는 이 표의 variance에 섞지 않았다.''',
           table(['Stage','변동 요인','Variance ×10⁶ Brier² [95%]','전체 변동의 비중','Component SD (Brier)'],rows),
           table(['Stage','Batch − future variance ×10⁶ [simultaneous 95%]','전체 1×1 SD','4×4 평균 SD','평균 signed gain [pointwise 95%]'],
                 [[stage,interval(comp[k,1]-comp[k,0],bands[k,0],2,1e6),f"{np.sqrt(t('grid_var')[k,0]):.5f}",f"{np.sqrt(t('grid_var')[k,-1]):.5f}",interval(t('gain')[k],ci('gain')[k])] for k,stage in enumerate(stages)]),
           '''Component SD는 각 variance의 제곱근이므로 서로 더할 수 없다. 마지막 표의 SD는 한 환경에서 반복 표집한 gain 또는 그 평균의 조건부 표준편차를 variance 평균의 제곱근으로 요약한 값이다. 전체 실험의 평균 gain에 대한 confidence interval과는 단위가 같아도 질문이 다르다. 한 future를 고정하고 batch만 바꿔 생기는 변동에는 batch main effect와 interaction이 모두 들어간다. Future를 바꿀 때도 같은 interaction이 들어간다. 따라서 interaction을 양쪽에 중복 배분하거나 임의로 반분하지 않았다.''',
           '### 새 반복 하나를 더 구입한다면',
           f"Crossed random-effects 모형에서 {formula}이다. 아래는 동일한 수의 future×batch cell을 조사하면서 배치를 다르게 한 경우다. 각 열의 값은 조건부 RMS 오차(Brier) 추정치다. 한 batch는 128개 world, 한 future는 2개 또는 4개의 새로운 probe이므로, sensor·GPU 비용이 같다는 뜻은 아니다.",
           table(['(Futures, batches)','2→4 RMS','4→8 RMS'],[[f'{g[0]} × {g[1]}',*[f'{np.sqrt(max(x,0)):.5f}' for x in t('grid_var')[:,j]]] for j,g in enumerate(data['grids'])]),
           '## 3. 환경 순위도 재현되는가?',
           '''같은 law의 여덟 환경에서 둘을 골라 어느 쪽의 gain이 더 높은지 판단한 후 다른 반복에서 순서가 유지되는지 계산했다. 각 환경은 독립적인 future 및 batch를 가지므로 환경 쌍마다 허용되는 반복 조합을 독립적으로 열거했다. 동률은 0.5, 무작위 순서의 기준은 50%다. 아래의 |Δgain|과 0.003 초과율은 한 prefix에서 반복 평가가 서로 얼마나 다른지를 나타낸다.''',
           table(['반복 비교','2→4 순서 일치 % [95%]','평균 |Δgain|¹','0.003 초과','4→8 순서 일치 % [95%]','평균 |Δgain|¹','0.003 초과'],rank_rows).replace('평균 |Δgain|¹','평균 절대 gain 차이'),
           '''두 축이 모두 새로운 1×1 순서 비교와 앞 절의 variance 차이, 각 두 stage를 묶어 총 4개 primary contrast에 simultaneous 95% interval을 계산했다. 나머지는 별도 표기 없는 한 pointwise interval이다. 세 law에서 독립 block 여섯 개씩을 law 내에서 paired bootstrap 10,000회 재표집했다. 28개 환경 쌍, noise 32회, 반복 조합들은 별도의 독립 실험이 아니다. 법칙당 block 수가 여섯 개여서 특히 law별 interval의 꼬리는 조심해서 해석해야 한다. [통계 계획](STATISTICAL_PLAN_KO.md) · [통계 코드](stats31.py) · [전체 추정치와 구간](results/crossed_stats31.json)''',
           table(['Contact law','Stage','평균 gain [95%]','Future ×10⁶','Batch ×10⁶','Interaction ×10⁶','독립 1×1 순서 일치 % [95%]'],perlaw),
           '표 안의 음수 variance component가 있다면 제한을 두지 않은 method-of-moments 추정의 유한 표본 오차다. 추정치를 0으로 바꿔 우세한 축을 결정하지 않았다.',
           (ROOT/'PRIMARY_INTERPRETATION_KO.md').read_text().strip(),
           '## 4. 그 순위로 실제 probe를 배분하면',
           '''각 stage에서 후보 8곳 중 4곳에만 같은 수의 probe를 더 준다고 가정했다. 표는 무작위로 4곳을 고를 때의 정확한 기대 gain에 **더해지는** Brier 감소량이다. 선택한 4곳에서 얻은 gain을 전체 8곳으로 나눴다. 양수면 같은 비용에 이익이다. 1×1 또는 2×2로 표시한 행은 미래 결과를 먼저 본 사후 진단이다. 반면 frozen scorer와 uncertainty-only는 paid prefix만 보고 순위를 정한다. 이는 두 stage를 각각 떼어 본 실험이어서 v30 전체 stopping policy의 Brier로 읽을 수 없다.''',
           table(['네 환경을 선택한 방법과 평가 자료','2→4회 추가 배분 이득 [95%]','4→8회 추가 배분 이득 [95%]'],selection_rows),
           f"독립 반복으로 고른 상위 네 환경의 평균 겹침 비율은 1×1에서 {p(pval[0,6])}·{p(pval[1,6])}, 2×2에서 {p(pval[0,7])}·{p(pval[1,7])}였다(2→4·4→8 순서). 서로 무작위로 네 곳을 고르면 평균 50%가 겹친다. [사전 보조 분석 설계](PORTFOLIO_PLAN_KO.md) · [재현 코드](portfolio31.py) · [보조 결과](results/portfolio31.json)",
           '''별도 질문으로, 같은 population coupling이라도 factual pool·paid prefix·numerical bank를 모두 새로 뽑아서 순위를 매겼다. 각 독립 block 안에서 4×4 gain을 평균하고 같은 여덟 coupling을 대조했다. 아래는 6개 독립 block을 나눈 기술 통계다. 새 prefix까지 일반화하는 질문이라 앞 절의 동일 prefix 재현성과 구분한다.''',
           table(['동일 coupling을 새 prefix에서 재측정','2→4회 순서 일치율','4→8회 순서 일치율'],[[n,*[p(x) for x in np.array(replay['independent_prefix_pooled'])[:,j]]] for j,n in enumerate(['1개 block 대 독립 1개','3개 평균 대 겹치지 않는 3개 평균'])]),
           (ROOT/'CONTROL_KO.md').read_text().strip(),
           (ROOT/'MATCHED_RESULTS_KO.md').read_text().strip(),
           (ROOT/'ENV_RANK_KO.md').read_text().strip(),
           '![추가 calibration probe의 변동 요인과 환경별 순위 재현성](results/v31.png)',
           '## 6. 논문 기여 가능성, 한계와 다음 결정',(ROOT/'ASSESSMENT_KO.md').read_text().strip(),
           (ROOT/'LITERATURE_KO.md').read_text().strip(),
           (ROOT/'CONTACT_STUDIES_KO.md').read_text().strip(),
           '## 재현 자료',(ROOT/'PROVENANCE_KO.md').read_text().strip()]
    save_fig(comp,rank,rank_ci,bands)
    (ROOT/'REPORT_KO.md').write_text('\n\n'.join(parts)+'\n')


def save_fig(components, ranks, ranks_ci,bands):
    matched=read_pool('crossed_stats31.json')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.9),layout='constrained')
    ax,order=axes
    colors=['#4e829e','#cc8f4a','#578568']
    labels=['Calibration future','Deployment batch','Future × batch']
    values=np.concatenate([components,matched['pooled']['components']['estimate']])*1e6
    x=np.array([0,1,2.4,3.4])
    for i,color in enumerate(colors):
        ax.bar(x+(i-1)*.23,values[:,i],width=.23,color=color,label=labels[i])
    ax.axhline(0,c='0.4',lw=.7)
    ax.set(xticks=x,xticklabels=['2 to 4\nNew','4 to 8\nNew','2 to 4\nPool','4 to 8\nPool'],
           ylabel='Variance (Brier² × 1,000,000)',title='What changes the measured gain?')
    ax.set_ylim(0,values.max()*1.24)
    ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper center",columnspacing=1.0,handlelength=1.4)
    matched_rank=np.asarray(matched['pooled']['ranks']['estimate'])
    matched_ci=np.asarray(matched['pooled']['ranks']['pointwise95'])
    mbands=np.asarray(matched['primary']['simultaneous95'])
    for j,(vals,ci,band) in enumerate([(ranks,ranks_ci,bands),(matched_rank,matched_ci,mbands)]):
        for i,col in enumerate(['#306c89','#aa632d']):
            idx=[0,1,2,5]
            ys=np.arange(4)+(j*2+i-1.5)*.105
            pt=vals[i,idx]
            lohi=ci[i,idx].copy()
            lohi[2]=band[i,1]+.5
            order.errorbar(pt,ys,xerr=np.maximum(0,np.stack([pt-lohi[:,0],lohi[:,1]-pt])),
                           fmt=('o' if j==0 else 's'),mfc=(col if j==0 else 'white'),
                           ms=4.5,capsize=2,elinewidth=.9,
                           label=f"{['New object','Known pool'][j]} · {['2 to 4','4 to 8'][i]}",color=col)
    order.axvline(.5,ls='--',c='0.5',lw=.8)
    order.set(yticks=np.arange(4),yticklabels=['New batch only','New future only','Both new (1×1)','Both new (2×2)'],
              xlabel='Pairwise concordance',title='Does the order of two environments survive?')
    order.set_ylim(-.76,3.8)
    order.legend(frameon=False,fontsize=7.4,ncol=2,loc='best')
    fig.savefig(RESULTS/'v31.png',dpi=160)
    plt.close(fig)


def read_pool(name):
    return json.loads((ROOT/'results_pool'/name).read_text())


if __name__=='__main__':
    main()
