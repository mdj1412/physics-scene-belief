"""Render the matched sensitivity without altering confirmatory fresh-world outputs."""
import json
from pathlib import Path
import numpy as np
from render_stats31 import bounds, table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_pool'


def main():
    read=lambda s:json.loads((OUT/s).read_text())
    panel=read('crossed_stats31.json')
    diff=read('paired_pool31.json')
    rows=[]
    for stage,st in enumerate(('2→4','4→8')):
        c=panel['pooled']['components']['estimate'][stage]
        s=panel['pooled']['component_shares']['estimate'][stage]
        d=panel['primary']['estimate'][stage]
        ci=np.asarray(panel['primary']['simultaneous95'][stage])
        rows.append([st,*[f'{x*1e6:.2f} ({100*y:.1f}%)' for x,y in zip(c,s)],
                     bounds(d[0],ci[0],1e6,2),bounds(.5+d[1],.5+ci[1],100,1)+'%'])
    comparisons=[]
    for stage,st in enumerate(('2→4','4→8')):
        for j,(name,sc,dec) in enumerate([('평균 Brier gain',1,5),('Batch−future variance ×10⁶',1e6,2),
                                           ('1×1 concordance (%), 차이는 %p',100,2),
                                           ('2×2 concordance (%), 차이는 %p',100,2)]):
            lo=diff['leave_one_difference']['leave_one_block_min'][stage][j]
            hi=diff['leave_one_difference']['leave_one_block_max'][stage][j]
            comparisons.append([st,name,f"{diff['pooled_fresh'][stage][j]*sc:.{dec}f}",
                                f"{diff['pooled_pool'][stage][j]*sc:.{dec}f}",
                                bounds(diff['paired_difference'][stage][j],diff['simultaneous95'][stage][j],sc,dec),
                                f'[{lo*sc:.{dec}f}, {hi*sc:.{dec}f}]'])
    parts=['### 원래 1,024개 물체 안에서 추가 접촉 대상을 고른 보조 실험',
        '별도의 18개 law-block 계산이 주 결과의 factual pool, paid prefix, bank, deployment worlds와 decoder Brier를 그대로 공유한다. 추가 대상은 처음 2개 또는 4개 paid IDs를 제외한 원래 pool에서 뽑는다. 한 future 안에서 중복은 없고 서로 다른 future에서는 같은 물체가 다시 선택될 수 있다. 각 draw는 별도의 probe noise를 사용한다. 따라서 이 조건에서는 추가 접촉과 동시에 새로운 factual 입력을 구입하지 않는다. 순위와 분산은 같은 고정 표본·paid prefix에 조건부다.',
        table(['Stage','Future variance ×10⁶ (비중)','Batch variance ×10⁶ (비중)','Interaction ×10⁶ (비중)',
               'Batch−future variance ×10⁶ [95%]','1×1 독립 concordance [95%]'],rows),
        '보조 panel 내부의 두 stage·두 질문에 separate simultaneous 95% interval을 사용했다. 새 물체와 finite pool의 paired 차이는 아래처럼 같은 law-block을 두 panel에서 함께 재표집해 계산했다.',
        table(['Stage','대상','Fresh','Pool','Pool − fresh [8-contrast simultaneous 95%]','단일 block 제외 paired 차이 범위'],comparisons),
        '두 방법 모두 물체를 추출하지만 fresh는 새 물체와 그 factual 입력·probe 응답을 함께 더하고, pool은 이미 본 factual 기록을 선택해 추가 probe만 관측한다. 두 실험의 estimand는 다르다. 차이의 구간에 0이 포함되어도 두 방법의 동등성이 입증되는 것은 아니다. [사전 paired 분석 설계](POOL_STATS_PLAN_KO.md) · [Pool 추정량](results_pool/crossed_stats31.json) · [Paired 차이와 law/block sensitivity](results_pool/paired_pool31.json) · [CPU 검산 코드](paired_pool31.py)']
    (ROOT/'POOL_STATS_KO.md').write_text('\n\n'.join(parts)+'\n')
if __name__=='__main__':
    main()
