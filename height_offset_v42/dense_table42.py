"""Describe the first supported sign transition without assuming monotonicity."""
import json
import numpy as np
from stats42 import ROOT,FAMILY,HEIGHT
def main():
    data=json.loads((ROOT/'dense42.json').read_text())
    main=json.loads((ROOT/'stats42.json').read_text()); offsets=np.array(data['offset_mm'])
    ms,ls,us=[np.array(data['curve'][k]) for k in ('mean','lo','hi')]
    o=np.array(main['offset_mm'])
    idx=np.array([np.abs(offsets-x).argmin() for x in o])
    np.testing.assert_allclose(ms[:,:,idx],main['group_geometry']['mean'],atol=3e-12,rtol=0)
    lines=['## 0.025 mm 간격에서 손익이 바뀌는 범위','',
           '사전 지정한 아홉 점 사이의 경계를 좁히기 위해 같은 held-out 물체를 81개 높이에서 다시 채점했다. 모델의 새 학습이나 독립 재현은 없었다. 4개 shape·height × 81개 높이의 324개 구간을 함께 보정했다. 각 행은 0에서 해당 방향으로 움직이면서 최초로 손해가 확인된 점과 그 전에 가장 가까운 이득이 확인된 점이다. 마지막 열은 최초 손해 이후 더 멀리 가서 다시 이득이 확인됐는지를 나타낸다. 수치는 장치의 보편적 허용오차가 아니다.','',
           '| Shape / 명목 높이 | 이동 방향 | 마지막 확인 이득 (mm) | 최초 확인 손해 (mm) | 이후 확인된 이득 복귀 |','|---|---|---:|---:|---|']
    summary={}
    for i,f in enumerate(FAMILY):
        for j,h in enumerate(HEIGHT):
            for sign in (-1,1):
                coords=np.flatnonzero(offsets*sign>=0)
                coords=coords[np.argsort(abs(offsets[coords]))]
                loss=ls[i,j,coords]>0; gain=us[i,j,coords]<0
                ids=np.flatnonzero(loss)
                first=int(ids[0]) if ids.size else len(coords)
                prior=np.flatnonzero(gain[:first])
                later=np.flatnonzero(gain & (np.arange(len(gain))>first))
                last_help=offsets[coords[prior[-1]]] if prior.size else None
                first_harm=offsets[coords[first]] if ids.size else None
                restored=offsets[coords[later[0]]] if later.size else None
                val=lambda v: '확인 안 됨' if v is None else f'{v:+.3f}'
                name=f'{f}_{h}_{sign:+d}'
                summary[name]=dict(help=last_help,harm=first_harm,return_to_help=restored)
                lines.append(f"| {f} / {h[1:]} mm | {'위 (+)' if sign>0 else '아래 (−)'} | {val(last_help)} | {val(first_harm)} | {val(restored)} |")
    lines+=['','원래 아홉 점에서는 독립 물리 식의 label을 주 저장값과 개별 물체마다 대조했고, 여기서 재계산한 projection Brier 차이도 주 분석과 일치했다. [미세 scan 전체 구간](dense42.json) · [재계산 코드](dense42.py)']
    (ROOT/'DENSE_KO.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'dense_threshold42.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
