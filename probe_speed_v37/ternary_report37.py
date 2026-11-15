"""Document v37's source-only three-action diagnostic and its uncertainty."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent

def fmt(x):return f'{x:+.5f}'
def band(ci):return '['+', '.join(map(fmt,ci))+']'
def pct(x):return f'{100*x:.1f}%'

def main():
    d=json.loads((ROOT/'ternary37.json').read_text())
    v=d['scopes']['local_equal']
    lines=['## 같은 source gain 표에서 shrinkage를 세 번째 행동으로 허용한 진단','',
        'Primary gate는 예측한 full correction gain이 0을 넘으면 전체 보정을 적용하고, 그렇지 않으면 원래 physics probability를 반환한다. 보조 정책은 같은 source gain 표와 posterior에서 z, 0.325z, 0의 gain을 각각 읽어 가장 큰 행동을 선택했다. Threshold나 coefficient를 target에서 조정하지 않았다. 후보마다 적용한 관측, deployment action, 원래 probability가 같다. 선택이 동률이면 고정 shrinkage를 우선했다.','',
        '| Speed | 3-action Brier | 3-action − fixed [95%] | 3-action − binary [95%] | Off / fixed / full 선택률 |',
        '|---|---:|---:|---:|---|']
    for i,s in enumerate(d['speeds']):
        m=v['means'][i]; effect=v['difference'][i]; ci=v['simultaneous_ci'][i]
        lines.append('| '+' | '.join([f'{s:.2f}',f'{m[0]:.5f}',fmt(effect[0])+' '+band(ci[0]),
                         fmt(effect[1])+' '+band(ci[1]),' / '.join(pct(m[k]) for k in (4,3,5))])+' |')
    lines.extend(['', '세 local regime를 같은 비중으로 두었다. 세 속도와 두 baseline의 6개 차이에 simultaneous 95% paired interval을 적용했다. 평균은 물체와 표면 조합을 고정하고 32개 독립 block 안에서 paired orientation과 sensor noise를 공유한다.','',
      '| Surface | 0.50에서 3-action − fixed | 0.75에서 3-action − fixed | 1.10에서 3-action − fixed |',
      '|---|---:|---:|---:|'])
    for name in ('strong','weak','null','weak_null','global_null'):
        case=d['scopes'][name]
        rows=[fmt(a[0])+' '+band(b[0]) for a,b in zip(case['difference'],case['simultaneous_ci'])]
        lines.append('| '+' | '.join([name,*rows])+' |')
    x=v['absolute_speed_change']; ci=v['absolute_speed_ci']
    did=v['difference_in_differences']; di=v['simultaneous_speed_ci']
    lines+=['',f'3-action 자체의 .75−.50 Brier 변화는 {fmt(x[0])} {band(ci[0])}, 1.10−.50은 {fmt(x[1])} {band(ci[1])}였다. 고정 32.5% 대비 이점의 속도별 변화(difference-in-differences)는 {fmt(did[0][0])} {band(di[0][0])} 및 {fmt(did[1][0])} {band(di[1][0])}였다. 이 표에서 속도별 쌍의 음수 차이는 빠른 쪽이 낮은 loss를 의미한다.','',
      '이 진단은 source model의 expected Brier gain이 더 큰 행동을 고른다는 규칙을 그대로 사용한다. Shrinkage가 후보에 들어가도 source utility의 순위와 target outcome의 관계가 달라질 수 있다. 이 정책의 latency를 별도로 측정하지 않았다. [수치와 interval](ternary37.json) · [평가 코드](binary37.py)']
    (ROOT/'TERNARY_KO.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
