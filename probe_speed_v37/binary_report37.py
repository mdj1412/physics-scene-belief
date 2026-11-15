"""Summarize the pair-conditioned weak/null classification diagnostic."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
def band(x):return f'[{x[0]:+.4f}, {x[1]:+.4f}]'
def main():
    p=json.loads((ROOT/'binary37.json').read_text())
    lines=['## 두 surface만을 후보로 가정했을 때의 판별력','',
      '일반 posterior mean strength는 strong, weak, local-null, global-null을 모두 고려한다. 이 요약값이 weak와 local-null의 작은 차이를 희석하는지 검사하기 위해 후보를 둘로 좁힌 source likelihood ratio를 계산했다. 각 class에서 네 source family와 양쪽 orientation을 적분했다. 어떤 표면인지 정답은 사용하지 않지만 가능한 상태가 둘뿐이라는 정보는 추가로 제공한다. 두 점수는 같은 paid observations를 보았다.','',
      '| Speed | Posterior mean strength AUROC | Weak/null log likelihood ratio AUROC | 두 값의 차이 [95%] |','|---|---:|---:|---:|']
    for j,s in enumerate(p['speeds']):
        avg=p['auroc'][j]
        lines.append(f'| {s:.2f} | {avg[0]:.4f} | {avg[1]:.4f} | {p["method_change"][j]:+.4f} {band(p["simultaneous_method_ci"][j])} |')
    delta=p['specialized_speed_change'];ci=p['simultaneous_speed_ci']
    lines+=['',f'두 후보 진단에서 0.50 대비 AUROC 변화는 0.75에서 {delta[0]:+.4f} {band(ci[0])}, 1.10에서 {delta[1]:+.4f} {band(ci[1])}였다. Object–surface·첫 방향을 층화한 paired bootstrap으로 32개 seed block을 12,000회 재표집했고, 두 speed change에 simultaneous interval을 적용했다. 이 진단에서는 correction action과 GPU p90을 재측정하지 않았다. [계산 및 구간](binary37.json) · [분석 코드](binary37.py)']
    (ROOT/'BINARY_KO.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
