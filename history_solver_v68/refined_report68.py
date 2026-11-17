"""Write the terminal-grid sensitivity appendix after the forecast disclosure gate."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent

def entry(a):
    return f'{a[0]:+.6f} [{a[1]:+.6f}, {a[2]:+.6f}]'
def main():
    d=json.loads((H/'refined68.json').read_text());out=d['summary']
    text=['# v68 · 사후 target 해상도 재검증','',
          '이 실험에서는 standard history로 얻은 동일한 봉인 forecast를 사용했다. 사전에 지정했던 32/128 표본의 정밀도 검산에서 일부 부호가 바뀌어 결과를 본 뒤 확대했다. Compliant terminal solver를 12개 catalog 전체에서 9 sites → 17 sites로, time step을 약 4.63 → 2.31 μs로 바꿨다. 각 environment의 첫 8개 물체에서는 33 sites 및 약 1.16 μs 계산까지 수행했다. 모든 환경·law·shape·surface와 radius를 유지했다. 기존 32개 물체의 fine slip speed와 label, 미분리 상태를 전수 bit 단위로 다시 맞췄다. [추가 계획](REFINEMENT_KO.md) · [계산 코드](refined68.py) · [채점 코드](refined_score68.py)','',
          'Full fine label을 계산하는 동안 별도 fine-history forecast를 만들었다. 양쪽의 seal이 생성된 후에만 여기서 전체 결과를 채점했다. 첫 32개 물체의 이전 민감도 결과는 이미 알려져 있었고 추가 생성은 동시 진행했다. [fine-history 봉인](fine_history/FORECASTS_SEALED) · [전체 label 봉인](refined_labels.sha256) · [채점 release](REFINED_RELEASED)','',
          '| 계산 및 사용 물체 | Contact law | Wendland soft − full | Dispersive soft − full | 두 family의 restored − dropped |',
          '|---|---|---:|---:|---:|']
    names={'standard_all':'9 sites 128/128','fine_all':'17 sites 128/128','fine_same8':'17 sites 첫 8/128','super_same8':'33 sites 첫 8/128'}
    for name,v in out.items():
        for i,law in enumerate(d['laws']):
            text.append(f"| {names[name]} | {law} | {entry(v['soft'][0][i])} | {entry(v['soft'][1][i])} | {entry(v['pooled_rescue'][i])} |")
    text.extend(['','같은 해상도·같은 표본에서 두 law×세 contrast에 대한 동시 95% paired t interval을 사용했다. 네 source seed를 먼저 평균하고 12개 catalog를 독립 단위로 삼았다. 17 sites 첫 8개와 33 sites 첫 8개를 직접 비교해야 수치 해상도 변화와 구성 sample의 차이를 구별할 수 있다. 8개 물체를 대상으로 한 재계산만으로 모든 고해상도 수치가 수렴했다고 판단할 수 없다.','',
                 '| 해상도 · 사용 물체 | Family | Law | 미분리/전체 | Soft−full의 임의 binary label 허용 범위 | Restored−dropped의 임의 binary label 허용 범위 |',
                 '|---|---|---|---:|---:|---:|'])
    for key in ('fine_all','super_same8'):
        v=out[key]
        for j,f in enumerate(d['families']):
            for i,l in enumerate(d['laws']):
                n,u=v['counts'][j][i]
                soft,rec=v['arbitrary_label_means'][j][i]
                text.append(f"| {names[key]} | {f} | {l} | {u}/{n:,} | [{soft[1]:+.6f}, {soft[2]:+.6f}] | [{rec[1]:+.6f}, {rec[2]:+.6f}] |")
    text+=['','각 알려지지 않은 충돌에 대해 동일 target label을 모든 source seed와 noise replicate에 공유하고, 두 binary completion 중 해당 contrast의 평균을 작게/크게 만드는 경우를 선택했다. 이 범위는 미분리 label의 민감도이고 표본 신뢰구간은 위 표에 별도로 실었다. [전체 catalog와 구간](refined68.json) · [기본 물리 검산](SOLVER_KO.md)','']
    (H/'REFINED_KO.md').write_text('\n'.join(text))
if __name__=='__main__':main()
