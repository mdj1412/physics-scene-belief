"""Build a publication-readable report from the locked independent result archives."""
from pathlib import Path
import re,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
def md(f):
    return (ROOT/f).read_text().strip()
def tab(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]+['| '+' | '.join(map(str,x))+' |' for x in rows])
def iv(v,b):
    return f'{v:+.5f} [{b[0]:+.5f}, {b[1]:+.5f}]'
def write():
    s=read('policy_stats33.json');ge=read('results_prefix/prefix_gain33.json');tl=read('transfer33.json');me=read('mechanism33.json')
    locks=read('results_prefix/discovery_lock.json');fixture=read('physics_audit33.json');meta=read('results_eval/completion.json')
    names=['1×1','2×2','Fixed 4','Uncertainty-only']; comp=['2×2 − 1×1','1×1 − fixed 4','2×2 − fixed 4','1×1 − uncertainty-only','2×2 − uncertainty-only']
    la=read('design_eval.json')['laws'];ob=ge['selected'];bp=ge['B']['paired_means'];bb=ge['B']['simultaneous95'];endpoint=['High posterior SD: absolute gain','High posterior SD: high − low','선택 집단: absolute gain','선택 집단 − 나머지']
    fk={'posterior_friction_sd':'friction coupling posterior SD','paid_tangential_spread':'관측한 접선 방향 probe velocity의 spread','paid_velocity_covariance':'factual outgoing velocity와 관측한 probe tangential velocity의 covariance'}
    rule=f"{ob['direction']} 방향, {ob['stage']}→{ob['stage']*2}회 단계에서 {fk[ob['feature']]}가 같은 block 여덟 환경 중 {'상위' if ob['half']=='upper' else '하위'} 네 곳인 prefix"
    text=[]
    def add(*v):text.extend(v)
    plus,minus,interaction=s['orientation']
    policy_concl='고정 배분보다 안정적으로 낮은 loss를 보인 learned policy는 없었다.' if all(v[1]>=0 for v in (s['ci'][1],s['ci'][2])) else '일부 learned policy는 fixed 4보다 낮은 loss를 보였다. 구간과 실용적 크기는 아래 표와 같다.'
    sub_concl=('선택한 집단은 별도 cohort에서도 양의 절대 gain과 나머지 집단 대비 우위를 모두 보였다.' if ge['confirmed_rules'][1] else '탐색에서 고른 집단이 별도 cohort에서도 양의 절대 gain과 나머지 집단 대비 우위를 함께 갖는다는 가설은 확인되지 않았다.')
    add('# v33 · 새 물체와 표면에서 어떤 접촉 이력에 probe를 더 쓸 가치가 있는가?',
        f'**{sub_concl} {policy_concl}** 동일한 training compute로 학습한 2×2와 1×1의 전체 held-out Brier 차이(2×2 − 1×1)는 {iv(s["delta"][0],s["ci"][0])}였다. 작은 값이 좋은 지표다. 아래 prefix 분석에서는 반대로 gain이 클수록 좋다.',
        f'**직접 검증한 prefix:** {rule}. 독립 cohort B의 absolute gain은 {iv(bp[2],bb[2])}, 반대 절반 대비 gain 차이는 {iv(bp[3],bb[3])}였다. 한 환경의 한 번의 충돌이 아니라, 그 환경에서 새로 만난 물체 128개에 대한 Brier를 얼마나 줄였는지가 질문이다.',
        '## 동일 비용에서 전체 정책은 어땠는가?',
        tab(['정책','세 학습 쌍 평균 Brier ↓','전체 GPU p90, ms [95%] ↓','Deployment-only p90, ms'],[[n,f'{s["mean"][j]:.6f}',f'{s["full"]["q"][j]:,.1f} [{s["full"]["ci"][j][0]:,.1f}, {s["full"]["ci"][j][1]:,.1f}]',f'{s["warm"]["q"][j]:,.1f}'] for j,n in enumerate(names)]),
        '모든 정책이 block당 8개 환경에 calibration probe 32회, 이어지는 1,024개 물체에 deployment probe 104회를 썼다. Fixed 4는 모두 4회씩 배정한다. 다른 세 정책은 모두 2회로 시작해 4개 환경만 4회, 그 중 2개만 8회까지 계속한다. 1×1과 2×2는 같은 ridge architecture를 쓰고 학습용 gain을 측정하는 방식이 다르다. Uncertainty-only는 각 시점의 friction coupling posterior SD로 다음 환경을 고른다. 128개 물체 중 13개에만 probe를 쓸 수 있고 나머지는 query view를 본다. 뒤이어 별도 collision에서 stick할 확률을 공통 decoder로 채점한다.',
        tab(['Brier 차이','고정 세 모델의 평균 [simultaneous 95%]','training seed까지 재표집 [simultaneous 95%]'],[[n,iv(s['delta'][j],s['ci'][j]),iv(s['delta'][j],s['crossed_ci'][j])] for j,n in enumerate(comp)]),
        'Primary interval은 네 object–surface 조합 안에서 8개씩의 독립 block을 재표집한다. 양쪽 방향, 같은 물체의 32 sensor noise draws, 각 training seed를 독립 데이터처럼 더하지 않았다. 두 번째 구간은 독립 training pairs가 세 개뿐인 sensitivity다. 다섯 대비에 미리 정한 기준은 absolute Brier 0.003이다. [통계 계획](POLICY_PLAN_KO.md) · [원수치·latency ratio](policy_stats33.json)',
        '## 어떤 paid prefix에서 추가 접촉이 유리했는가?',
        '처음 지불한 접촉 2회 또는 4회가 paid prefix다. 같은 prefix에서 그대로 멈출 때와 추가로 2회 또는 4회를 살 때를 비교했다. Gain = Brier(stop) − Brier(continue). 발견 집단 A와 검증 집단 B는 각각 네 물체·표면 조합에서 독립 block 4개씩, 모두 16개를 사용했다. 각 block은 여덟 환경의 같은 물체를 양쪽 방향으로 시험한다. 한 stage와 방향에서 8개 환경 중 고른 절반은 cohort마다 64개이며, 독립 분석 단위는 16개 blocks다. 두 독립 calibration futures와 두 독립 deployment batches를 교차해 얻은 네 값을 평균했다. Stop과 continue는 똑같은 deployment 물체·정답·sensor noise를 공유한다. 평가는 모델의 예상치가 아닌 simulator의 실제 stick outcome으로 한다.',
        f'Cohort A에서만 관측 가능한 3가지 feature × 상·하위 절반 × 2 stages × 2 방향, 합계 24개 후보를 비교했다. 대상의 absolute gain과 나머지 절반 대비 gain 차이 중 더 작은 값이 가장 큰 규칙을 골랐다. 그 규칙은 **{rule}**였다. 선정 파일과 A의 지문을 기록한 뒤 새 물체·새 접촉·새 noise로 cohort B를 생성했다. 추가로 방향이나 stage를 고르지 않는 사전 가설인 높은 posterior SD도 검증했다.',
        tab(['대상 / 검증량','탐색 A','독립 B [simultaneous 95%]','별도 policy worlds의 1×1 측정 [simultaneous 95%]'],[[n,f'{ge["A"]["paired_means"][j]:+.5f}',iv(bp[j],bb[j]),iv(tl['values'][j],tl['ci'][j])] for j,n in enumerate(endpoint)]),
        f'높은 posterior SD 가설의 두 하한이 모두 0을 넘었는가: **{"예" if ge["confirmed_rules"][0] else "아니오"}**. A가 선택한 규칙은 **{"예" if ge["confirmed_rules"][1] else "아니오"}**. 두 하한이 모두 0.003을 넘었는가는 각각 **{"예" if ge["exceeds_003"][0] else "아니오"} / {"예" if ge["exceeds_003"][1] else "아니오"}**다. 이 실용 기준은 실제 한 번의 접촉 가격을 환산한 값이 아니다.',
        f'B에서 선택 집단의 반대 절반의 absolute gain은 {bp[2]-bp[3]:+.5f}였다. Probe 추가 전후에 deployment observation이 바뀐 비율은 선택 집단 {ge["B"]["selected_switch_rates_target_other_excess"][0]*100:.1f}%, 나머지 {ge["B"]["selected_switch_rates_target_other_excess"][1]*100:.1f}%였다. 구매 결정을 바꿔도 새 관측이 더 정확하지 않으면 gain은 증가하지 않는다. ',
        '마지막 열은 A와 B 어느 쪽에서도 쓰지 않은 full-policy worlds에 같은 규칙을 적용했다. 같은 합성 표면 조건이지만 한 prefix에서 continuation·deployment batch가 각각 하나이므로 B의 2×2 측정보다 시끄럽다. Third panel로 A의 후보를 다시 선택하지 않았다. 모든 환경의 4회 prefix를 평가하는 것과 adaptive policy에서 4회까지 살아남은 환경만 평가하는 것도 다른 질문이다. 어떤 집단에 더 많이 쓸지 정할 때에는 그 stage에 실제 도달할 확률까지 필요하다. [후보 24개와 선정 지문](results_prefix/discovery_lock.json) · [독립 B 전체 결과](results_prefix/prefix_gain33.json) · [stage별 결과와 대각 반복](PREFIX_RESULTS_KO.md) · [제3 집단 기준](TRANSFER_PLAN_KO.md) · [제3 집단 결과](transfer33.json)',
        md('AUXILIARY_KO.md'),'![전체 policy와 독립 gain 검증](v33.png)', '## Friction 방향을 바꾸면 v32의 결과가 유지되는가?',
        'V32 directional law에서 2×2 − 1×1은 세 학습 쌍 평균 +0.00088이었다. 당시 첫 seed의 부호를 본 후 세분화한 결과였다. V33은 그 결과를 직접 시험하기 위해 두 object profile과 surface response 두 가지에 대해 방향을 양쪽으로 미리 지정했다. 같은 물체와 noise를 쓰되 반대 orientation에서 물리 법칙이 내는 probe response와 stick label을 다시 생성했다. [v32 결과](../compute_matched_stopping_v32/REPORT_KO.md)',
        tab(['방향·대비','2×2 − 1×1 Brier [세 대비 simultaneous 95%]'],[[n,iv(q,c)] for n,q,c in zip(['Plus','Minus','Plus − minus'],s['orientation'],s['orientation_ci'])]),
        f'Plus 방향에서 세 모델 쌍 각각의 2×2 − 1×1은 {s["seed_by_sign"][0][0]:+.5f}, {s["seed_by_sign"][1][0]:+.5f}, {s["seed_by_sign"][2][0]:+.5f}였다. V32에서와 달리 세 seed가 같은 방향을 가리키지 않았다.',
        tab(['합성 물체 / 표면 / 방향',*names,'2×2 − 1×1'],[[n.replace('_',' / '),*[f'{v:.5f}' for v in a],f'{a[1]-a[0]:+.5f}'] for n,a in zip(la,np.array(s['by_condition']).reshape(8,4))]),
        f'방향을 반전하면 같은 물체의 query stick label이 {100*s["mirror_label_change"]:.1f}%에서 바뀌었다. 두 learned policy는 방향 전체에서 환경별 calibration 횟수를 {100*s["allocations_changed"]:.1f}% 다르게 배정했다. 하지만 최종 128개 물체에서 실제로 서로 다른 sensor를 고른 비율은 {100*s["observations_changed"]:.1f}%였다. 횟수 배정과 최종 관측 선택 사이에서 효과가 얼마나 작아지는지 보여주는 수치다.',
        f'두 방향 사이의 policy 차이를 같은 횟수에서의 downstream risk와 횟수 배정 변화로 분해했다. 배정 변화의 기여는 {iv(me["means"][0],me["pointwise95"][0])}, downstream risk 변화의 기여는 {iv(me["means"][1],me["pointwise95"][1])}이었다. 두 점추정의 합이 위의 plus − minus 차이와 일치한다. 후자에는 direction에 따른 probe 응답·query label·관측 선택 변화가 함께 들어 있다. [보조 분석 계획](MECHANISM_PLAN_KO.md) · [계산식과 결과](mechanism33.json)',
        '전체 정책 결과를 본 후 탐색적으로, 동일한 32개 budget에서 420개 배분을 모두 대조했다. 정답을 먼저 알고 같은 block에서 고르는 사후 Brier는 0.12114였지만, 같은 object·surface·orientation의 다른 일곱 blocks에서 배분을 정한 뒤 옮기면 0.12548이었다. 균등 무작위 배분의 기댓값은 0.12562였다. Hindsight가 확보한 0.00456의 여지는 새 prefix에서 달성 가능한 이득의 증거가 아니다. Block 간 이전에는 실제 population coupling을 알려 준 보조 기준을 사용했다. [사후 계산과 조건](headroom33.json)', '## Training compute와 재현 범위',
        tab(['Training pair','1×1 학습 환경 / GPU wall-s','2×2 학습 환경 / GPU wall-s','1×1 target Brier','2×2 target Brier'],[[i+1,*[f'{v[k][0]} / {v[k][1]:.2f}' for k in ('broad','repeated')],*[f'{b:.6f}' for b in s['byseed'][i][:2]]] for i,v in enumerate(s['training'])]),
        md('GPU_PROTOCOL_KO.md').split('\n',1)[1].strip(),
        f'평가는 {meta["gpu"]}, allocation {meta["job"]}에서 실행했다. 원확률에서 재계산한 stage Brier와 기록값의 최대 절대 차이는 {s["brier_audit_error"]:.2e}다. Training seed와 target streams의 중복을 검사했고, mirror pair에만 같은 target streams를 허용했다. 독립 physics audit에서 {fixture["unique_audit_objects"]:,}개 물체를 별도로 생성해 contact normal과 stick 분기를 검증했다. [GPU 완료 기록](results_eval/completion.json) · [평가 코드](evaluate33.py) · [통계·quota 독립 검산](policy_stats33.py) · [물리식 검산](physics_audit33.json) · [독립 prefix GPU 완료 기록](results_prefix/completion.json)',
        md('PHYSICAL_KO.md'),'## 논문에서 주장할 수 있는 부분과 다음 판단',md('ASSESSMENT_KO.md'),
        md('LITERATURE_KO.md'),
        '## 재현 파일','서버 프로젝트: `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/directional_prefix_v33`. [재현 순서](REPRODUCE_KO.md) · [고정 simulator](target33.py) · [평가 설계](design_eval.json) · [prefix 설계](PREFIX_PLAN_KO.md) · [paired gain 측정](prefix33.py) · [confirmatory 분석](prefix33_stats.py) · [측정 데이터 출처](physical33.json).')
    result='\n\n'.join(text)+'\n'
    (ROOT/'REPORT_KO.md').write_text(result)
    for table in re.findall(r'(?m)(?:^\|.*\|\n?)+',result):
        assert len(set(l.count('|') for l in table.splitlines()))==1,table[:80]
    def link(m):
        lab,url=m.groups()
        if url.startswith(('http','https','#')):return m[0]
        dest=(ROOT/url).resolve()
        assert dest.is_file(),dest
        return lab+'('+str(dest)+')'
    pub=re.sub(r'(!?\[[^\]]+\])\(([^)]+)\)',link,result)
    out=ROOT.parents[2]/'outputs'/'physics_scene_belief_directional_prefix_v33_report.md'
    if out.parent.is_dir():
        out.write_text(pub)
    print(len(result),len(result.splitlines()),'lines; canonical:',hashlib.sha256(result.encode()).hexdigest())
if __name__=='__main__':write()
