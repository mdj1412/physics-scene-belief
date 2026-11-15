"""Assemble the Korean report from audited outcomes and primary papers."""
import json,re,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text()); md=lambda f:(R/f).read_text().strip()
def iv(v,c):return f'{v:+.5f} [{c[0]:+.5f}, {c[1]:+.5f}]'
def table(names,rows):return '\n'.join(['| '+' | '.join(names)+' |','|'+'---|'*len(names)]+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    s=read('decision_stats34.json');p=read('primary_stats34.json'); h=read('head_timing34.json')
    se=read('selector_stats34.json');te=read('terminal_stats34.json');ex=read('external34.json');bal=read('balance_stats34.json');audit=read('independent_audit34.json')
    sch=['한 방향','양방향 교대'];me=['Frozen gain scorer','Fixed 4','Uncertainty-only','Neutral gain head','Estimated-sign head','True-sign head','Placebo head'];mv=np.asarray(s['means'])
    row=[]
    for m in (0,1,2,4):
        series=p['full']['p90'] if m<3 else h['full']['p90']
        j=m if m<3 else 0
        di=m if m<3 else 10;dd=s['core'] if m<3 else s['aug'];ci=s['core_band'] if m<3 else s['refit_band']
        row.append([me[m],*[f'{q:.6f}' for q in mv[:,m]],iv(dd[di],ci[di]),*[f'{series[i][j]:,.1f}' for i in (0,1)]])
    gg=[]
    for a,shift in enumerate((0,5)):
        for i,n in enumerate(['Estimated − neutral','True − neutral','Estimated − placebo','Estimated − fixed 4','Estimated − uncertainty-only']):
            j=shift+i
            gg.append([sch[a],n,iv(s['aug'][j],s['aug_band'][j]),iv(s['aug'][j],s['refit_band'][j])])
    detail=[]
    for a in (0,1):
        for name,obj,idx,base in [('Stopping',s,5*a,'neutral learner'),('Sensor selection',se,4*a,'neutral selector'),('Terminal prediction',te,3*a,'neutral decoder')]:
            output=obj['aug'] if name=='Stopping' else obj['delta']
            ci=obj['refit_band'] if name=='Stopping' else obj['ci']
            detail.append([sch[a],name,iv(output[idx],ci[idx]),iv(output[idx+1],ci[idx+1]),base])
    titles=['Estimated stopping − neutral','True stopping − neutral','Estimated terminal − neutral','True terminal − neutral']
    text=['# v34 · 같은 접촉 횟수에서 한 방향과 양방향 probing 비교',md('FINDINGS_KO.md'),'**가장 유망한 단서는 stopping이 아니라 terminal prediction이었다.** 결과를 본 뒤 시행한 anchored control에서 fixed 4의 원래 확률에 추정한 표면 방향만 반영하자 Brier가 단방향에서 0.00596, 양방향에서 0.00535 낮아졌다. 아래에 원 decoder와의 직접 대조, 별도 물체 자료의 이전 결과 및 사후 분석의 한계를 함께 제시한다.',
          '![방향 판독과 결정 단계별 Brier 차이](v34.png)',
          '왼쪽 그림의 정답은 표면에서 마찰이 강한 방향이고, 오른쪽 그림의 정답은 별도 충돌에서의 stick 여부다. 왼쪽의 더 낮은 loss가 오른쪽의 개선을 보장하지 않는다. 오른쪽은 estimated sign을 추가한 head에서 똑같은 입력 구조의 neutral head를 뺀 값이며, 음수가 개선이다.',
          '## 전체 stopping policy: 동일 예산의 성능과 시간',
          table(['정책','한 방향 Brier ↓','양방향 Brier ↓','양방향 − 한 방향 [simultaneous 95%]','한 방향 GPU p90 ms','양방향 GPU p90 ms'],row),
          '모든 방법은 8개 환경에 **calibration probe 32회**를 쓴 뒤, 1,024개 새 물체에 **deployment probe 104회**를 썼다. Fixed 4는 환경당 4회, 나머지는 2회에서 시작해 네 곳을 4회로 늘리고 그중 두 곳만 8회까지 간다. Frozen scorer는 v32의 1×1 모델을 그대로 쓰고, 새 estimated-sign head는 양쪽 schedule의 gain 자료에 같은 방식으로 fit한 공통 head다. Frozen scorer는 일방향 좌표계에 맞춰 양의 명령에서 tangential velocity 부호를 반사했다. 사용자가 보는 표면 orientation을 정책에 알려 주지 않았다.',
          f'GPU p90은 {p["device"]}에서 1,024개 물체 전체의 posterior update, 순차 stopping, acquisition 및 선택된 관측 decoding까지의 synchronized wall time이다. 각 정책·schedule당 48회, 독립 timing clusters는 8개다. 초기화·data generation·실제 접촉 및 reset 시간은 제외했다. 새 head는 별도 timing pass에서 fixed 4 및 uncertainty-only와 순서를 섞어 실측했다. 그 pass의 baseline p90(한 방향/양방향)은 fixed 4 {h["full"]["p90"][0][1]:.1f}/{h["full"]["p90"][1][1]:.1f} ms, uncertainty-only {h["full"]["p90"][0][2]:.1f}/{h["full"]["p90"][1][2]:.1f} ms였다. [기본 정책의 Brier와 GPU timing](primary_stats34.json) · [새 head와 두 baseline의 GPU timing](head_timing34.json)',
          table(['기본 정책의 동일 schedule 내 차이','한 방향 [simultaneous 95%]','양방향 [simultaneous 95%]'],[[n,iv(s['core'][3+j],s['core_band'][3+j]),iv(s['core'][6+j],s['core_band'][6+j])] for j,n in enumerate(['Frozen − fixed 4','Frozen − uncertainty-only','Uncertainty-only − fixed 4'])]),
          '위 표와 기본 정책 세 개의 schedule 차이에는 넓게 잡은 9개 대비의 simultaneous interval을 적용했다. GPU runner가 독립적으로 정한 7개 대비(frozen 중심)도 [별도 검산](primary_stats34.json)에 남겼다. 각 family에서 초기 방향별로 네 block씩 함께 재표집한다. 먼저 가는 방향의 유리함과 교대 여부의 효과가 섞이지 않게 했다. Practical margin은 사전에 정한 absolute Brier 0.003이다.',
          md('PROTOCOL_POLICY_KO.md'),md('DIAG_RESULTS_KO.md'),
          '## 같은 정보가 어느 결정에서 쓸모가 있는가?',
          '세 위치를 비교했다. **Stopping**은 32회 calibration budget을 나눈다. **Sensor selection**은 fixed 4를 유지한 채 각 환경의 128개 물체 중 probe를 줄 13개만 고른다. **Terminal prediction**은 fixed 4와 기존 sensor 선택까지 유지하고 이미 받은 관측에서 stick을 예측하는 마지막 확률만 바꾼다. 각 행의 neutral과 estimated/true는 같은 task, 같은 예산, 같은 train family와 feature 구조를 쓴다. 단계 간 절대 Brier를 직접 비교해 어느 모델이 더 우수하다고 해석하지 않는다.',
          table(['Schedule','정보를 넣은 위치','Estimated sign − neutral [95%]','True sign − neutral [95%]','비교 대상'],detail),
          'Estimated sign은 별도 calibration corpus에서 학습한 P(orientation=plus)다. True sign은 simulator의 정답을 준 privileged control이다. 매 block의 8개 환경이 같은 orientation을 공유하므로 sign을 모든 score에 상수로 더하면 stopping 순위가 바뀌지 않는다. 따라서 gain과 sensor selection 모두 cue×prefix/factual interaction을 포함한다. Terminal decoder는 원 logit·선택된 sensor와 sign 간 interaction으로 조정한다. 각 단계의 표는 서로 다른 확인 가설 family에 대해 보정한 구간이다. Stopping은 source와 held-out blocks를 함께 다시 뽑아 head까지 refit한 구간을 썼다. Selector/decoder는 학습된 모델에 조건부인 구간이며 물리 법칙 모집단 전체에 대한 구간이 아니다.',
          table(['Schedule','Stopping의 Brier 대비','조건부 구간','Gain head까지 refit한 구간'],gg),
          f'Estimated와 neutral이 환경별로 다른 접촉 횟수를 배정한 비율은 한 방향 {100*s["counts_changed_est_v_neutral"][0]:.1f}%, 양방향 {100*s["counts_changed_est_v_neutral"][1]:.1f}%였다. Placebo는 paired +/− world의 cue를 block 단위로 뒤집어 일정 비율에서 실제 방향과의 대응을 끊는다. 같은 환경 여덟 곳을 독립적으로 뒤섞지 않았다. Placebo는 permutation distribution 전체가 아니라 사전 고정한 balanced permutation 한 번이므로 단독으로 정보 유무를 판정하지 않는다. [결정 모델 및 통계 계획](DECISION_PLAN_KO.md) · [실제 held-out Brier와 두 가지 구간](decision_stats34.json)',
          table(['Schedule','원 fixed4/원 selector','Neutral selector','Estimated selector','True-sign selector','Placebo selector'],[[sc,*[f'{v:.6f}' for v in se['means'][a]]] for a,sc in enumerate(sch)]),
          table(['Schedule','원 fixed4/원 decoder','Neutral decoder','Estimated decoder','True-sign decoder'],[[sc,*[f'{v:.6f}' for v in te['means'][a]]] for a,sc in enumerate(sch)]),
          'Selector는 training family의 실제 potential loss 차이를 supervision으로 쓴다. Test 물체의 두 potential outcomes는 선택이 확정된 뒤에만 채점에 쓴다. Terminal 모델은 4회 prefix에서의 sign을 받고 동일하게 선택된 관측만 해석한다. 새로운 sensor나 label을 test time에 사지 않는다. 원래 고정 decoder는 참 population coupling을 아는 privileged reference이지만 orientation을 모른다. Neutral calibration과도 비교해야 단순한 재보정에서 오는 효과와 sign cue의 추가 효과를 분리할 수 있다. [Sensor selection 설계](SELECTOR_PLAN_KO.md) · [관측 정책별 결과](selector_stats34.json) · [Terminal calibration 결과](terminal_stats34.json)',
          md('ANCHORED_RESULTS_KO.md'),'### 사전 설계한 두 head를 다른 물체 집단에 그대로 적용한 결과',
          table(['Brier 대비','V33의 별도 32개 paired blocks [simultaneous 95%]'],[[n,iv(v,c)] for n,v,c in zip(titles,ex['delta'],ex['ci'])]),
          '원 물체와 표면의 target family별로 source family 한 곳의 v34 자료만 사용한 뒤, 그 모델을 재학습 없이 v33의 별도 32,768개 물체에 적용했다. v33은 negative 방향만 probing했으므로 교대 효과를 검증하는 자료는 아니다. 이전 버전 연구자는 v33의 기본 정책 결과를 본 적이 있다. 새로운 v34 모델의 결과를 본 후 source family나 계수를 바꾸지 않았다. [외부 archive 적용 코드](external34.py) · [전이 결과](external34.json)',
          '### Probe 시작 방향에 따른 상쇄 검사',
          table(['정책','처음이 낮은 마찰 방향: 교대 − 단방향','처음이 높은 마찰 방향: 교대 − 단방향','사전 orientation을 알려 준 routing − 단방향'],[[n,iv(bal['initial_low_then_high'][0][j],bal['simultaneous95'][j]),iv(bal['initial_low_then_high'][1][j],bal['simultaneous95'][3+j]),iv(bal['oracle_routed_minus_same'][j],bal['simultaneous95'][6+j])] for j,n in enumerate(me[:3])]),
          '마지막 열은 첫 운동이 표면의 어느 방향인지 별도로 안다고 가정한다. 어느 schedule이 유리한지는 training family에서만 고른 뒤 뺀 family에서 평가한다. 이 실험에서 측정한 조건부 관계가 실제 표면의 방향을 무료로 알 수 있음을 의미하지 않는다. [분석 계획](BALANCE_SENSITIVITY_KO.md) · [12개 대비](balance_stats34.json)',
          '## 논문 기여 가능성과 후속 결정',md('ASSESSMENT_KO.md'),md('LITERATURE_KO.md'),
          '## 실행·재현 자료',
          f'서버 위치: `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/bidirectional_probe_v34`. 기본 GPU 평가는 allocation {p["allocation"]}에서 완료했다. 새 head에서 {audit["replayed_records"]}개 schedule·orientation·block의 순차 결정을 독립 재생하고, timing에 들어간 {audit["real_gpu_timing_record_matches"]}개 기록도 대조했다. 실제 선택 확률에서 재계산한 평균 Brier의 최대 오차는 {audit["max_abs_error"]:.2e}였다. [재현 순서](REPRODUCE_KO.md) · [Signed action 물리와 likelihood 검증](action_audit34.json) · [GPU 평가 코드](run34.py) · [기본 GPU 완료](results_policy/completion.json) · [추가 GPU timing 완료](results_head_time/completion.json) · [결정 재생 검산](independent_audit34.json).']
    body='\n\n'.join(text)+'\n';(R/'REPORT_KO.md').write_text(body)
    for t in re.findall(r'(?m)(?:^\|.*\|\n?)+',body):
        assert len({x.count('|') for x in t.splitlines()})==1,t[:100]
    def fix(m):
        name,link=m.groups()
        if link.startswith(('http://','https://','#')):return m[0]
        f=(R/link).resolve();assert f.is_file(),f
        return name+'('+str(f)+')'
    pub=re.sub(r'(!?\[[^\]]+\])\(([^)]+)\)',fix,body);out=R.parents[2]/'outputs'
    if out.is_dir():(out/'physics_scene_belief_bidirectional_probe_v34_report.md').write_text(pub)
    print('report ok:',len(text),'elements,',len(body),'characters')
if __name__=='__main__': main()
