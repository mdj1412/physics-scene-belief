"""Generate the shareable Korean v36 report from paired statistical summaries."""
from pathlib import Path
import json,re
R=Path(__file__).resolve().parent
def read(n):return (R/n).read_text().strip()
def data(n):return json.loads(read(n))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(row)+' |' for row in rows])
def f(x):return f'{x:.5f}'
def d(x):return f'{x:+.5f}'
def band(p):return '['+', '.join(d(x) for x in p)+']'
def db(v,j):return d(v['delta'][j])+' '+band(v['ci_simultaneous'][j])
def percent(x):return f'{100*x:.1f}%'
names=['Conventional physics','v35 correction','Matched physics','Matched always apply','Matched gate · 세 측정','Matched gate · 네 측정']
groups={'local_equal':'세 local regime 평균','strong':'Local strong','weak':'Local weak','null':'Local null','global_null':'Global null','four_equal':'네 regime 평균'}
def main():
    a=data('stats36.json');g=a['scopes'];T=a['timing'];base=g['local_equal']; A=a['auc']
    sec=data('increment36.json');supp=data('supplement36.json')
    tbl=table(['방법','Local 평균','Strong','Weak','Local null','Global null','GPU p90 (ms)'],
        [[n,*[f(g[k]['loss'][j]) for k in ('local_equal','strong','weak','null','global_null')],f'{T["p90_ms"][j]:,.1f}'] for j,n in enumerate(names)])
    comp=table(['같은 world에서의 비교','Local 평균 Δ Brier [95%]','Weak Δ [95%]','Local null Δ [95%]','Global null Δ [95%]'],
        [[n,*[db(g[k],i) for k in ('local_equal','weak','null','global_null')]]
         for i,n in enumerate(['Gate4 − conventional physics','Gate4 − v35 correction','Gate4 − Gate3','Gate4 − matched always apply','Gate4 − matched physics','Gate3 − matched always apply','Matched physics − conventional physics'])])
    rates=table(['표면','Gate3 보정 적용률','Gate4 보정 적용률','네 번째 측정 후 결정 변경','실제로 보정이 유리했던 환경¹'],
        [[n,*[percent(g[k]['extras'][j]) for j in (0,1,5,8)]] for k,n in groups.items() if k in ('strong','weak','null','global_null')])
    rankings=table(['두 표면을 구별하는 문제','첫 세 관측 AUROC','네 번째 관측 포함 AUROC [95%]'],
        [[n,f'{A[k]["auc"][0]:.3f}',f'{A[k]["auc"][1]:.3f} [{A[k]["near_ci95"][0]:.3f}, {A[k]["near_ci95"][1]:.3f}]'] for k,n in [('strong_vs_low','Strong 대 weak/null'),('strong_vs_weak','Strong 대 weak'),('strong_vs_null','Strong 대 local null'),('weak_vs_null','Weak 대 local null')]])
    gain_rows=[]
    for k in ['local_equal','strong','weak','null','global_null']:
        x=g[k]['extras'];on=x[1]
        gain_rows.append([groups[k],d(x[12]),d(-x[4]),d(-x[3]/on) if on else '해당 없음',
                          d(-(x[4]-x[3])/(1-on)) if on!=1 else '해당 없음',d(x[10])])
    gain_table=table(['표면','Gate4의 평균 예측 gain','모두 보정할 때 실제 gain','선택된 환경의 실제 gain','건너뛴 환경의 실제 gain','분할 oracle Δ Brier'],gain_rows)
    timed=table(['Gate4 p90 비교 대상','차이 (%)','동시 95% interval','+2% 기준'],
      [[names[j],f'{(v-1)*100:+.2f}%',f'[{(lo-1)*100:+.2f}%, {(hi-1)*100:+.2f}%]',('상한 이내' if hi<1.02 else ('상한 초과' if lo>1.02 else '미확정'))] for j,(v,(lo,hi)) in enumerate(zip(T['gate4_ratio'],T['ci']))])
    limits={}
    for h,u in [(0.,1.10),(.023,.425),(.023,.5),(.023,.75),(.023,1.10),(.025,.425)]:
      row=next(v for v in a['sensor'] if v['height']==h and v['speed']==u)
      limits[(h,u)]=row
    limtable=table(['Probe 높이','Tangential speed','Strong 대 local null의 정확도 상한','두 표면의 velocity mean이 같은 비율','Weak 대 local null의 정확도 상한'],
        [[f'{h*1000:g} mm',f'{u:g}',percent(v['values'][0][0]),percent(v['values'][0][1]),percent(v['values'][2][0])] for (h,u),v in limits.items()])
    soft=table(['표면','고정 32.5% 보정','세 측정 soft correction','네 측정 soft correction','Gate4 − 고정 32.5% [95%]'],
        [[n,*[f(x) for x in a['shrinkage'][k]['brier']],d(a['shrinkage'][k]['contrasts'][0])+' '+band(a['shrinkage'][k]['ci'][0])] for k,n in groups.items() if k!='four_equal'])
    pdiag=sec['scopes']['local_equal']; cov=base['covariance']; rg=a['prevalence_sweep']
    sens=', '.join(f'{p*100:g}%일 때 {d(v)}' for p,v,_,_ in rg['reference_v35']['curve'])
    bits=[
      '# v36 · 네 번 중 한 번을 query 부근에서 측정하면 friction 보정이 필요한 표면을 가려낼 수 있는가?',
      read('VERDICT_KO.md'),tbl,
      'Local 평균은 strong, weak, local null에 같은 비중을 준 값이다. 네 번째 접촉까지 기존 방식으로 진행한 두 방법을 conventional, 네 번째만 query 부근으로 옮긴 네 방법을 matched로 표시했다. 모든 loss는 독립 deployment 물체에서 측정한 held-out Brier이며 낮을수록 좋다. 각 행의 GPU p90에는 표에 해당하는 보정 계산까지 들어간다.',
      '## 어떤 차이가 네 번째 probe의 효과인가',comp,
      'Δ는 왼쪽 방법에서 오른쪽 방법을 뺀 값이다. 각 열 안에서 7개 비교에 동시 95% interval을 사용했다. Primary는 세 local regime의 평균이다. 각 regime의 열은 조건별 보조 비교다. v35와 같은 absolute Brier 0.003을 실용 기준으로 사용했다. 같은 원래 decoder를 쓰는 Gate4−Gate3와 Gate4−matched physics를 함께 보면 추가 관측과 최종 보정의 이득을 구분할 수 있다. [사전 분석 계획](ANALYSIS_PLAN_KO.md) · [수치 및 검산](stats36.json)',
      '## Gate가 방향이 약한 표면을 읽었는가',rankings,rates,
      'AUROC는 posterior mean friction strength가 강한 표면에 더 높은 값을 주는지 측정한다. 0.5는 무작위 순위에 해당한다. Gate는 expected Brier gain을 기준으로 보정을 선택하므로, 같은 strength라도 방향 추정의 확신과 원래 decoder의 오차가 다르면 다른 결정을 내릴 수 있다. ¹보정이 실제로 유리했던 환경은 같은 held-out 환경에서 always apply와 matched physics의 Brier를 직접 비교한 기술값이다. 환경마다 물체 128개를 사용하므로 이 분류에는 측정 오차가 있다.',
      f'보정 적용률을 전체 local 평균에 맞춘 무작위 정책에 비해 실제 Gate4의 Δ는 {d(cov)} {band(base["covariance_ci"])}였다. 환경별 always-apply − never-apply 차이를 D, 적용 여부를 G라 하면 이 값은 E[GD]−E[G]E[D]다. 환경의 geometry와 sign에 따라 보정 여부를 잘 선택해도 좋아질 수 있는 지표이므로, 아래에는 같은 환경에서 local regime만 바꾼 대조도 제시한다.',
      gain_table,
      '이 표에서 gain은 matched physics의 loss에서 always-apply loss를 뺀 값이므로 양수가 보정에 유리하다. 선택·건너뜀 두 열은 Gate4가 고른 환경과 건너뛴 환경에서 같은 correction을 실제로 적용했을 때의 평균이다. 마지막 열은 환경 내 128개 deployment 물체를 64개씩 나눠 한쪽의 정답으로 보정 여부를 고른 뒤 다른 쪽에 적용하고 교대로 평가한 Δ다. 음수일수록 matched physics보다 좋다. 이 진단은 배포 전에 실제 정답을 추가로 요구한다.',
      read('INCREMENT_KO.md').replace('# ','## ',1),read('SUPPLEMENT_KO.md').replace('# ','## ',1),
      '## 같은 예산에서 보정량만 줄이는 대조',soft,
      '고정 32.5%는 네 source regime에서 가정한 near-field multiplier의 평균이다. Soft correction은 같은 βz에 각 환경의 posterior mean multiplier를 곱한다. 모두 동일한 matched physics probability와 deployment actions를 사용했다. 위 세 secondary 방법의 GPU p90은 따로 측정하지 않았다. [계산 코드](stats36.py)',
      f'Strong 표면의 실제 빈도가 달라지면 전체 효과도 달라진다. Weak와 local null의 비율을 서로 같게 두고 strong 비율만 바꿔 이미 측정된 loss를 재가중하면 Gate4−v35는 {sens}였다. 이는 배포 집단 비율에 대한 sensitivity이며 각 비율에 맞춰 다시 fitting한 결과는 아니다.',
      '## 한 번의 측정이 제공하는 정보량',limtable,
      '각 물체의 latent geometry, orientation, 두 후보 표면의 정확한 law까지 아는 관찰자가 단 한 번의 noisy velocity로 어느 표면인지 추측하는 문제를 계산했다. 두 표면의 사전확률은 같다. 두 Gaussian likelihood의 평균 거리 D를 noise standard deviation으로 정규화하면 최적 정확도는 Φ(D/2)다. 같은 물체의 두 충돌이 모두 stick하면 tangential impulse가 friction coefficient에 의존하지 않아 D가 0이 될 수 있다. 이 열은 실제 gate의 달성 정확도와 구별되는 privileged upper bound다. 속도를 바꾼 행은 동일 물체에 대한 물리 계산이며 해당 동작을 쓰는 policy의 Brier나 latency 실측은 아직 없다. [계산식](diagnose36.py)',
      '![v36 results](v36.png)',
      '## 전체 GPU p90',timed,
      '각 방법에 대해 8개 독립 timing block에서 세 번씩 총 24회 측정했다. 각 block에 모든 방법을 함께 넣어 randomized execution order로 실행했다. GPU p90은 한 대의 NVIDIA RTX PRO 6000 Blackwell Server Edition에서 환경 8개, 총 1,024개 deployment 물체를 처리한 wall time이다. CPU에서 실행된 gate, GPU synchronization과 결과의 host 반환을 포함한다. 실제 로봇 이동 시간은 별도로 검증해야 한다. 각 비교의 interval은 같은 timing block을 대응시켜 재표집하고 다섯 비교에 동시 보정했다.',
      read('DESIGN_KO.md'),read('GATE_KO.md').replace('# ','## ',1),
      read('CONTRIBUTION_KO.md'),
      read('LITERATURE_KO.md').replace('# ','## ',1),
      '## 재현 기록',
      f'GPU에서 32개 paired block을 완료했다. Raw archive {a["verified_archives"]}개의 SHA-256을 대조하고 6개 방법의 예측에서 Brier를 다시 계산했다. θ로 재생성한 query label 및 near velocity도 일치했다. Local regimes에서 conventional 4개 응답·원래 예측·세 측정의 gate가 모두 원소별로 동일함을 확인했다. Calibration, deployment와 bank의 seed ID {a["distinct_seeds"]}개는 서로 다른 block 간에 겹치지 않는다. [Source 고정값](gate36_freeze.json) · [GPU 코드](run36.py) · [실행 결과](results/completion.json) · [분석 수치](stats36.json) · [재현 방법](REPRODUCE_KO.md)',
      '서버 보고서 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/query_matched_probe_v36/REPORT_KO.md`다.'
    ]
    md='\n\n'.join(bits)+'\n'
    (R/'REPORT_KO.md').write_text(md)
    base=R.parents[2];folder=base/'outputs'
    if folder.exists() and (base/'work').exists():
        prefix='../work/physics-scene-belief/query_matched_probe_v36/'
        rewrite=lambda m:m[0] if '://' in m[1] or m[1].startswith(('/','#')) else ']('+prefix+m[1]+')'
        (folder/'physics_scene_belief_query_matched_probe_v36_report.md').write_text(re.sub(r'\]\(([^)\s]+)\)',rewrite,md))
if __name__=='__main__':main()
