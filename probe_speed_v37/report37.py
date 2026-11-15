"""Render the Korean v37 report and local delivery copy with resolvable links."""
from pathlib import Path
import json,re
R=Path(__file__).resolve().parent
def load(n):return json.loads((R/n).read_text())
def read(n):return (R/n).read_text().strip()
def num(x):return f'{x:.5f}'
def delta(x):return f'{x:+.5f}'
def pct(x):return f'{100*x:.1f}%'
def interval(x):return '['+', '.join(delta(v) for v in x)+']'
def simple(x):return '['+', '.join(f'{v:.3f}' for v in x)+']'
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(row)+' |' for row in rows]])
def brier(g,k):return delta(g['brier_delta'][k])+' '+interval(g['brier_simul95'][k])
def main():
    a=load('stats37.json');groups=a['scopes']; local=groups['local_equal'];timing=a['timing'];AU=a['discrimination']
    names=['+0.50','+0.75','+1.10']
    scopes={'local_equal':'Local 평균','weak_null_equal':'Weak/null 평균','strong':'Strong','weak':'Weak','null':'Local null','global_null':'Global null','four_equal':'네 regime 평균'}
    primary=table(['네 번째 speed','Slip','Weak 대 local null AUROC','32.5% Brier ↓','Gate Brier ↓','Gate − 32.5% [동시 95%]','GPU p90, 고정 / gate (ms)'],
       [[n,pct(local['extras'][j][0]),f'{AU["weak_vs_null"]["auc"][j]:.3f}',num(local['losses'][j][2]),num(local['losses'][j][3]),brier(local,j),f'{timing["p90_ms"][j][0]:,.1f} / {timing["p90_ms"][j][1]:,.1f}'] for j,n in enumerate(names)])
    results=table(['표면 분포','+0.50 Δ [95%]','+0.75 Δ [95%]','+1.10 Δ [95%]'],
       [[name,*[brier(groups[key],i) for i in range(3)]] for key,name in scopes.items()])
    discr=table(['구별 문제','+0.50 AUROC','+0.75 AUROC','+1.10 AUROC','+0.75 − .50 [95%]','+1.10 − .50 [95%]'],
        [[label,*[f'{v:.3f}' for v in AU[key]['auc']],*[f'{AU[key]["delta"][i]:+.3f} '+simple(AU[key][('delta_simul95' if key!='strong_vs_null' else 'delta_pointwise95')][i]) for i in range(2)]] for key,label in [('weak_vs_null','Weak 대 local null'),('strong_vs_low','Strong 대 weak/null'),('strong_vs_null','Strong 대 local null')]])
    behavior=table(['표면','Slip · +0.50 / +0.75 / +1.10','Gate 적용률 · 같은 순서','고속에서 gate 선택 변경 · .75 / 1.10'],
        [[name,' / '.join(pct(e[0]) for e in groups[key]['extras']),' / '.join(pct(e[1]) for e in groups[key]['extras']),' / '.join(pct(e[2]) for e in groups[key]['extras'][1:])] for key,name in scopes.items() if key in ('strong','weak','null','global_null')])
    weights=load('prevalence37.json')
    in_domain=[]
    for i,entry in enumerate(weights):
        vals=[delta(v['gate_minus_fixed']) for v in entry['examples']]
        root=entry['break_even']
        in_domain.append([names[i],*vals,(pct(root) if root is not None else '교차 없음')])
    prevalence=table(['속도','Strong 0%','25%','50%','75%','100%','손익분기 strong 비율'],in_domain)
    cs=[]
    for i,n in enumerate(names[1:]):
        j=i+1
        cs.append([n,brier(local,j+2),brier(local,j+4),delta(local['extra_speed_change'][i][4])+' '+interval(local['extra_speed_ci95'][i][4]),pct(local['extras'][j][3])])
    speedchange=table(['고속 − +0.50','실제 Gate Brier 차이 [95%]','Gate−32.5% 차이의 변화 [95%]','공통 +0.50 predictions에서 Gate 차이 [95%]','배포 action 변경률'],cs)
    diag=table(['속도','Physics','항상 보정','세 관측 gate','네 관측 gate','Soft 보정','과거에 고른 고정값','Gate − 과거 고정값 [95%]','분할 oracle¹'],
        [[n,num(local['losses'][j][0]),num(local['losses'][j][1]),num(local['extras'][j][8]),num(local['losses'][j][3]),num(local['extras'][j][7]),num(local['extras'][j][9]),delta(local['tuned_fixed_delta'][j])+' '+interval(local['tuned_fixed_ci95'][j]),num(local['extras'][j][5])] for j,n in enumerate(names)])
    ti=table(['속도','32.5% GPU p90 (ms)','Gate GPU p90 (ms)','Gate 비용 변화 [동시 95%]','+2% 기준'],
        [[n,f'{p[0]:,.1f}',f'{p[1]:,.1f}',f'{100*(r-1):+.2f}% [{100*(ci[0]-1):+.2f}%, {100*(ci[1]-1):+.2f}%]',('상한 이내' if ci[1]<1.02 else ('상한 초과' if ci[0]>1.02 else '미확정'))] for n,p,r,ci in zip(names,timing['p90_ms'],timing['ratio'],timing['ratio_simul95'])])
    oc=table(['후보 표면','속도','Privileged accuracy','같은 clean response 비율'],
       [[tag,n,pct(a['sensor'][key]['accuracy'][i]),pct(a['sensor'][key]['identical'][i])] for key,tag in [('weak_vs_null','Weak 대 local null'),('strong_vs_null','Strong 대 local null')] for i,n in enumerate(names)])
    doc=[
      '# v37 · 마지막 probe의 속도가 slip, 표면 구별, 충돌 예측에 미치는 영향',
      read('VERDICT_KO.md'),primary,
      'Local 평균은 세 local regime과 네 object × surface cell에 각각 같은 비중을 준다. Brier는 별도 물체가 다음 충돌에서 stick할 확률로 채점했다. Δ가 음수면 gate가 고정 32.5%보다 유리하다. 95% interval은 같은 world를 대응시켜 계산했다. [전체 수치](stats37.json) · [사전 분석 계획](ANALYSIS_PLAN_KO.md)',
      '## 속도를 높이면 어떤 표면을 더 잘 읽는가',behavior,
      '세 local regime 평균에서 +0.50 대비 slip 비율 변화는 '+', '.join(f'{n}에서 {100*local["extra_speed_change"][i][0]:+.1f} pp [{100*local["slip_delta_simul95"][i][0]:+.1f}, {100*local["slip_delta_simul95"][i][1]:+.1f}]' for i,n in enumerate(names[1:]))+'였다. pp는 percentage point이며 두 변화에 대해 동시에 95%를 보정했다.',discr,
      'AUROC는 강도가 높은 표면에 더 높은 posterior mean strength를 부여할 확률로, 같은 점수일 때 0.5를 부여한다. 서로 다른 물체와 orientation이 일상적 관측처럼 섞인 상태에서 계산했다. Weak 대 local null과 strong 대 weak/null의 두 속도 변화에 대해 총 4개 simultaneous interval을 썼다. 나머지 strong 대 local null은 보조 pointwise interval이다. 이 지표의 정답은 표면 강도이고, Brier의 정답은 별도 충돌의 stick 여부다.',
      '## 같은 speed에서 고정 보정보다 나은가',results,
      '각 행은 같은 speed, 같은 signed proposal, 같은 deployment observations를 가진 gate와 고정 32.5% 사이의 비교다. 각 행 안의 세 표 값은 사전에 정한 7개 Brier 대조의 simultaneous interval에 속한다. 표면의 빈도가 다른 집단에는 해당 regime별 loss를 그 집단 비율로 재가중해야 한다.',
      prevalence,'Weak와 local null의 비중을 서로 같게 놓고 strong 비중을 바꾼 재가중 결과다. 표 안의 수치는 Gate−32.5% Brier의 point estimate이다. 손익분기점은 이 혼합 가정 아래의 기술값이며 deployment에서 관측한 표면 비율이 아니다. [재가중 수치](prevalence37.json)',
      speedchange,
      '속도를 올리면 acquisition likelihood도 바뀌어 13개 deployment probe를 받을 물체가 달라질 수 있다. 네 번째 열은 실제 gate 선택만 각 속도에서 가져와 동일한 +0.50 deployment probabilities 위에서 재채점한 대조다. 이 열의 95% interval은 pointwise이고 실제 온라인 방법의 latency를 뜻하지 않는다.',
      diag,
      f'과거 자료로 고른 고정 보정량은 {100*load("fixed_validation37.json")["scale"]:.1f}%다. v36의 192개 archive에서 +0.50의 Brier로 한 번 고른 뒤 새 세 속도에 그대로 적용했다. 이 방법은 target과 같은 simulator family의 과거 labels를 이용한 보조 비교다. Soft 보정은 같은 βz에 posterior mean near-field strength를 곱했다. ¹분할 oracle은 각 환경에서 64개 deployment 물체의 label로 항상 보정 또는 건너뛰기를 고른 뒤 나머지 64개에서 점수화하고 양쪽 역할을 바꿔 평균했다. 이 값은 실제 정책이 쓸 수 있는 정보보다 많은 label을 요구한다. [과거 자료 및 계수](fixed_validation37.json)',
      '## 속도가 만들어 낸 센서 신호',oc,
      '이 계산에서는 latent 물체와 orientation, 두 후보 contact law를 모두 알려준 상태에서 하나의 noisy velocity로 어느 표면인지 추측한다. 두 표면의 prior가 같고 공통 Gaussian noise의 표준편차가 σ라면 최적 정확도는 Φ(‖μ₁−μ₂‖/(2σ))다. 동일 물체의 두 반응이 모두 stick이면 impulse가 계수에 무관해 clean velocity가 같을 수 있다. 이 표는 oracle accuracy, 위 표는 실제 posterior의 AUROC이므로 두 수치의 크기를 직접 빼지 않는다.',
      read('CHANNEL_KO.md'),
      '![Speed, slip and decision loss](v37.png)',
      '## 보정까지 포함한 GPU p90',ti,
      'RTX PRO 6000 Blackwell Server Edition 한 장에서 환경 8개와 총 1,024개 deployment 물체를 처리하는 한 호출을 측정했다. 각 방법·속도당 8개 timing block × 3회, 총 24회이며 block 안에서 여섯 조합의 실행 순서를 무작위화했다. 속도별 likelihood, 취득할 관측 선택, 선택 관측의 decoding, CPU에서 계산한 correction과 host 반환을 포함한 synchronized wall time이다. Bank 초기 생성, simulated data 생성, 실제 접촉 및 robot reset 대기는 제외했다. [GPU 코드](run37.py) · [완료 기록](results/completion.json)',
      read('DESIGN_KO.md'),
      read('CONTRIBUTION_KO.md'),
      '## 문헌 근거와 후속 연구 질문',read('LITERATURE_KO.md'),
      '## 부록 및 재현',
      read('APPENDIX_KO.md'),
      f'서버 GPU raw archive {a["verified_archives"]}개를 CPU에서 다시 열어 Brier와 query labels, 세 속도의 slip/velocity, seed와 archive hash를 검산했다. 실행에 사용된 파일 {a["source_hashes"]}개의 hash와 서로 다른 seed ID {a["distinct_seeds"]}개를 확인했다. [재현 방법](REPRODUCE_KO.md) · [독립 계산](stats37.py) · [source 고정 및 확인](SOURCE_KO.md). 서버 정본은 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_speed_v37/REPORT_KO.md`에 있다.'
    ]
    txt='\n\n'.join(doc)+'\n';(R/'REPORT_KO.md').write_text(txt)
    work=R.parents[2]
    if (work/'work').exists() and (work/'outputs').exists():
        prefix='../work/physics-scene-belief/probe_speed_v37/'
        rewrite=lambda m:m[0] if '://' in m[1] or m[1].startswith(('/','#')) else ']('+prefix+m[1]+')'
        (work/'outputs/physics_scene_belief_probe_speed_v37_report.md').write_text(re.sub(r'\]\(([^)\s]+)\)',rewrite,txt))
if __name__=='__main__':main()
