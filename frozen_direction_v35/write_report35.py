"""Build the canonical Korean report and workspace reading copy from immutable results."""
import json, re
from pathlib import Path
from statistics import mean
D=Path(__file__).resolve().parent
def load(n):return json.loads((D/n).read_text())
def read(n):return (D/n).read_text().strip()
def lines_table(labels,rows):return '\n'.join(['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(labels))+'|',*['| '+' | '.join(row)+' |' for row in rows]])
def ci(x):return f'[{x[0]:+.5f}, {x[1]:+.5f}]'
def pct(x):return f'{100*x:.1f}%'
def sr(x):return f'{x:+.5f}'
def f5(x):return f'{x:.5f}'
def risk(n):
    if n['practically_replicated']:return '실용 기준 통과'
    if n['improved']:return '개선, 실용 기준 미확정'
    return '개선 미확정'
def ms(x):return f'{x:,.1f}'
def main():
    a=load('audit35.json');stress=load('transfer_stress35.json');scal=load('source_cal35.json');mech=load('physics_audit35.json')
    micro=load('head_cost35.json');t=a['timing']; r=a['risks']; arms=('same','alternating');names=('한 방향','양방향 교대')
    rows=[];times=[];groups=[];checks=[];ctl=[];mvs=[]
    for j,(arm,k) in enumerate(zip(arms,names)):
        x=r[arm]; ov=x['overall']; band=x['simultaneous_ci95']
        rows.append([k,f5(ov['original']),f5(ov['corrected']),sr(ov['delta']),ci(band),risk(x)])
        tr=t['ratio_simultaneous_ci95'][j]
        times.append([k,*[ms(v) for v in t['p90_ms'][j]],f'{(t["ratio"][j]-1)*100:+.2f}%',f'[{(tr[0]-1)*100:+.2f}%, {(tr[1]-1)*100:+.2f}%]', '충족' if t['within_2pct'][j] else '미확정'])
        checks.append([k,pct(ov['direction_accuracy']),f5(ov['direction_brier']),sr(ov['wrong_cue']),sr(ov['true_cue']),', '.join(sr(ov['fold'+str(i)]) for i in (1,2,3))])
        mvs.append([k,f5(ov['quadratic']),f5(-ov['alignment']),sr(ov['delta'])])
    for cell in ('sinusoid_atan','sinusoid_algebraic','quartic_atan','quartic_algebraic'):
        groups.append([cell.replace('_',' × '),*[sr(r[arm]['groups'][cell]['delta']) for arm in arms]])
    ss=[]
    for reg,label in [('full','같은 directional law'),('half','query amplitude 절반'),('zero','query만 isotropic'),('reverse','query에서 부호 역전'),('S1_abs_oriented','20개 측정 magnitude를 query에 대입')]:
        vals=[stress[reg][s] for s in arms]
        ss.append([label,pct(vals[0]['changed_label_fraction']),*[sr(v['delta'])+' '+ci(v['pointwise95']) for v in vals]])
    for arm,x in zip(arms,names):
        vals=scal['isotropic']['absolute_brier'][0][arms.index(arm)]
        met=scal['isotropic']['methods']
        ctl.append([x,', '.join(met),', '.join(f5(v) for v in vals)])
    tbl0=lines_table(['Calibration motion','원래 physics','Frozen correction','Δ Brier ↓','Simultaneous 95% interval','판정'],rows)
    tbl1=lines_table(['Calibration motion','원래 p90 (ms)','보정 포함 p90 (ms)','변화','Simultaneous 95% interval','+2% 상한'],times)
    tbl2=lines_table(['새 물체 × 표면','한 방향 Δ','양방향 교대 Δ'],groups)
    tbl3=lines_table(['Calibration motion','방향 accuracy','방향 Brier','부호를 잘못 뒤집으면 Δ','실제 부호를 주면 Δ','다른 세 frozen 쌍 Δ'],checks)
    tbl4=lines_table(['Calibration motion','확률을 옮긴 비용 E[d²]','residual과 맞은 정도 2E[d(y−p)]','차이 Δ'],mvs)
    tb5=lines_table(['query law만 바꾼 조건','기존 정답이 바뀐 비율','한 방향 Δ [95%]','양방향 교대 Δ [95%]'],ss)
    text='\n\n'.join([
      '# v35 · 네 번의 probe로 읽은 friction 방향은 새 물체의 충돌 예측에도 유효한가?',
      read('VERDICT_KO.md'),tbl0,
      'Δ는 같은 world·같은 관측에서 **보정 − 원래 physics decoder**다. 음수가 개선이다. 사전에 정한 실용 기준은 interval 상한이 −0.003보다 작은 것이다. 두 schedule의 구간은 동시에 보정했다. [전체 통계와 원자료 검산](audit35.json)',
      '## GPU p90: 방향을 읽고 보정하는 시간까지 포함한 비용',tbl1,
      f'장치는 **{load("results/run.json")["gpu"]}** 한 장이다. 한 호출은 물체 1,024개를 함께 처리한다. 각 방법·schedule당 {t["timed_calls_per_method_schedule"]}회, 독립 timing block은 {t["timed_blocks"]}개다. 반복별 실행 순서를 바꿨다. +2%는 시간 증가에 대해 미리 정한 허용 상한이다. 이 한 세션에서 얻은 interval이며 평균 속도나 단일 물체의 latency로 읽으면 안 된다.',
      f'보정 계산만 따로 실행한 server CPU microbenchmark의 p90은 한 방향 {micro["schedule"]["same"]["p90_ms"]:.3f} ms, 교대 {micro["schedule"]["alternating"]["p90_ms"]:.3f} ms였다. 원래 확률이 이미 host에 도착한 시점부터 paid rows 추출·reader·1,024개 odds update를 포함한다. 이 값을 전체 GPU p90끼리 뺀 값으로 대체할 수 없다. 서로 다른 실행의 p90은 더하고 뺄 수 있는 단계별 시간이 아니다. [측정 원자료](results/run.json) · [보정 단계만의 측정](head_cost35.json) · [실행 범위](REPRODUCE_KO.md)',
      read('DESIGN_KO.md'),
      '## 어느 물체에서 얼마나 좋아졌나?',tbl2,
      '이 표는 조합별 point estimate다. 네 조합은 전체 물체·표면에서 무작위 추출한 표본이 아니다. 같은 실험의 방향 판독 성능과 잘못된 부호를 넣었을 때의 결과를 함께 보면 cue가 성공한 이유와 실패 경계를 더 좁힐 수 있다.',tbl3,
      '부호 반전은 추정 cue에 −1을 곱한 대조, 실제 부호는 q 추정을 정답으로 바꾼 진단이다. 둘 다 같은 β를 쓴다. 실제 부호를 제공한 진단은 배포 가능한 방법이나 모든 모델에 대한 하한이 아니다. 다른 세 쌍은 v34 모델을 재학습하지 않고 교체한 sensitivity이고, 그중 가장 좋은 값을 보고 모델을 고르지 않았다.',
      '## 무엇이 개선을 설명하는가?',tbl4,
      'd=p′−p라 하면 ΔBrier=E[d²]−2E[d(y−p)]가 정확히 성립한다. 첫 항은 예측을 움직인 것 자체의 비용이다. 둘째 항은 움직인 방향과 크기가 원래 예측의 residual에 맞았을 때 생기는 이득이다. 방향 분류 accuracy가 높더라도 두 번째 항이 첫 번째 항보다 작으면 Brier가 나빠진다. 아래에서는 paid evidence와 예측은 그대로 두고 query 정답만 바꾸어 이를 확인했다.',
      tb5,
      '원래 query의 정답은 각 archive의 θ로 독립적으로 다시 계산했고 전체 원래 label과 일치해야 표를 생성하게 했다. 나머지 행은 동일 물체의 query에서만 directional friction을 약화·제거·반전한 반사실이다. 마지막 행은 Tramsen et al.의 공개 S1에 있는 20개 normalized amplitude의 절댓값을 같은 가중치로 사용한 sensitivity다. query의 부호는 원래 환경과 일치시킨 채 magnitude만 바꿨다. reader에는 정답 부호를 주지 않았다. 실물에서 얻은 stick label이나 실물 Brier가 아니다. 이 표의 interval은 조건별 pointwise 95%로 primary 판정에 쓰지 않았다. [사전 기준](TRANSFER_STRESS_PLAN_KO.md) · [계산값](transfer_stress35.json) · [S1 원자료](https://doi.org/10.6084/m9.figshare.c.3950773.v2)',
      read('INTERPRETATION_KO.md'),f'이 panel에서는 역전 시 정답 변화율이 {pct(stress["reverse"]["same"]["changed_label_fraction"])}여서, 위의 균등 혼합 예에서 하한은 {f5(.25*stress["reverse"]["same"]["changed_label_fraction"])} Brier다.',
      '![Direction transfer and full GPU latency](v35.png)',
      read('CONTROL_KO.md'),
      read('SOURCE_CAL_KO.md').replace('# v35에서','## v35에서',1),
      read('CONTRIBUTION_KO.md'),
      read('LITERATURE_KO.md').replace('# v35 문헌 검토와 논문 기여 평가','## 관련 문헌과 후속 연구 질문',1),
      '## 재현 및 검증 기록',
      f'Primary와 control을 합친 raw archive {mech["archives_verified"]}개와 source 파일 {mech["input_files_verified"]}개의 hash를 별도로 검산했다. Evidence·평가 seed 분리, paid IDs, deployment quota, 충돌 식과 label도 확인했다. [실험 코드](run35.py) · [데이터 생성](target35.py) · [실행 방법](REPRODUCE_KO.md) · [독립 통계 재계산](audit35.py) · [physics 및 hash 검산](physics_audit35.json) · [같은 v34 fold의 역사적 점수](source_replay35.json) · [run 완료 기록](results/completion.json). 서버 경로: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/frozen_direction_v35`.',
    ])+'\n'
    (D/'REPORT_KO.md').write_text(text)
    root=D.parents[2]
    prefix='../work/physics-scene-belief/frozen_direction_v35/'
    def fix(m):
        p=m.group(1)
        return m.group(0) if '://' in p or p.startswith(('#','/')) else ']('+prefix+p+')'
    rep=re.sub(r'\]\(([^)\s]+)\)',fix,text)
    if (root/'outputs').is_dir() and (root/'work'/'physics-scene-belief').is_dir():
        (root/'outputs'/'physics_scene_belief_frozen_direction_v35_report.md').write_text(rep)
if __name__=='__main__':main()
