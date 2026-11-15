"""Write numerical interpretation from preregistered primary and secondary summaries."""
from write_report35 import D,load,ci,f5,sr,pct,lines_table
a=load('audit35.json');b=load('source_cal35.json');s=load('transfer_stress35.json')
arms=('same','alternating');names=('한 방향','양방향 교대')
p=a['risks']['pooled']; g=p['overall']; iso=a['isotropic_control']; t=a['timing']
assert all(abs(s['full'][sc]['delta']-a['risks'][sc]['overall']['delta'])<1.e-9 for sc in arms)
vd='양쪽 schedule에서 사전에 정한 실용적 개선도 재현됐다.' if all(a['risks'][sc]['practically_replicated'] for sc in arms) else ('양쪽 schedule에서 개선을 확인했지만, 실용 기준은 아래 표처럼 나누어 판정해야 한다.' if all(a['risks'][sc]['improved'] for sc in arms) else '각 schedule에서 개선 여부와 실용 기준을 따로 판정해야 한다.')
def signword(v):return '낮아졌다' if v<0 else '높아졌다'
(D/'VERDICT_KO.md').write_text(f'''**새 synthetic 물체 두 종류와 directional surface law 두 종류를 교차한 결과, fixed 4의 held-out Brier는 원래 physics decoder {g["original"]:.5f}에서 v34의 frozen correction {g["corrected"]:.5f}로 {abs(g["delta"]):.5f} {signword(g["delta"])}.** 두 schedule을 같은 비중으로 둔 paired Δ의 95% interval은 {ci(p["ci95"])}다. {vd}

비교한 두 모델은 probe도 observation도 같은 것을 사용했다. 달라진 것은 다음 충돌에서 stick할 확률뿐이다. 방향성이 없는 별도의 표면에서는 오히려 pooled Brier가 {abs(iso["pooled_delta"]):.5f} {signword(iso["pooled_delta"])}. 따라서 보고할 수 있는 주장은 probe로 읽은 방향이 query 충돌에도 이어지는 조건에서의 **terminal prediction** 개선이다. 아래에 일반적인 score 보정만으로 설명되는 부분과, 같은 evidence가 있어도 보정이 실패하는 조건을 분리했다.''')
ct=[]
for j,k in enumerate(names):
    ct.append([k,*[f5(x) for x in iso['mean_brier'][j]],sr(iso['delta'][j]),ci(iso['simultaneous_ci95'][j]),f5(iso['mean_absolute_cue'][j])])
th=load('threshold35.json')['schedule']
def cross(sc):
    return ', '.join(f'{100*lo:g}~{100*hi:g}%' for lo,hi in th[sc]['crossings']) or '교차 없음'
(D/'CONTROL_KO.md').write_text('\n\n'.join(['## 방향이 실제로 없는 surface에서의 검증',
    '앞의 query-only stress와 달리 이 control은 calibration과 deployment 모두에 isotropic law를 사용했고, 별도 16개 seed block에서 처음부터 다시 실행했다. 물리적으로 참인 표면 방향이 없으므로 reader의 방향 accuracy는 정의하지 않았다.',
    lines_table(['motion','원래 Brier','보정 Brier','Δ','Simultaneous 95% interval','평균 |z|'.replace('|','∣')],ct),
    f'두 schedule 평균은 {sr(iso["pooled_delta"])} {ci(iso["pooled_ci95"])}였다. |z|는 reader가 중립값에서 얼마나 멀어졌는지 나타낸다. 두 방향 중 어느 것인지 구분하도록 학습한 reader에는 “표면에 방향이 없음”이라는 별도 상태가 없다. 가능한 두 부호를 잘 구분하는 것과, 방향이 없는 표면을 알아보는 것은 다르다.',
    f'탐색적으로 같은 query-only stress의 amplitude 비율을 0부터 150%까지 2.5%p씩 늘려 평균 Δ의 부호가 바뀐 구간도 확인했다. 한 방향 {cross("same")}, 교대 {cross("alternating")}였다. 기준은 각 표면 원래 amplitude의 백분율이다. 이 값은 네 synthetic 조합을 평균한 격자상의 교차이며 deployment gate나 실제 마찰 계수 threshold로 학습·검증한 것이 아니다. [수치와 전체 곡선](threshold35.json)']))
zz=b['directional']; ri=zz['absolute_brier'][0]; cc=zz['delta'][0]; iv=zz['joint_ci95'][0]
summary=[]
for j,n in enumerate(names):
    summary.append(f'{n}: frozen signed {ri[j][3]:.5f}, temperature {ri[j][2]:.5f}, temperature 뒤 같은 signed correction {ri[j][4]:.5f}. 마지막 두 방법의 차이는 {sr(cc[j][2])} {ci(iv[j][2])}.')
(D/'CONTRIBUTION_KO.md').write_text('\n\n'.join(['## 논문 기여 가능성: 무엇을 주장할 수 있나',
    '가장 설득력 있는 결과는 단순한 보정식 자체가 아니다. 이미 지불한 네 번의 접촉에서 읽은 정보를, 나중에 다른 물체가 stick할 확률에 어느 조건에서 옮길 수 있는지 prospective transfer와 실패 대조로 함께 검증했다는 점이다. Shape와 surface response를 바꾸고도 같은 reader와 계수를 썼고, 원래 decoder와 동일한 actions 및 실제 service boundary를 사용했다. Query에서 정보의 의미를 바꾸면 이득이 어떻게 사라지는지도 같은 예측으로 검증할 수 있다.',
    '일반적인 확률 보정과의 구분이 기여의 핵심이다. '+ ' '.join(summary)+' 여기서 temperature는 target 결과를 보기 전에 v34에서만 fit했다. 원래 physics에 이긴 결과와 이미 보정된 강한 baseline 위에 남은 결과를 구분해서 보고해야 한다. 합성 모델의 결과는 secondary이고 그 모델의 전체 GPU p90을 별도로 측정하지 않았다.',
    '**현재 증거의 한계.** 낮은 Brier는 held-out event probability가 더 정확했다는 증거다. 그것만으로 reliability 자체의 개선, 실제 friction mechanism의 식별, 더 나은 observation selection을 보인 것은 아니다. Acquisition과 budget은 고정했고 base decoder에는 실제 population coupling을 주었다. 하나의 simulator, 두 contour, 두 directional response와 고정된 query 방향을 사용했다. 마찰의 부호가 query의 접선 운동과 결합하는 방식은 그대로다. 실물에서의 성능 및 로봇 전체 cycle time도 측정하지 않았다.',
    '**제가 논문을 계속 쓴다면:** 지금 결과를 재현·반증이 가능한 핵심 simulation figure로 쓰겠다. 독립적인 robotics 방법 논문의 중심 주장으로 삼기에는 실물 transfer와 강한 물리 baseline이 아직 비어 있다. 다음 실험에서는 (1) 같은 네 번의 force·velocity로 directional friction의 연속크기와 isotropic 가능성을 함께 추정하는 physics decoder를 만든다. (2) probe와 다른 높이·속도에서 새 물체의 stick/slip을 관측하고 object ID와 surface session을 동시에 hold out한다. (3) 원래 네 번의 budget 안에서 query-matched probe 한 번을 쓰는 방법, frozen correction, temperature, 연속 physics model을 비교한다. 실험 전 event 판정과 Brier margin, paired session 수, sensor/reset 포함 latency를 고정한다. 물리 baseline 위에 남는 이득 또는 query-matched probe만이 해소하는 실패가 독립 자료에서도 재현될 때 알고리즘 기여 주장을 강화할 수 있다. 기존 공개 pulling 표에는 이 평가에 필요한 짝지어진 impact outcome이 없다.',
]))
