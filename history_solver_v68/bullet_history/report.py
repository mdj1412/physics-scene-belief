import json,sys,os
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent; H=B.parent
if not (B/'forecasts_sealed.json').is_file():raise RuntimeError('no forecast seal')
r=json.loads((B/'external_history_result.json').read_text())
families=('Wendland','Dispersive')
laws=('strengthening','weakening')
n=np.zeros((12,2,2,4),dtype=int)
sensor=[]
for k in range(12):
    sub=f'panel_{k:02d}'
    sensor.append(json.loads((B/'tapes'/sub/'diagnostics.json').read_text()))
    for zf in (H/'bullet_labels'/sub).glob('*.npz'):
        f,s,l,v,h=zf.stem.split('_');i=[x.lower() for x in families].index(f);j=laws.index(l)
        with np.load(zf) as z:
            valid=z['valid'];speed=z['speed']
            n[k,i,j]+=[valid.size,valid.sum(),(valid&(speed<=.020)).sum(),(valid&z['loaded_end']).sum()]
assert int(n[...,1].sum())==r['valid'] and int(n[...,0].sum())==r['total']
sensors={key:sum(a[key] for a in sensor) for key in ('shots','unloaded','loaded_end')}
(B/'validity.json').write_text(json.dumps(dict(per_catalog_family_law=n.tolist(),columns=['total','valid','stick','loaded_end'],
                            sensor_counts=sensors,sensor_details=sensor),indent=2))
def fmt(x):return f'{x[0]:+.6f} [{x[1]:+.6f}, {x[2]:+.6f}]'
lines=['# v68 · PyBullet로 첫 calibration부터 마지막 impact까지 실행한 보조 검정','',
'두 family, 두 새 law, 12개 catalog 전부를 PyBullet의 rigid frictional contact로 재생했다. 각 환경의 128개 candidate 전체로 13회의 probe 배정을 결정하고, 사전에 정한 환경별 첫 8개 prototype에서 평가했다. 4개 source axis와 8개 head는 v67에서 학습한 값을 고정했다. 네 차례 calibration에 대한 v38 log-odds correction도 다시 학습하지 않았다. [사전 계획](PLAN_KO.md) · [기존 PyBullet terminal 설계](../BULLET_PLAN_KO.md)','',
'## 방법과 판정','',
'같은 12개 catalog × 4개 환경에는 서로 다른 노이즈 반복과 shape·surface·law의 짝지은 개입이 있다. 비교의 독립 단위는 12개 catalog다. 둘째 engine을 사용하는 단계는 객관적으로 확인할 수 있다. ① calibration에 쓸 네 물체의 factual contact, ② 세 far probe와 네 번째 query-matched probe, ③ deployment candidate의 factual contact, ④ 선택된 물리 probe, ⑤ terminal impact velocity를 모두 PyBullet에서 계산했다. Query 영상 geometry와 optical route, sigma 및 noise seed는 공통이다. Deployment용 미구매 센서는 GPU 입력에서 NaN으로 가렸다. Calibration은 최종 log odds를 조절하고 관측 선택은 factual channel에 반응한다. [센서 생성](history.py) · [GPU 예측](forecast.py)','',
'Factual/paid 접촉은 semi-infinite mesh 위 회전 고정 sphere(반경 0.25 mm), 20 μs step, 0.80 ms horizon, 0.10 mm mesh로 모사했다. 새 law는 최초 접촉 위치와 진입 접선 속도에서 계수를 정한다. Height 0의 10 mm 표면 mesh ordinate가 완전히 일치할 때만 계산을 공유했다. Signed velocity의 순서는 이전 factual feature와 같다. Geometry channel은 기존 tape에서 가져왔다. 센서용 simulated shot는 총 '+f"{sensors['shots']:,}"+'회였고 미접촉 '+str(sensors['unloaded'])+'회, 마지막 step에 여전히 하중이 걸린 경우 '+str(sensors['loaded_end'])+'회였다. 이 수에는 동등한 mesh의 계산 재사용이 반영돼 있다. [입력과 forecast checksum](forecasts_sealed.json) · [모든 counts](validity.json)','',
'## Held-out Brier loss','',
'같은 terminal label 및 같은 paid budget으로 비교했다. 차이는 낮을수록 왼쪽 방법에 유리하다. 6개 secondary contrast에 Bonferroni 동시 95% paired t interval을 사용했다. 두 family 반전은 각 법칙에서 Wendland의 구간 전체가 0 아래, Dispersive의 구간 전체가 0 위일 때 확인한다. 두 family를 평균한 복원 이득은 restored−dropped 구간 전체가 사전 기준 −0.002보다 낮을 때 확인한다.','',
'| PyBullet law | Wendland soft − full | Dispersive soft − full | 두 family 평균 restored − dropped |','|---|---:|---:|---:|']
for j,l in enumerate(laws):
    lines.append(f"| {l} | {fmt(r['soft_full'][0][0][j])} | {fmt(r['soft_full'][0][1][j])} | {fmt(r['restored_dropped'][0][j])} |")
lines+=['','| Family | Law | Full | Soft | Hard | Dropped | Restored | stick / valid / total |','|---|---|---:|---:|---:|---:|---:|---:|']
for i,f in enumerate(families):
    for j,l in enumerate(laws):
        total,valid,stick,loaded=n[:,i,j].sum(0)
        lines.append('| '+' | '.join([f,l,*[f'{v:.6f}' for v in r['abs_brier'][0][i][j]],f'{stick:,} / {valid:,} / {total:,}'])+' |')
lines+=['','| 같은 PyBullet terminal, 다른 이력 | Law | Wendland soft − full | Dispersive soft − full | 두 family 평균 restored − dropped |','|---|---|---:|---:|---:|']
for m in (1,2):
    for j,l in enumerate(laws): lines.append('| '+' | '.join([r['order'][m],l,*[f"{r['soft_full'][m][i][j][0]:+.6f}" for i in range(2)],f"{r['restored_dropped'][m][j][0]:+.6f}"])+' |')
delta=r['route_change_compliant_to_Bullet'][0]*100
lines+=['',f'동일 후보군에서 compliant와 PyBullet history 사이에 query/probe 선택이 바뀐 비율은 {delta:.3f}%였다. 세 이력의 절대 risk, 12개 catalog별 mean, source seed별 차이를 JSON에 보관했다. [채점 결과](external_history_result.json) · [채점 코드](score.py)','',
'이 분석은 기존 primary 설계 이후, 새 외부 terminal outcome을 이 분석자가 열기 전에 추가로 계획했다. 다른 프로세스가 같은 표본의 외부 terminal label을 생성하고 있었으며 GPU 예측기는 terminal 파일을 읽지 않았다. Sensor scene은 두 엔진 간 비교를 가능하게 하는 동일 생성기·law와 제한된 sphere geometry를 사용한다. 실물 접촉이나 새로운 object population으로의 전이를 평가하려면 별도의 holdout이 필요하다.']
ss=np.asarray(r['soft_full'][0]);rr=np.asarray(r['restored_dropped'][0])
passed=bool(np.all(ss[0,:,2]<0) and np.all(ss[1,:,1]>0) and np.all(rr[:,2]<-.002))
lines.insert(2,("두 friction law 모두에서 여섯 사전 방향·margin을 충족했다. " if passed else "여섯 사전 판정 가운데 모두를 통과하지 못했다. ")+"Wendland와 Dispersive 각각에서 likelihood 억제의 부호를 검사했고, 두 family 평균에서 좌표 복원 이득을 별도로 검사했다. 네 source axis의 family별 억제 부호는 각 법칙에서 동일했다." if (np.all(np.array(r['seed_soft'])[0]<0) and np.all(np.array(r['seed_soft'])[1]>0)) else "각 source axis에 대한 개별 수치는 별첨 JSON에 남겼다.")
lines.insert(3,"")
(B/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
