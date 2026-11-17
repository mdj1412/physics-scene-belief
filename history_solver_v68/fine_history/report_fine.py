import json
from pathlib import Path
D=Path(__file__).resolve().parent
def fmt(a):
    return f'{a[0]:+.6f} [{a[1]:+.6f}, {a[2]:+.6f}]'
def main():
    data=json.loads((D/'fine_scores.json').read_text())
    seal=json.loads((D/'FORECASTS_SEALED').read_text())
    status=[json.loads(p.read_text())['48'] for p in sorted(D.glob('history_status_*.json'))]
    result=['# 전체 fine history와 전체 fine terminal outcome의 교차 검산','',
        '기본 해상도와 부분 표본의 fine 결과를 확인한 뒤 추가한 민감도 분석이다. 같은 12개 catalog 전체, 네 source axis, source weight, 네 개 calibration pair, acquisition 및 noise를 고정했다. Contact support를 9→17로 늘리고 step size를 절반으로 줄여 calibration, deployment factual, paid probe, terminal contact를 모두 다시 생성했다. 표의 9/17은 앞에서부터 관측 이력/terminal solver의 공간 해상도이고 temporal resolution도 함께 변한다. 두 hybrid row는 두 경로를 분리하기 위한 계산이다. 판정 단위는 이번 sensitivity에서 지정한 6개 paired contrast이며 각 interval은 해당 6개에 대해 보정했다. [계획](PLAN_KO.md) · [예측 봉인](FORECASTS_SEALED) · [원자료](fine_scores.json)','',
        '| History/terminal | Law | Wendland soft−full | Dispersive soft−full | 두 형상 평균 restored−dropped |','|---|---|---:|---:|---:|']
    for key,rec in data['contrasts'].items():
        for law,values in rec.items():result.append('| '+' | '.join([key,law,*map(fmt,values)])+' |')
    result+=['','| 해상도 변화, paired Δ | Law | Wendland soft−full | Dispersive soft−full | 두 형상 평균 restored−dropped |','|---|---|---:|---:|---:|']
    for key,rec in data['paired_changes'].items():
        for law,values in rec.items():result.append('| '+' | '.join([key,law,*map(fmt,values)])+' |')
    result+=['','| Fine/fine law | 형상 | Full | Soft | Hard | Dropped | Restored |','|---|---|---:|---:|---:|---:|---:|']
    for i,f in enumerate(['Wendland','Dispersive']):
        for j,law in enumerate(['strengthening','weakening']):
            result.append('| '+' | '.join([law,f,*[f'{x:.6f}' for x in data['absolute'][1][1][i][j]]])+' |')
    result+=['','미분리 trajectory의 binary label을 각각 두 가지 모두로 두어 얻은 동시 95% interval의 외곽이다. 각 법칙의 순서는 Wendland, Dispersive, 두 형상 평균 복원이다.','']
    for key,x in data['full_fine_unknown_envelopes'].items():result.append(f'- {key}: [{x[2]:+.6f}, {x[3]:+.6f}]')
    result+=['',
        f"측정용 fine contact 호출 {sum(x['total'] for x in status):,}행 중 horizon 연장 {sum(x['retry'] for x in status):,}행, 여전히 미분리 {sum(x['still_open'] for x in status):,}행이다. 원 coarse catalog 위험 재계산 최대 차이는 {data['original_replay_error']:.2g}였다. Fine GPU 예측 실행은 {seal['duration_s']:.1f}초였다. 이미 첫 32개 fine label의 경향을 알고 있었고 전체 fine label 생성은 별도 CPU에서 병행했다. 전체 fine history 예측을 별도 봉인한 후 scoring을 시작했다. [파이프라인](fine_predict.py) · [2×2 검산](score_fine.py)",'']
    (D/'FINE_HISTORY_KO.md').write_text('\n'.join(result))
if __name__=='__main__':main()
