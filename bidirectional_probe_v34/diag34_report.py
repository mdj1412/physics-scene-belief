from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results_diag'
read=lambda s:json.loads((OUT/s).read_text())
def interval(a,b):
    return f'{a:+.5f} [{b[0]:+.5f}, {b[1]:+.5f}]'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
def main():
    large=read('raw_analysis.json');small=read('small_analysis.json');eq=read('equal_scalar_analysis.json')
    rows=[]
    for j,k in enumerate((2,4,8)):
        rows.append([k,f'{large["mean"][0][j]:.5f}',f'{large["mean"][1][j]:.5f}',interval(large["delta"][j],large["delta_simultaneous95"][j]),
                    f'{large["accuracy"][0][j]*100:.1f}%',f'{large["accuracy"][1][j]*100:.1f}%'])
    mrows=[]
    for j,s in enumerate(('한 방향','교대')):
        for i,k in enumerate((2,4)):
            v=small['means'][j][i]
            mrows.append([s,k,*[f'{x:.5f}' for x in v[1:]]])
    srows=[]
    for j,s in enumerate(('한 방향','교대')):
        for i,k in enumerate((2,4)):
            v=eq['mean'][j][i]
            srows.append([s,k,*[f'{x:.5f}' for x in v],interval(eq['delta'][j][i],eq['ci'][j][i])])
    lines=['## Paid prefix에 surface orientation이 담겼는가?',
           '같은 물체를 양쪽 surface orientation에 놓고 trajectory와 sensor noise를 짝지었다. 마찰 방향을 분류하는 정답은 surface orientation이고, deployment에서 최종적으로 예측하는 정답은 별도 collision의 stick 여부다. 각 조합의 classifier는 평가할 물체와 표면을 모두 교체한 한 개 조합으로 학습했다. 예를 들어 rounded–fine에서 쓸 classifier는 ridged–coarse만 보고 학습했다. 각 policy가 배포될 평가 조합에 맞는 classifier를 미리 선택하는 조건이다. 어느 쪽 sign인지는 classifier에 제공하지 않았다.',
           '학습에는 조합마다 독립 blocks 64개, 평가에는 48개를 사용했다. 각 block은 8개 coupling 환경을 포함한다. 전체 test는 독립 blocks 192개, 독립 calibration pools 1,536개다. 두 signs, 두 schedules, 2·4·8회 stages는 해당 pools에서 여러 번 측정한 값이다. 처음 진행 방향을 양쪽으로 균형 배정해 출발 방향과 물체 구성이 특정 orientation을 알려 주지 않게 했다. [계획과 seed 범위](DIAG_PLAN_KO.md) · [분리된 classifier와 학습 지문](diag_models.json)',
           table(['Paid probes','한 방향 orientation Brier','교대 orientation Brier','교대 − 한 방향 [simultaneous 95%]','한 방향 accuracy','교대 accuracy'],rows),
           '2회만 한 방향으로 접촉해도 새 물체·표면에서 surface orientation을 80.8% 구별했다. 교대로 접촉하면 2회와 8회에서 확률의 Brier가 낮아졌다. 2회의 accuracy는 80.6%여서 같은 변화가 없었다. 4회의 interval에는 0이 포함됐다. 2회 차이의 upper bound는 −0.00007로 0에 가깝다. 각 stage마다 ± orientation을 동일 비중으로 계산했다. 95% interval은 고정된 네 family와 학습된 readout에 조건부이며 조합 안의 paired blocks를 16,000회 재표집했다.',
           'Raw readout은 접촉 전후에 측정한 velocity로부터 normal·tangential impulse 비율을 근사하고, 이를 실제로 준 action 방향과 결합한 features를 쓴다. 주어진 incoming 방향, paid observations와 factual contour만 사용한다. 방향이 반대인 두 환경의 factual prefix는 같은 값이어서 factual-only orientation Brier는 0.25000이었다. Calibration을 한 뒤에도 각각의 쌍에서 target orientation을 무작위로 뒤집어 다시 학습·평가하면 0.254~0.258이었다. [입력과 proxy 식](diag34_model.py) · [여섯 조건의 원결과](results_diag/raw_analysis.json)',
           '### 기존 scorer에서 방향을 되읽을 수 있는가?',
           '똑같은 training·test histories에 source posterior likelihood를 다시 계산하고, 각각의 15 features와 첫 1×1 frozen gain score를 저장했다. 각 조합별 학습·test 각 8 blocks, test 전체 32 blocks를 썼다. 다음 값은 모두 orientation Brier다. Linear와 RBF 모두 start 방향을 제공하고 같은 logistic penalty를 썼다. Raw도 이 표에는 동일한 소규모 train split에서 다시 fit한 값을 사용했다.',
           table(['Schedule','Paid probes','Raw','15 features · linear','Gain scalar · linear','15 features · RBF','Gain scalar · RBF'],mrows),
           'Alternating에서 15 features나 gain score만으로 새 family의 방향을 분류한 Brier는 0.25 부근 또는 그보다 높았다. 방향별 마찰 반응 차이를 알려면 어느 방향에서 어느 반응이 관측됐는지 알아야 한다. 현재 15 features는 motion 방향을 공통 좌표로 변환한 뒤 velocity의 평균·분산 등으로 요약한다. 한 방향 조건에서는 같은 scalar나 feature에서도 일부 orientation 정보를 되읽을 수 있었다. 이 비교가 정량화하는 것은 두 지정된 readout에서의 OOD decodability다. Frozen scorer는 원래 gain을 학습했으므로 orientation 분류가 나쁘다는 사실만으로 더 좋은 stopping rule의 존재가 입증되지는 않는다. [16개 비교의 simultaneous interval](results_diag/small_analysis.json)',
           '### 양쪽 readout의 차원을 하나로 맞추면?',
           '큰 raw feature set의 효과를 본 뒤, 다시 생성한 제3 test cohort에서 두 입력을 모두 scalar 하나로 제한했다. Physics scalar는 각 접촉의 추정 friction이 ± orientation에서 나올 확률비를 nuisance friction 후보 네 개에 대해 근사하고 이력에 걸쳐 더한다. Frozen scalar는 기존 모델이 출력한 expected gain이다. 각각 같은 13개 predictor terms와 intercept의 logistic readout을 같은 source folds로 학습했다. 32개 fresh paired blocks로 평가했다. Physics scalar는 simulator friction 분포의 근사치를 사전 지식으로 사용한다. 두 scalar의 정보 차이와 여기에 투입한 domain knowledge가 함께 반영된다. [실험 시점과 계획](DIAG_SCALAR_PLAN_KO.md) · [독립 test 실행 코드](diag34_scalar.py)',
           table(['Schedule','Paid probes','Physics scalar','Frozen gain scalar','Frozen − physics [simultaneous 95%]'],srows),
           '추가 접촉의 가치를 판단할 때에는 같은 예산에서 실제 observation이 달라지고 최종 stick Brier가 줄어드는지도 함께 보아야 한다. 여기서 채택한 family별 readout은 같은 geometry·surface를 직접 fit하지 않고 반대 family에서 학습했다. 다른 표면 전체나 실제 물체에 대한 orientation accuracy에는 별도 검증이 필요하다. [동일 차원 보조 분석 원자료](results_diag/equal_scalar_analysis.json)']
    universal=read('universal_analysis.json')
    rows=[[k,f'{universal["mean"][0][s]:.5f}',f'{universal["mean"][1][s]:.5f}',interval(universal['delta'][s],universal['ci'][s])] for s,k in enumerate((2,4,8))]
    lines+=['### 물체·표면 이름 없이 같은 classifier를 쓰면?', '앞의 진단을 본 뒤 family를 입력하지 않는 조건을 추가했다. 중간 geometry와 directional response를 가진 하나의 새로운 synthetic reference family에서만 classifier를 학습했다. 이 모델을 네 조합에 공통 적용하고, 이전 모든 panels와 seed가 겹치지 않는 64개 paired blocks에서 결과를 평가했다. 세 stage 모두 한 방향과 교대 양쪽에서 0.25보다 낮은 orientation Brier를 기록했다. 이 새 cohort에서 교대의 점추정은 모두 더 낮았지만 세 interval은 0을 포함했다. 어느 schedule에서든 방향을 읽을 수 있다는 결과가 유지됐고, 교대의 추가 효과 크기에는 source와 test 구성이 영향을 줄 수 있다. [추가 설계와 source 계수](DIAG_UNIVERSAL_PLAN_KO.md) · [보조 분석 전체 결과](results_diag/universal_analysis.json)',table(['Paid probes','단일 source model 한 방향','단일 source model 교대','교대 − 한 방향 [simultaneous 95%]'],rows)]
    (ROOT/'DIAG_RESULTS_KO.md').write_text('\n\n'.join(lines)+'\n')
if __name__=='__main__':main()
