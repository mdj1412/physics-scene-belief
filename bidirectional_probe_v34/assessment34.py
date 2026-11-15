"""Rendering helper. Draw conclusions from paired CIs, not raw rankings."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
read=lambda f:json.loads((R/f).read_text())
def iv(v,c):return f'{v:+.5f} [{c[0]:+.5f}, {c[1]:+.5f}]'
def main():
    s=read('decision_stats34.json');q=read('selector_stats34.json');t=read('terminal_stats34.json');ext=read('external34.json')
    rate=np.array(s['means']);a=s['aug'];c=s['refit_band']
    orient='단방향 paid prefix에도 friction 방향 정보가 있었다. 한 방향으로 2·4·8회 접촉했을 때 독립 표면 방향 분류의 Brier는 0.133·0.077·0.035였고, 양방향 교대는 0.123·0.072·0.025였다. Factual-only의 0.250은 양쪽 방향에서 같은 관측을 내는 설계와 일치한다. 2회와 8회에서 교대가 유리했지만, family tag 없이 단일 readout을 옮긴 보조 실험에서는 교대의 추가 이득 구간이 모두 0을 포함했다.'
    txt=f'**표면 방향을 읽을 수 있다는 것과 다음 probe를 유리하게 배분할 수 있다는 것은 구분해야 한다.** {orient}'
    def classify(b):
        if b[1]<0:return '개선'
        if b[0]>0:return '악화'
        return '우열 미확정'
    full=f'충돌의 stick 여부를 채점했을 때, **estimated-sign stopping head의 교대 − 단방향 Brier는 {iv(a[10],c[10])}**였다. 같은 gain 모델에 sign을 추가한 효과(estimated − neutral)는 단방향 {iv(a[0],c[0])}, 양방향 {iv(a[5],c[5])}였다. 각각 {classify(c[0])}, {classify(c[5])}에 해당한다. 아래에서 같은 정보가 최종 확률의 calibration에는 어떤 효과가 있었는지 분리해서 보여준다.'
    (R/'FINDINGS_KO.md').write_text(txt+'\n\n'+full+'\n')
    stage=[]
    for j,name in enumerate(('단방향','양방향')):
        i=3*j;k=4*j
        stage.append(f'{name}에서 estimated cue를 넣었을 때 sensor selection의 Brier는 neutral 대비 {iv(q["delta"][k],q["ci"][k])}, 이미 같은 관측을 선택한 뒤 terminal decoder의 Brier는 {iv(t["delta"][i],t["ci"][i])}였다.')
    e=f'V33의 다른 물체에 모델을 그대로 이전한 단방향 진단에서는 estimated cue의 stopping 효과가 {iv(ext["delta"][0],ext["ci"][0])}, terminal 효과가 {iv(ext["delta"][2],ext["ci"][2])}였다.'
    conc='**논문의 중심 주장.** 방향을 식별하는 문제와 그것을 downstream 위험을 낮추는 위치에 쓰는 문제를 분리한 점이 기여 후보다. 근거는 새 합성 물체에서 사전 고정한 intervention을 수행하고, 동결한 scorer의 정보 손실을 동일 용량 판독기로 확인한 뒤, 동일 probe quota에서 sign을 stopping·sensor selection·terminal decoding 세 위치에 따로 넣은 실험이다. 물체를 움직여 마찰을 알아내는 방법, 양방향 마찰 측정이나 decision utility를 쓰는 것 자체에는 선행연구가 있다. [실물에서의 방향별 friction](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [active pushing/pulling](https://arxiv.org/html/2411.09020v1) · [downstream decision utility](https://openreview.net/pdf?id=zBG7WogAvm)'
    lim='**주장의 경계.** 표면 orientation은 합성 법칙의 binary sign이고, contact regime과 속도 크기는 고정했다. 동일 물체를 되돌려 측정한 실험이 아니며 서로 다른 calibration 물체의 friction과 restitution이 달라진다. 반복적인 reversal 그 자체가 도움이 된 것인지, 두 운동 방향을 한 번씩이라도 포함하면 충분한지는 `--++`와 `-+-+`를 따로 비교해야 한다. Gain head는 맞은편 family 하나의 독립 blocks 8개에서 학습하며, 한쪽에서 loss가 줄지 않은 것이 모든 scorer의 불가능성을 뜻하지 않는다. Family를 뺀 분석에는 fold별 classifier/head를 사용했다. Real deployment에서 family에 맞는 head를 공짜로 고를 수 있다는 주장은 하지 않는다. 단일 reader의 transfer 확인은 방향 판독에 관한 결과지 모든 downstream heads에 대한 실증이 아니다.'
    future='**다음 검증.** 먼저 한 표면의 날짜·마모·reset이 다른 session에서 독립 pulling test로 마찰 방향을 확정하고, calibration 물체와 deployment 물체를 나눈다. 각 block에서 처음 방향을 무작위화한 뒤 unilateral, alternating, 같은 방향별 접촉 횟수를 유지한 grouped order를 비교한다. 하나의 공유 모델을 보지 않은 물체와 surface session에 그대로 배포하고, 독립 sign Brier와 실제 stick Brier, motion/reset을 포함한 end-to-end p90을 따로 잰다. 새 방법의 채택 기준은 Brier 0.003과 같은 사전 기준 또는 접촉 시간을 반영한 비용으로 정해야 한다. Simulator에서 확인된 정보 판독만으로 실제 로봇의 비용 절감을 주장하지 않는다.'
    (R/'ASSESSMENT_KO.md').write_text('\n\n'.join(['**어디에서 차이가 생겼나.** '+' '.join(stage)+' '+e,conc,lim,future])+'\n')
if __name__=='__main__':main()
