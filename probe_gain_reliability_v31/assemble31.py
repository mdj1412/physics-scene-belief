"""Assemble quantitative interpretation after both panels and independent audits."""
import json
from pathlib import Path
from summarize_precision31 import main as precision_summary
from finite_gain_ceiling31 import main as ceiling_summary
from replication31 import main as matched_summary
from report31 import main as generate

ROOT=Path(__file__).resolve().parent
def load(path):
    return json.loads((ROOT/path).read_text())

def main():
    precision_summary()
    ceiling_summary()
    matched_summary()
    pool=load('results_pool/crossed_stats31.json')
    replay=load('results_pool/portfolio31.json')
    precis=load('results/precision31.json')
    bounds=load('results/finite_gain_ceiling31.json')
    coverage=load('results_pool/runner_validation.json')
    groups=pool['pooled']
    a,b=pool['primary']['simultaneous95']
    concordance=[stage[2] for stage in groups['ranks']['estimate']]
    shares=groups['component_shares']['estimate']
    def percent(x):
        return f'{100*x:.1f}%'
    def stage_pair(vals,fn):
        return '·'.join(fn(float(x)) for x in vals)
    ranks=stage_pair(concordance,percent)
    ci=' 및 '.join(f'{100*(.5+lo):.1f}–{100*(.5+hi):.1f}%' for lo,hi in [a[1],b[1]])
    gains=stage_pair(groups['gain']['estimate'],lambda x:f'{x:+.5f}')
    signed=groups['gain']
    g_ci=', '.join('['+', '.join(f'{v:+.5f}' for v in pair)+']' for pair in signed['pointwise95'])
    each_share=[stage_pair([row[k] for row in shares],percent) for k in range(3)]
    inside=stage_pair([r[0] for r in replay['pooled']],lambda x:f'{x:+.5f}')
    out=stage_pair([r[1] for r in replay['pooled']],lambda x:f'{x:+.5f}')
    findings=[
        '**추가 calibration probe의 가치는 어느 deployment batch에서 재느냐에 크게 좌우됐다.** 같은 과거 접촉을 본 환경마다 독립적인 접촉 경로 4개와 미래 물체 집단 4개를 교차했다. 세 contact law, 144개 cohort에서 각 환경의 다음 접촉이 다른 환경보다 가치 있는지를 두 번 독립적으로 판단했다. 아래에서 두 수치를 나란히 적을 때는 probe를 2→4회, 4→8회로 늘린 순서다.',
        f'**이미 factual frame을 보유한 물체에 probe만 더 샀을 때**, 실제 held-out Brier gain은 평균 **{gains}**였다(pointwise 95% interval {g_ci}). Gain의 variance 중 calibration future의 main effect는 {each_share[0]}, deployment batch의 main effect는 {each_share[1]}, 두 축의 interaction은 {each_share[2]}였다. 같은 접촉 결과가 어떤 배포 집단에서는 선택을 개선하고 다른 집단에서는 악화시키는 변동까지 interaction에 포함된다.',
        f'두 환경 중 어느 쪽에 추가 probe를 주는 게 나은지, 양쪽 난수를 모두 바꿔 다시 매긴 **순서 일치율은 {ranks}**였다(simultaneous 95% interval {ci}, 무작위 기준 50%). 새 물체에 factual과 probe를 함께 추가한 별도 panel에서도 49.5%·49.8%였다. 한 번의 gain으로 선별한 네 환경을 그 자료에서 다시 채점하면 무작위 배분 대비 {inside} Brier만큼 좋아 보였으나, 같은 선택을 독립 반복에서 채점한 이득은 {out}였다. 전체 실험은 같은 prefix에서 접촉의 가치가 재현되는지 평가하며, 온라인에서 이 선별에 쓴 미래 정답을 이용할 수는 없다.'
    ]
    (ROOT/'FINDINGS_KO.md').write_text('\n\n'.join(findings)+'\n')
    overall=[]
    for n,key in [('새 물체','fresh'),('기존 물체','pool')]:
        x=bounds[key]
        overall.append(f"{n} {x[0]['upper_gain']:.5f}·{x[1]['upper_gain']:.5f}")
    portion=' / '.join(overall)
    nr=precis['results_pool']
    nx=[r['balanced_n']['0.002'] for r in nr]
    assessment=[
        '**환경 순위를 맞히는 문제의 실질적 크기.** 단일 stage에서 여덟 환경에 대한 실제 기대 gain을 μ, 네 곳에 probe를 주는 indicator를 w라 두면 균등 무작위 기준을 넘는 기대 이득은 여덟 환경에 대한 평균 (w−0.5)(μ−평균 μ)다. Cauchy–Schwarz로 이는 0.5 × √[(7/8) × 환경 간 sample variance] 이하다. Block·law에 같은 가중치를 주고 Jensen 부등식을 적용해도 성립한다. 이때의 variance는 finite grid noise를 뺀 기대 gain의 환경 간 분산이다.',
        f'그 분산의 pointwise 95% bootstrap interval 상단을 대입한 **탐색적 상한**은 {portion} Brier였다(2→4·4→8). 제한된 block에서 추정한 interval에 의존하므로 엄밀한 보장이나 모든 stopping policy의 상한으로 쓸 수 없다. 완벽한 기대 순위를 알면 이득이 반드시 이 상한에 도달한다는 뜻도 아니다. 실현된 단일 cell에서 상위 네 곳을 고른 낙관적 이득이 왜 그대로 배포 목표가 되기 어려운지를 크기까지 설명하는 진단이다. [분산과 상한](results/finite_gain_ceiling31.json) · [계산 코드](finite_gain_ceiling31.py)',
        f'**무엇을 더 반복할지.** 두 panel 모두 batch main effect가 future main effect보다 컸으므로 같은 cell 수에서 batches를 먼저 늘리는 편이 평균 gain의 sampling variance를 더 줄인다. 다만 interaction도 커서 동일 future만 계속 쓰면 독립 미래 접촉에 대한 검증은 부족해진다. 기존 물체 panel에서 한 prefix의 gain 추정 RMS를 0.002로 낮추려면 이 분산 모형의 plug-in 외삽으로는 각 축을 같은 수로 늘릴 때 {nx[0]}×{nx[0]} 및 {nx[1]}×{nx[1]} 수준이 필요하다. 이는 4×4 밖의 미검증 외삽이며, 물리 로봇이 사용해야 할 probe 수나 순위가 보장되는 표본 수로 해석해서는 안 된다. [RMS 외삽 코드와 결과](results/precision31.json)',
        (ROOT/'PAPER_DECISION_KO.md').read_text().strip()
    ]
    (ROOT/'ASSESSMENT_KO.md').write_text('\n\n'.join(assessment)+'\n')
    prov=(ROOT/'PROVENANCE_KO.md').read_text().split('\n- 기존 pool')[0].rstrip()
    duplicates=coverage['cross_future_duplicate_ids_by_stage']
    prov+=f"\n- 기존 pool에서 구입한 보충 실험은 GPU block 계산 약 {coverage['total_block_wall_seconds']:.1f}초였다. 두 stage에서 future 간 중복으로 재선택된 object ID 수는 {duplicates[0]}·{duplicates[1]}개다(전체 선택 수 1,152·2,304). 별도 CPU에서 RNG와 noise를 재생하여 저장된 contact response를 검산했고, 두 GPU 실행의 종료 기록에서 외부 GPU process와의 충돌은 없었다. [보충 실험 검증](results_pool/runner_validation.json) · [실행 코드](runner_pool31.py) · [Paired 분석](paired_pool31.py)\n"
    (ROOT/'PROVENANCE_KO.md').write_text(prov)
    generate()

if __name__=='__main__':
    main()
