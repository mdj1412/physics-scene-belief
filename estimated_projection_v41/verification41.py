"""Render the completed execution, independent replay, and measured setup costs."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def main():
    plan=read('results/completion.json')
    audit=read('audit41.json')
    physical=read('physics_audit41.json')
    inverse=read('decoder_audit41.json')
    direction=read('direction_audit41.json')
    assert audit['blocks']==physical['complete_blocks']==plan['blocks']==576
    assert audit['run_fingerprint']==plan['fingerprint'] and direction['stored_pairs_and_directions_verified']
    fits=[]; setups=[]
    for p in range(12):
        ids=(('flare','roll')[p%2],('h19','h31')[(p//2)%2],('base','hollow','bulge')[(p//4)%3],('saturating','crossover')[(p//3)%2])
        path=ROOT/'results'/'_'.join(ids)/f'catalog_{p:02d}'/f'block_{p%2:02d}.npz'
        with np.load(path) as a:
            for key,v in [('fit_ms',fits),('setup_ms',setups)]:
                assert a[key].shape==(3,3,3)
                v.extend(a[key][:,2,:].ravel().tolist())
    assert len(fits)==108
    aerr=audit['max_absolute_error']
    perr=physical['max_absolute_errors']
    out=['## 실행과 독립 검산','',
       f"서버 `datacenter3`의 Slurm allocation {plan['job']}에서 시작 직전 compute process 0개, memory 0 MiB인 {plan['gpu']} 한 장을 사용했다. 주 실험 576개 block의 dependency fingerprint와 파일별 SHA-256을 검산했다. GPU 환경은 PyTorch {plan['torch']}, CUDA {plan['cuda']}였다. 세 agent가 세계 생성·물리 검산, selector 학습·선택 검산, GPU 실행·결과 검산을 분담했다. [서버 기록](results/completion.json) · [재현 절차](REPRODUCE_KO.md)",'',
       f"Calibration과 배포에 독립 catalog를 사용했다. 12×2×8×128={physical['raw_deployment_draws']:,}개 deployment draw에 실제로 서로 다른 prototype은 {physical['distinct_deployment_prototypes']:,}개였다. 실제 calibration 접촉에 쓰인 서로 다른 prototype은 {physical['distinct_paid_prototypes']:,}개다. 12개 독립 panel 안에서 contact law·형상·높이·시작 방향·noise의 짝 구조를 유지했다. [물리 검산](physics_audit41.json)",'',
       f"별도의 NumPy collision 식으로 {physical['archived_stick_labels']:,}개 저장 label을 재구성했다. 불일치는 {perr['archived_stick_label']:.0f}이었다. {audit['reconstructed_expected_losses']:,}개 환경별 loss 재계산의 최대 절대 오차는 {aerr['recomputed_Brier']:.2e}였고, 동일 물체의 세 contour에서 oracle 확률의 최대 차이는 {aerr['oracle_contour_invariance']:.2e}였다. 16,384개 가설의 단일 bank를 점검했을 때, 기존 25 mm event를 그대로 사용하면 실제 19 mm·31 mm에서 각각 {inverse['candidate_events_changed_vs_25mm']['0.019']}개·{inverse['candidate_events_changed_vs_25mm']['0.031']}개 가설이 다른 event를 예측했다. 새 높이의 likelihood는 직접 계산한 squared distance와 {inverse['max_absolute_error']['direct_squared_distance_prediction']:.2e} 이내로 일치했다. [Brier 검산](audit41.json) · [Decoder 검산](decoder_audit41.json)",'',
       f"이미 메모리에 있는 64개 profile에서 estimated direction을 다시 구하는 host 구간은 median {np.median(fits):.3f} ms, p90 {np.quantile(fits,.9):.3f} ms였다. Estimated template과 event를 새로 만드는 전체 setup의 p90은 {np.quantile(setups,.9):.2f} ms였다. 두 값은 12개 panel × 3회 × 3개 정책에 걸친 108회 setup에서 계산했다. 실제 64개 profile의 획득·대응·형상 교체를 포함한 총시간은 별도로 측정해야 한다.",'',
       '서버 작업 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/estimated_projection_v41`다. 주 결과는 [통계 JSON](stats41.json), [GPU runner](runner41.py), [label-free 추정](world41.py), [frozen selector](selector41.py)에서 다시 확인할 수 있다.']
    (ROOT/'VERIFICATION_KO.md').write_text('\n'.join(out)+'\n')
if __name__=='__main__': main()
