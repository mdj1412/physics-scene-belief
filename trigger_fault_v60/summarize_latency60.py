"""Validate all saved timing rows and produce the two-population table."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    data=json.loads((ROOT/'timing60.json').read_text())
    q=np.array(data['p90_ms']); assert len(q)==6
    ind=data['independent_uniform_delay']; iq=np.array(ind['p90_ms'])
    times=[];assigned=[];errors=[]
    for i in range(24):
        with np.load(ROOT/'timing60'/f'panel_{i:02d}.npz') as d:
            times.append(d['ms']);assigned.append(d['labels']);errors.append(d['errors'])
    ms=np.asarray(times);cells=np.asarray(assigned)
    np.testing.assert_allclose(np.percentile(ms,90,axis=(0,1)),q,atol=1e-12)
    np.testing.assert_allclose(np.percentile(ms[cells[:,:,1]==1],90,axis=0),iq,atol=1e-12)
    assert not np.any(np.asarray(errors)>2e-9)
    names=('Corrected loss-weighted reset','Corrected loss-weighted sham','항상 v54 gate',
           'Uncorrected loss-weighted reset','Uncorrected loss-weighted sham','Corrected MAP reset')
    txt=['| 정책 | Independent fault·균등 delay GPU p90 (ms/512 objects) | 세 fault 관계·균등 delay GPU p90 (ms/512 objects) |',
         '|---|---:|---:|']
    for name,x,y in zip(names,iq,q):
        txt.append(f'| {name} | {x:.3f} | {y:.3f} |')
    txt+=['','| 같은 모집단에서 corrected reset − 대조 | Independent fault 차이 [동시 95% CI], ms | 전체 차이 [동시 95% CI], ms |','|---|---:|---:|']
    for j,name in enumerate(['Corrected loss-weighted sham','항상 v54 gate']):
        ci,cp=ind['simultaneous95'][j],data['primary_two_simultaneous95'][j]
        txt.append(f'| {name} | {iq[0]-iq[j+1]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {q[0]-q[j+1]:+.3f} [{cp[0]:+.3f}, {cp[1]:+.3f}] |')
    txt+=['',f"데이터는 independent 모집단 정책당 {ind['blocks']}회, 전체 {data['measured_per_arm']}회다. 24개 catalog를 함께 16,000회 복원추출해 각 모집단에서 두 primary contrast에 대한 동시 95% CI를 계산했다. 실행 순서 및 직전 정책의 각 조합은 전체에서 각각 12회 나타났다. 관측된 최대 forecast replay 오차는 {max(data['max_replay_error']):g}였다. Cell별 8개 관측의 p90은 작은 표본의 기술통계다. [Raw pixel-to-terminal 결과](timing60.json) · [원자료 예시](timing60/panel_00.npz)",
          '', '별도의 CPU microbenchmark에서는 동일 32개 영상의 marker 검출·pixel tracking·policy routing만 따로 실행했다(정책당 432회, compute node56). p90은 corrected reset 0.3474 ms, corrected sham 0.3462 ms, uncorrected reset 0.3451 ms, 항상 gate 0.0012 ms였다. 두 reset 조건의 preprocessing 차이는 0.00235 ms였고, 전체 GPU p90의 차이는 이 값에서 단순히 계산할 수 없다. [Pixel-to-routing 비용](reader_cost60.json) · [측정 코드](reader_cost60.py)']
    sec='\n'.join(txt)
    (ROOT/'timing_table60.md').write_text(sec+'\n')
    note=ROOT/'GPU_WORLD_KO.md'; text=note.read_text()
    if '<!--TIMING-->' in text:note.write_text(text.replace('<!--TIMING-->',sec))
    print(sec)
if __name__=='__main__':main()
