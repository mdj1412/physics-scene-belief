"""Rerun full v63 end-to-end benchmark on v64 two-light simulated input."""
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LEGACY=ROOT.parent/'reflective_tracking_v63'
sys.path.insert(0,str(LEGACY))
import bench63 as bench
assert os.getenv('SLURM_JOB_ID')
if __name__=='__main__':
    dest=ROOT/'sim_timing_work64'
    dest.mkdir(exist_ok=True)
    for field,source in (('pilots',ROOT),('fixtures',ROOT),('forecasts',LEGACY)):
        link=dest/field
        if not link.exists():link.symlink_to(source/field,target_is_directory=True)
        assert link.resolve()==source/field
    bench.ROOT=dest
    bench.main()
    data=json.loads((dest/'timing63.json').read_text())
    data['input']='v64 independent simulated lamp captures only'
    data['cells']=[[-30,8],[30,8],[-30,26],[30,26]]
    data['common_decoder']='frozen v63, with v63 stored forecast used for replay validation only'
    (ROOT/'sim_latency64.json').write_text(json.dumps(data,indent=2)+'\n')
