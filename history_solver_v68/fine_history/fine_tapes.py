"""Recompute all measured velocities at the committed 17-site, 48-step resolution."""
import json
import os
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
import tapes68
import label_status68
import mechanics68
def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    stats=label_status68.setup()
    def simulate(*args,**kwargs):
        return mechanics68.simulate(*args,**kwargs,fine=True)
    tapes68.H=H; tapes68.simulate=simulate
    tapes68.locked_run.configure();tapes68.w.torch.set_num_threads(1)
    for arg in sys.argv[1:]:
        tapes68.build(int(arg),'compliant')
        (H/('tape_ready_'+str(int(arg)))).write_text('complete\n')
    (H/('history_status_'+sys.argv[1]+'.json')).write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':main()
