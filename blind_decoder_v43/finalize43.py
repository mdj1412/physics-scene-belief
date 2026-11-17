"""Run CPU-only integrity and scoring after both target shards, then append timing."""
import json
import time
from pathlib import Path
import audit43
import residual43
import stats43

ROOT=Path(__file__).resolve().parent


def wait_for(name):
    while not (ROOT/'results'/name).exists():
        time.sleep(5)


def dump(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n')


def main():
    wait_for('complete_even.json')
    wait_for('complete_odd.json')
    results=stats43.analyze()
    dump('pred_stats43.json',results)
    print('held-out scoring verified',flush=True)
    audit43.run()
    residual43.main()
    print('blind replay checks finished',flush=True)
    wait_for('complete_timing.json')
    results['timing']=stats43.timing()
    dump('stats43.json',results)
    print('timing analysis finished',flush=True)


if __name__=='__main__':
    main()
