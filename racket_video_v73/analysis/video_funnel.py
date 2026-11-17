"""Enumerate the prior video candidates without conditioning the coverage denominator on tracking success."""
from collections import defaultdict
from pathlib import Path
import csv
import json
VERSION = Path(__file__).resolve().parents[1]
PRESTON = VERSION.parent/'racket_external_v72/audit/preston'
def rows(path):
    with path.open() as f: return list(csv.DictReader(f))
def main():
    cards = rows(PRESTON/'validation/title_inventory.csv')
    intervals = {r['id']:r for r in rows(PRESTON/'sign_sensitivity.csv')}
    tracks = {r['id'] for r in rows(PRESTON/'anti_eligible.csv')}
    assert len(cards)==24 and tracks==set(intervals)
    counts=defaultdict(lambda:dict(attempted=0,kinematics=0,interval=0,margin010=0))
    for row in cards:
        r=intervals.get(row['image_ref'])
        for key in ('all',row['speed_card']):
            c=counts[key]
            c['attempted']+=1
            if r is None: continue
            c['kinematics']+=1
            for margin,field in [(0,'interval'),(.1,'margin010')]:
                c[field]+= (float(r['min_out'])>margin or float(r['max_out'])< -margin)
    (VERSION/'analysis/video_funnel.json').write_text(json.dumps(dict(counts),indent=2)+'\n')
if __name__=='__main__':main()
