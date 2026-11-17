"""Prospective blocking plan. Rows are planned sessions, never acquisition records."""
from pathlib import Path
import csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(650113)
    surfaces=[f'{finish}-{j+1:02}' for finish in ('P','B','S') for j in range(4)]
    # One independently reserved audit session per coupon; three per day×board block.
    sham_blocks=rng.permutation(np.tile(np.arange(4),3))
    sham_lamps={s:v for j in range(3) for s,v in zip(surfaces[4*j:4*j+4],rng.permutation(['AA','AA','BB','BB']))}
    rows=[]
    for block in range(4):
        cells=[(s,'AB') for s in surfaces]+[(s,sham_lamps[s]) for s,k in zip(surfaces,sham_blocks) if k==block]
        for order,j in enumerate(rng.permutation(len(cells))):
            surface,mode=cells[j]
            orders=rng.permutation(['12','21']*4)
            resets=rng.integers(0,2,4)
            objs=rng.permutation(np.arange(1,9))
            for slot in range(8):
                rows.append(dict(session_id=f'V65-{block+1}-{order+1:02}',
                    day=1+block//2,board='Y' if block in (1,2) else 'X',order_in_block=order+1,
                    planned_surface=surface,mode=mode,slot=slot+1,
                    object_reservation=f'C-{block+1}-{order+1:02}-{objs[slot]}',
                    heldout_reservation=f'Q-{block+1}-{order+1:02}',
                    head_pulse_order=orders[slot],
                    electronic_reset_before_object=('' if slot%2==0 else str(resets[slot//2]))))
    sessions=[r for r in rows if r['slot']==1]
    assert len(sessions)==60 and len(set(r['session_id'] for r in sessions))==60
    assert sum(r['mode']=='AB' for r in sessions)==48
    assert len(set(r['object_reservation'] for r in rows))==480
    assert all(sum(r['planned_surface']==s and r['mode']=='AB' for r in sessions)==4 for s in surfaces)
    with (ROOT/'planned_run_sheet65.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
