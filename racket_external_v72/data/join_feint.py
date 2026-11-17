"""Match Fig.11 and Fig.12 observations within 2 rps across all three panels."""
import csv
import math
from pathlib import Path

root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open() as f:return list(csv.DictReader(f))

left=read('feint_fig12_paired.csv')
right=[r for r in read('araki_fig11.csv') if r['pairing_status']=='unique']
fields=['id','n_in_rps','u_out_mps','n_out_rps','e_normal','normal_v_out_nominal_mps',
        'v_in_normal_nominal_mps','spin_axis_rps_span','n11_rps','n12a_rps','n12b_rps',
        'near_zero_release','marker_a','marker_b','marker_square']
used=set()
with (root/'feint_ox_digitized.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in left:
        ns=[float(r['n_a_rps']),float(r['n_b_rps'])]
        matched=[s for s in right if max(ns+[float(s['n_in_rps_fig11'])])-min(ns+[float(s['n_in_rps_fig11'])])<=2.]
        assert len(matched)==1
        s=matched[0];assert s['square_index'] not in used
        used.add(s['square_index']);alln=ns+[float(s['n_in_rps_fig11'])]
        # Incoming spin is averaged over the two axes of the directly measured outcome panels.
        # Source radius .019 m and thin-shell dynamics. A sign count requires release above scan resolution.
        sign_loss=abs(float(r['u_out_mps'])-0.019*2*math.pi*float(r['n_out_rps']))
        w.writerow(dict(id=r['id'],n_in_rps=r['n_in_rps'],u_out_mps=r['u_out_mps'],n_out_rps=r['n_out_rps'],
                        e_normal=s['normal_restitution'],normal_v_out_nominal_mps=s['normal_velocity_nominal_m_s'],
                        v_in_normal_nominal_mps=r['v_in_normal_nominal_mps'],spin_axis_rps_span=round(max(alln)-min(alln),4),
                        n11_rps=s['n_in_rps_fig11'],n12a_rps=r['n_a_rps'],n12b_rps=r['n_b_rps'],
                        near_zero_release=int(sign_loss<=.18),marker_a=r['marker_a'],marker_b=r['marker_b'],marker_square=s['square_index']))
assert len(used)==len(left)==13
