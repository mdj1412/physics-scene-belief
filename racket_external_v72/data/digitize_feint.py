"""Convert manually read Fig.12(a,b) centers to physical values and pair identifiable impacts.

Araki, Sato and Yamazaki, IJTTS 3 (1996), pp.17-49. The readout uses the page
29 image at 260 dpi from pdftoppm, rotated clockwise 90 degrees, full width 3041 px.
In the width-1600 reference, crop box a=(428,122,780,372), b=(958,127,1310,380).
Convert these to native raster coordinates via round(value*3041/1600), crop, resize
to 1056x750. All centers and corners below are pixels in those enlarged crops.
Coordinates were estimated without using any predictive model.
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Top-left, top-right, bottom-left, bottom-right at grid tick intersections.
CORNERS_A = ((30,25),(1028,35),(27,725),(1030,735))
CORNERS_B = ((26,25),(1028,32),(28,719),(1030,727))
# A uses u' in cm/s; B uses n' in revolutions/second.
A = [(27,725),(90,725),(114,728),(118,716),(150,690),(199,666),(234,627),
     (255,597),(342,531),(370,501),(438,465),(501,421),(529,362),(608,348),
     (670,274),(671,326),(749,244)]
B = [(27,602),(89,603),(111,371),(116,383),(147,360),(194,349),(229,374),
     (250,339),(339,339),(367,306),(436,281),(499,260),(527,237),(608,145),
     (671,65),(750,32)]

def scale(point, anchors, maximum, minimum):
    x, y = point
    tl,tr,bl,br=anchors
    a,b=.5,.5
    for _ in range(8):
        left=(1-b)*tl[0]+b*bl[0]; right=(1-b)*tr[0]+b*br[0]
        a=(x-left)/(right-left)
        top=(1-a)*tl[1]+a*tr[1]; bottom=(1-a)*bl[1]+a*br[1]
        b=(y-top)/(bottom-top)
    return 250*a, maximum+b*(minimum-maximum)

def run():
    panels=[[(scale(x,CORNERS_A,1400,0),x) for x in A],
            [(scale(x,CORNERS_B,50,-10),x) for x in B]]
    with (ROOT/'feint_fig12_centers.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['panel','marker_id','x_px','y_px','n_in_rps','ordinate'])
        for label,rows in zip(('12a_u_cmps','12b_spin_rps'),panels):
            for i,((n,v),(x,y)) in enumerate(rows):
                w.writerow((label,i,x,y,round(n,4),round(v,4)))
    cutoff=2.0
    joint=[]
    for i,((na,u),p) in enumerate(panels[0]):
        paired=[(j,nb,w) for j,((nb,w),_) in enumerate(panels[1]) if abs(na-nb)<=cutoff]
        if len(paired)!=1: continue
        j,nb,w=paired[0]
        if sum(abs(aa-nb)<=cutoff for ((aa,_),_) in panels[0])!=1:continue
        n=(na+nb)/2
        # Near-zero centers lie on the zero axes of the published charts.
        if abs(n)<.5:n=0.
        if abs(u)<3:u=0.
        if abs(w)<.25:w=0.
        joint.append([f'feint_{i:02d}',i,j,round(na,4),round(nb,4),round(n,4),round(u/100,6),round(w,5),8.3333333333])
    assert len(joint)==13, joint
    with (ROOT/'feint_fig12_paired.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','marker_a','marker_b','n_a_rps','n_b_rps','n_in_rps','u_out_mps','n_out_rps','v_in_normal_nominal_mps']);writer.writerows(joint)
    # The authors printed zero-intercept slopes for both panels. This is a scale diagnostic only.
    for rows in panels:
        slope=sum(n*v for ((n,v),_) in rows)/sum(n*n for ((n,_),_) in rows)
        print(f'{len(rows)} resolved visible marks; all-visible origin-fit slope {slope:.4f}')
if __name__=='__main__':run()
