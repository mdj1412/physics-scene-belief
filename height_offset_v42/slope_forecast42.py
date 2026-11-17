"""Pre-inspection geometric proxy of the advantage of keeping the contour signal."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OFFSETS=(-1.,-.5,-.25,-.125,0.,.125,.25,.5,1.)
WEIGHTS={'flare':(.8,-.85,.3,.95),'roll':(-.45,.55,.95,-.55)}
KNOTS=(11.,19.,23.,31.,39.)


def displacement(shape,height_mm):
    for l,r,c in zip(KNOTS,KNOTS[1:],WEIGHTS[shape]):
        if l<height_mm<r:
            x=(height_mm-l)/(r-l)
            return .9*c*16*(x*(1-x))**2,.9*c*32*x*(1-x)*(1-2*x)/(r-l)
    return 0.,0.


def main():
    shifts=np.arange(-6.,7.,2.)
    records=[]
    for f in WEIGHTS:
        for h in (19,31):
            y=np.array([displacement(f,h+dy)[0] for dy in shifts])
            # Orthogonality of symmetric signed heights to constant and quadratic terms.
            apparent=float(shifts@y/(shifts@shifts))
            for delta in OFFSETS:
                true=displacement(f,h+delta)[1]
                # Difference in flat vs apparent slope *squared-error*, for a full bulge.
                proxy=true**2-(apparent-true)**2
                records.append(dict(shape=f,height_mm=h,delta_mm=delta,apparent_slope=apparent,
                                    true_slope=true,flat_minus_apparent_slope_mse=proxy,
                                    projection_harm_predicted=bool(proxy>0)))
    (ROOT/'slope_forecast42.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
