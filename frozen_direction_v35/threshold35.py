"""Descriptive query-amplitude sweep, no target fit. Positive common-direction regimes."""
from pathlib import Path
import json
import numpy as np
from transfer_stress35 import CELLS, ARMS, labels, read_case, BASE
def main():
    mult=np.arange(0.,1.501,.025)
    rows=np.empty((4,8,2,2,len(mult)))
    raw=BASE/'results'
    for c,cell in enumerate(CELLS):
        maximum=.21 if cell.endswith('_atan') else .16
        for b in range(8):
            for j,sign in enumerate((1,-1)):
                p0,p1,h,y=read_case(raw,cell,b,sign)
                d=p1-p0
                movement=np.mean(d*d+2*d*p0,axis=(1,2,3))
                d=d.mean(-1)
                for k,m in enumerate(mult):
                    alt=labels(h,cell,sign,maximum*m)
                    rows[c,b,j,:,k]=movement-2*(d*alt).mean((1,2))
    pooled=rows.mean((0,1,2))
    sm=json.loads((BASE/'transfer_stress35.json').read_text())
    for i,sc in enumerate(ARMS):
        for m,name in [(0,'zero'),(.5,'half'),(1,'full')]:
            np.testing.assert_allclose(pooled[i,np.argmin(abs(mult-m))],sm[name][sc]['delta'],rtol=0,atol=1.e-10)
    report={}
    for i,sc in enumerate(ARMS):
        crossed=np.where(np.signbit(pooled[i,1:])!=np.signbit(pooled[i,:-1]))[0]
        report[sc]=dict(crossings=[[float(mult[k]),float(mult[k+1])] for k in crossed],delta=pooled[i].tolist(),
                        direction_violations=int((np.diff(pooled[i])>1e-10).sum()))
    obj={'description':'Descriptive finite target grid; no threshold fitted or deployed.', 'scale':mult.tolist(),'schedule':report}
    (BASE/'threshold35.json').write_text(json.dumps(obj,indent=2)+'\n')
if __name__=='__main__':main()
