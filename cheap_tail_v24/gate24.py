"""Small host-side regression. The online route reads the same raw host input as the comparator."""
from physics24 import CAL,FY,FactorBank,torch,t,np
from policy23 import TailGate
from policy22 import action
from run23 import loss
from common import R,read

BUDGETS=(768,1024,1280,1536,1792,2048)
PAIRS=((768,1536),(1024,1792),(1280,2048))
REFERENCE=2048
class Engine(TailGate):
    def __init__(self,pair,cal):
        super().__init__(pair,0)
        self.inner=FactorBank(pair[0],cal);self.outer=FactorBank(pair[1],cal)
    def choose(self,state,k):
        return action(self.risk(state,f"keep{k}m2048"))

class Features:
    def __init__(self,cal):
        self.fy=FY/(FY@FY)
        quadratic=FY**2-np.mean(FY**2)
        self.curve=.05*quadratic/(quadratic@quadratic)
        self.precision=np.log(CAL[cal][2]/CAL[cal][4])
        self.factual_noise=CAL[cal][0]*1000
    def __call__(self,x):
        # Raw slopes and quadratic observable, two velocities and known calibration.
        return np.array([x[:9]@self.fy,x[:9]@self.curve,x[9],x[10],self.precision,self.factual_noise])

class Gate:
    def __init__(self,model):
        self.mean=np.asarray(model["mean"]);self.scale=np.asarray(model["scale"])
        self.coef=np.asarray(model["coef"]);self.bias=model["bias"]
        self.cut=model["cut"]
    def score(self,v):
        h=np.clip((v-self.mean)/self.scale,-4,4)
        return float(self.bias+h@self.coef[:6]+(h*h)@self.coef[6:12]+
                     (h[:4]*h[4])@self.coef[12:16]+(h[:4]*h[5])@self.coef[16:])
    def high(self,v):return self.score(v)>=self.cut

def learn(x,y):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8)
    h=np.clip((x-mean)/scale,-4,4)
    mat=np.column_stack([h,h*h,h[:,:4]*h[:,4,None],h[:,:4]*h[:,5,None]])
    center=mat.mean(0);mat=mat-center
    beta=np.linalg.solve(mat.T@mat+.06*len(y)*np.eye(mat.shape[1]),mat.T@(y-y.mean()))
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=beta.tolist(),bias=float(y.mean()-center@beta),cut=0)
