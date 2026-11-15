"""Online policy: selected observation only; no access to targets or other actions."""
import numpy as np
from compact import Compact
from flexible import Expanded
from detection import pre_score,predictive_score,base_predict
from common import R,read

GATES={'primary_gate':'primary','expand_only':'native','probe_gate':'probe'}
METHODS=['base_native','base_probe','select_gate','expand_only','primary_gate','probe_gate','expanded_native','expanded_probe','qmc16_native','qmc16_probe','expanded_qmc16_native','expanded_qmc16_probe','expanded_qmc64_native','expanded_qmc64_probe']

class Engine:
    def __init__(self,name):
        self.name=name;self.thresholds=read(R/'thresholds.json')
        if 'qmc' in name:
            from bank import Bank
            self.method=Bank(name.startswith('expanded'),65536 if '64' in name else 16384)
        elif name.startswith('expanded_'):self.method=Expanded()
        else:self.method=Compact(5,4)
        self.expansion=Expanded() if name in GATES else None
    def start(self,x):
        s=self.method.fit(x);pre=False
        if self.name in GATES or self.name=='select_gate':pre=bool(pre_score(self.method,x).item()>self.thresholds['pre'])
        if self.name.endswith('_probe') or self.name=='probe_gate':a=3
        elif pre and self.name in ['primary_gate','select_gate']:a=3
        else:a=int(self.method.choose(s).item())
        return dict(x=x,state=s,action=a,pre=pre)
    def finish(self,ctx,observation):
        a=ctx['action'];s=ctx['state'];flag=False
        if self.name in GATES:
            flag=ctx['pre']
            if not flag and a:
                score=float(predictive_score(self.method,s,a,observation).item())
                flag=score>self.thresholds['post'][GATES[self.name]]
        if flag:
            method=self.expansion;s=method.fit(ctx['x']);s=method.update(s,a,observation) if a else s;p=method.predict(s)
        else:
            method=self.method;s=method.update(s,a,observation) if a else s
            p=base_predict(method,s) if isinstance(method,Compact) and not isinstance(method,Expanded) else method.predict(s)
        return p,flag
    def cache_bytes(self):return self.method.cache_bytes()+(self.expansion.cache_bytes() if self.expansion else 0)

