from flexible import Expanded
from acquisition import pair_select,nested_scores,nested_plan
METHODS=['compact_native','compact_pair','compact_probe','qmc16_native','qmc16_pair','qmc16_probe','qmc16_nested128','qmc64_native','qmc64_pair','qmc64_probe']
class Engine:
    def __init__(self,name):
        self.name=name;self.mode=name.split('_')[1];self.outer=None;self.plan=None
        if name.startswith('compact'):self.method=Expanded()
        else:
            from bank import Bank
            self.method=Bank(True,65536 if name.startswith('qmc64') else 16384)
            if self.mode=='nested128':self.outer=Bank(True,16384,20261113);self.plan=nested_plan()
    def start(self,x):
        s=self.method.fit(x)
        if self.mode=='probe':a=3
        elif self.mode=='native':a=int(self.method.choose(s).item())
        elif self.mode=='nested128':a=int(nested_scores(self.method,s,self.outer,self.outer.fit(x),plan=self.plan).argmax(1).item())
        else:a=int(pair_select(self.method,s,x,self.mode)[0].argmax(1).item())
        return s,a
    def finish(self,state,a,observation):return self.method.predict(state if a==0 else self.method.update(state,a,observation))
    def cache_bytes(self):return self.method.cache_bytes()+(self.outer.cache_bytes()+sum(v.numel()*v.element_size() for v in self.plan) if self.outer else 0)
    def state_bytes(self):return self.method.state_bytes()+(self.outer.state_bytes() if self.outer else 0)
