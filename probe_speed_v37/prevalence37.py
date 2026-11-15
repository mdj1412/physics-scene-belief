"""Descriptive reweighting of measured regime-specific risks with no change to either policy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    data=json.loads((ROOT/'stats37.json').read_text())
    answer=[]
    for k,speed in enumerate(data['speeds']):
        strong=data['scopes']['strong']['brier_delta'][k]
        low=data['scopes']['weak_null_equal']['brier_delta'][k]
        crossing=float(-low/(strong-low)) if low*strong<0 else None
        answer.append(dict(speed=speed,if_strong=strong,if_weak_null_equal=low,break_even=crossing,
            examples=[{'proportion_strong':v,'gate_minus_fixed':(1.-v)*low+v*strong} for v in (0.,.25,.50,.75,1.)]))
    (ROOT/'prevalence37.json').write_text(json.dumps(answer,indent=2)+'\n')
if __name__=='__main__':main()
