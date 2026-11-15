"""A pointwise finite-panel variance bound on expected gains under balanced targeting."""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parent


def main():
    reports=[]
    for panel in ['results','results_pool']:
        stat=json.loads((ROOT/panel/'crossed_stats31.json').read_text())
        res=stat['pooled']['between_corrected']
        row=[]
        for value,bounds in zip(res['estimate'],res['pointwise95']):
            row.append(dict(signal=value,signal_ci=bounds,upper_gain=.5*math.sqrt(7/8*max(0,bounds[1]))))
        reports.append(row)
    output=dict(fresh=reports[0],pool=reports[1],description='Upper envelope for ideal half selection versus uniform half, averaging blocks equally with the same candidate panel. Uses pointwise upper endpoints of two-sided percentile cluster bootstrap for unbiased between-prefix expected gain variance. Approximate exploratory bound, no test of a broader population or sequence of adaptive decisions.')
    (ROOT/'results/finite_gain_ceiling31.json').write_text(json.dumps(output,indent=2,allow_nan=False))
if __name__=='__main__':
    main()
