"""Reuse the frozen evaluation loop with a finer numerical reference only.

The method list and dense-model factory are overridden in this process; no
main source, data, policy threshold, or existing result is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import run as experiment

original_factory=experiment.BeliefModel


def refined_factory(cfg,forward,kind):
    return original_factory(cfg,forward,'fine' if kind=='dense' else kind)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    args=parser.parse_args()
    experiment.BeliefModel=refined_factory
    experiment.METHODS=[('fine_evsi','dense','evsi')]
    out=Path('results/refined')
    with torch.inference_mode():experiment.run(args.shard,out)
    path=out/f'runtime_{args.shard}.json'
    meta=json.loads(path.read_text())
    meta['code_hashes']['refine.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta['factory_override']='dense -> grid 257 x 65; all evaluation inputs/seeds identical'
    path.write_text(json.dumps(meta,indent=2))
