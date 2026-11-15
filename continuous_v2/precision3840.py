"""Generate an independent higher-temporal-resolution dataset; no overwrite."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import physics
import prepare

physics.DT=1/3840
prepare.DT=1/3840

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['table','data','merge'])
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    out=Path('precision3840/data');out.mkdir(parents=True,exist_ok=True)
    if args.stage=='table':prepare.table(args.shard,out)
    elif args.stage=='data':prepare.dataset(args.shard,out)
    else:prepare.merge(out)
    (out/f'precision_{args.stage}_{args.shard}.json').write_text(json.dumps(dict(dt=physics.DT,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_physics_sha256=hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest()),indent=2))
