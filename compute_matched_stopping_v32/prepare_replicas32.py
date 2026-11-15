"""Generate independent full-duration training replicas from the immutable primary code."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
FILES={'train_labels32.py':'cb28c2c30e9c49c2df03a3928b05e83d297c9953a4e62ed7d3a23967c923a709',
       'training_plan.json':'deab909601ceaac52fb30898cf573a19b80ed580eba71c984b30c06352c7216e',
       'fit32.py':'234cc580a63fac354d4d7527794dda44465eb9eb1df7e8e2aed81d225e68222c',
       'run_training.sh':'54cd531c8e1305c5683ed0ab4ebfea276c7af568295ae6b43749227d15800d98',
       'verify_training32.py':'d27da57fb1b7cb5f268ed036bef401298d071d5a4e7ac7f556a90827ec1cc846'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for name,h in FILES.items():
        assert sha(ROOT/name)==h, name
    for run in (2,3):
        out=ROOT.parent/f'{ROOT.name}_rep{run}'
        out.mkdir(exist_ok=True)
        for nam in ('fit32.py','run_training.sh','verify_training32.py','label_precision32.py'):
            shutil.copyfile(ROOT/nam,out/nam)
        plan=json.loads((ROOT/'training_plan.json').read_text())
        start=372000000+(run-2)*10000000
        plan['arms']['broad']['base_seed']=start;plan['arms']['repeated']['base_seed']=start+3000000
        plan['seeds']=plan['seeds'].replace('350000000 and 351000000',f'{start+6000000} and {start+7000000}')
        (out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        origin=(ROOT/'train_labels32.py').read_text()
        old='350000000+1000000*idx';new=f'{start+6000000}+1000000*idx'
        assert origin.count(old)==1
        (out/'train_labels32.py').write_text(origin.replace(old,new))
        assert (out/'train_labels32.py').read_text().replace(new,old)==origin
        note={'replication':run,'source_unchanged':FILES,'plan_seed_origins':plan['arms'],
              'warmup_seed_origins':[start+6000000,start+7000000],
              'delta':'Only base/warmup RNG origins and corresponding seed documentation differ. Frozen detector quadrature, features, label/selection likelihood, 32+104 probe quota, GPU budget, time matcher and fits unchanged. No target data or results used.',
              'source_view':'Predeclared before primary model fit or target evaluation. Three independently generated source training pairs; targets shared when comparing fitted policies.',
              'verification':{n:sha(out/n) for n in FILES}}
        (out/'REPLICA_PROVENANCE.json').write_text(json.dumps(note,indent=2)+'\n')

if __name__=='__main__':main()
