"""Independent standard-library check of the frozen source-only sign head."""
import array
import ast
import json
from pathlib import Path
import struct
import zipfile

ROOT=Path(__file__).resolve().parent/'result'
def double_rows(archive,key):
    with archive.open(key+'.npy') as f:
        marker=f.read(8)
        assert marker==b'\x93NUMPY\x01\x00'
        length,=struct.unpack('<H',f.read(2))
        hdr=ast.literal_eval(f.read(length).decode())
        assert hdr['descr']=='<f8' and hdr['shape'][-1]==3 and not hdr['fortran_order']
        flat=array.array('d');flat.frombytes(f.read())
        return flat
def sign(v,i):
    return v[3*i]>v[3*i+2]
def main():
    cut=json.loads((ROOT/'run.json').read_text())['fast_cut_mps']
    output={}
    for split in ['official','secondary']:
        with zipfile.ZipFile(ROOT/split/'predictions.npz') as z:
            x,y=double_rows(z,'incident'),double_rows(z,'target')
            n=len(y)//3
            fixed={}
            for k in [4,8]:
                p=double_rows(z,'source_only_k'+str(k))
                fixed[str(k)]=sum(sign(p,j)==sign(x,j%n) for j in range(len(p)//3))
            stats={}
            for g in ['fast','other','all']:
                ids=[i for i in range(n) if g=='all' or (abs(x[3*i+1])>=cut)==(g=='fast')]
                keep=[i for i in ids if abs(y[3*i]-y[3*i+2])>.20]
                stats[g]=dict(n=len(ids),resolved=len(keep),same=sum(sign(x,i)==sign(y,i) for i in keep),
                              reversed=sum(sign(x,i)!=sign(y,i) for i in keep))
            output[split]=dict(source_only_vs_negative_incident_sign_disagreements=fixed,groups=stats)
    (ROOT/'reversal_audit.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
