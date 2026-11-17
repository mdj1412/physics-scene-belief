"""Fetch declared public OPARA entries (ranged transfer with archive CRC verification)."""
import argparse
import io
import json
import zipfile
from pathlib import Path

from urllib.request import Request, urlopen

BASE = 'https://opara.zih.tu-dresden.de/server/api/core/bitstreams/5b0c368c-3ba0-4fb1-ab90-7fba1a5ecc54'
DATA = Path('/scratch2/mdj1412/datasets/physics_scene_belief_v64/opara655')
ROOT = Path(__file__).resolve().parent

class PublicArchive(io.RawIOBase):
    def __init__(self, size):
        self.size = size
        self.pos = 0
        self.cache = {}
    def seekable(self):
        return True
    def seek(self, amount, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + amount
        return self.pos
    def tell(self):
        return self.pos
    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b''
        block = 131072
        first = self.pos // block
        last = (end - 1) // block
        off=self.pos % block
        if not all(i in self.cache for i in range(first,last+1)):
            start=first * block
            stop=min((last+1)*block,self.size)-1
            with urlopen(Request(BASE+'/content', headers={'Range':f'bytes={start}-{stop}'}),timeout=45) as r:
                assert r.status == 206
                payload=r.read()
                assert len(payload) == stop-start+1
            for i in range(first,last+1):
                self.cache[i]=payload[(i-first)*block:(i-first+1)*block]
        out=b''.join(self.cache[i] for i in range(first,last+1))[off:off+end-self.pos]
        self.pos=end
        return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('names',nargs='*',help='Archive paths or file prefix ending in /')
    a=p.parse_args()
    DATA.mkdir(exist_ok=True,parents=True)
    with urlopen(BASE,timeout=40) as r:
        meta=json.load(r)
    (DATA/'repository_record.json').write_text(json.dumps(meta,indent=2))
    with zipfile.ZipFile(PublicArchive(meta['sizeBytes'])) as z:
        (ROOT/'opara64_inventory.json').write_text(json.dumps([dict(name=m.filename,bytes=m.file_size,compressed=m.compress_size,crc=m.CRC) for m in z.infolist()],indent=2))
        for m in z.infolist():
            if m.is_dir() or not any(m.filename==t or (t.endswith('/') and m.filename.startswith(t)) for t in a.names):
                continue
            target=DATA/m.filename; assert DATA.resolve() in target.resolve().parents
            target.parent.mkdir(exist_ok=True,parents=True)
            if not target.exists():
                tmp=target.with_suffix(target.suffix+'.part')
                tmp.write_bytes(z.read(m))
                tmp.replace(target)
            print(m.filename, m.file_size,flush=True)
if __name__=='__main__': main()
