"""Reanalyse every fitted coefficient in the two S1 tables, not the simulated contact labels."""
import sys, re, hashlib,json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
file=Path(sys.argv[1])
assert hashlib.md5(file.read_bytes()).hexdigest()=='92fa510e1a75c46ab09372e8000ff8e3'
pages=PdfReader(file).pages
rows=[]
for page,regimes,variants in ((0,['stiff on soft','soft on soft'],['flat','52','82','201']),
                  (1,['52','82','201','flat'],['high','medium','low'])):
    triples=[line for line in pages[page].extract_text(extraction_mode='layout').splitlines()
             if any(f in line for f in ('µalong','µagainst','∆µ')) and re.search(r'\d,\s*\d{3}',line)]
    def number(line):
        return np.array([float(v.replace(' ','').replace(',','.').replace('−','-')) for v in re.findall(r'−?\d,\s*\d{3}',line)])
    assert len(triples)==3*len(regimes)
    for i,regime in enumerate(regimes):
        along,against,pub=map(number,triples[i*3:i*3+3])
        err=abs(against-along-pub)
        assert len(err)==len(variants) and err.max()<.00101
        for j,tag in enumerate(variants):
            rows.append(dict(table=page+1,regime=regime,condition=tag,along=along[j],against=against[j],
                             delta_published=pub[j],signed_amplitude=(against[j]-along[j])/(against[j]+along[j])))
a=np.array([v['signed_amplitude'] for v in rows]);d=np.array([v['delta_published'] for v in rows])
out=dict(reference='https://doi.org/10.6084/m9.figshare.c.3950773.v2',download='https://ndownloader.figshare.com/files/10034878',
         md5=hashlib.md5(file.read_bytes()).hexdigest(),n=len(rows),positive=int((d>0).sum()),negative=int((d<0).sum()),
         exceeding_2_reported_se=int((abs(d)>.1).sum()),range_amplitude=[a.min(),a.max()],range_delta=[d.min(),d.max()],rows=rows)
(ROOT/'physical33.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:out[k] for k in ('n','positive','negative','exceeding_2_reported_se','range_amplitude','range_delta')})
