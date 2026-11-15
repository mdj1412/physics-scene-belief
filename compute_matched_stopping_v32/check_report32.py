"""Verify all report tables and local links, then publish the workspace Markdown copy."""
import hashlib
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[2]/'outputs/physics_scene_belief_compute_matched_stopping_v32_report.md'

def main():
    original=(ROOT/'REPORT_KO.md').read_text()
    links=[]
    def local(match):
        label,url=match.groups()
        if url.startswith(('https://','http://','#')):return match[0]
        path=(ROOT/url).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        links.append(str(path))
        return label+'('+str(path)+')'
    published=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',local,original)
    count=0
    for value in re.findall(r'(?m)(?:^\|.*\|\n?)+',original):
        counts=[r.count('|') for r in value.splitlines()]
        if min(counts)!=max(counts):
            raise ValueError('Invalid Markdown table: '+value[:160])
        count+=1
    assert count>=10 and len(links)>=20
    OUT.write_text(published)
    print(OUT,count,'tables',len(links),'local links; canonical SHA256',
          hashlib.sha256(original.encode()).hexdigest())
if __name__=='__main__':main()
