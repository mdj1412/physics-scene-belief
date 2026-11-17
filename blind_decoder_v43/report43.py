"""Assemble the result narrative and rewrite working links for the output copy."""
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PARTS = ('SUMMARY_KO.md','METHODS_KO.md','TABLES_KO.md','FINDINGS_KO.md','literature43.md','RECORD_KO.md')


def build():
    report = '\n\n'.join((ROOT/p).read_text().strip() for p in PARTS)+'\n'
    (ROOT/'REPORT_KO.md').write_text(report)
    output = ROOT.parents[2]/'outputs'
    if not output.is_dir():
        return
    def link(match):
        url = match.group(1)
        if url.startswith(('https://','http://','#')):
            return match.group(0)
        dest= (ROOT/url).resolve()
        if not dest.is_file():
            raise FileNotFoundError(dest)
        return ']('+os.path.relpath(dest,output)+')'
    (output/'physics_scene_belief_blind_decoder_v43_report.md').write_text(
        re.sub(r'\]\(([^)\s]+)\)',link,report))


if __name__=='__main__':
    build()
