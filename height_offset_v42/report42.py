"""Assemble canonical server report and resolve workspace-copy links for local readers."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARTS=('SUMMARY_KO.md','METHODS_KO.md','TABLES_KO.md','DENSE_KO.md','FINDINGS_KO.md','literature42.md','RECORD_KO.md')
def main():
    body='\n\n'.join((ROOT/n).read_text().strip() for n in PARTS)+'\n'
    def adapt(m,prefix):
        url=m.group(1)
        if url.startswith(('https://','http://','#')):return m.group(0)
        if not (ROOT/url).is_file(): raise FileNotFoundError(ROOT/url)
        return ']('+prefix+url+')'
    pattern=re.compile(r'\]\(([^)\s]+)\)')
    (ROOT/'REPORT_KO.md').write_text(pattern.sub(lambda m:adapt(m,''),body))
    outputs=ROOT.parents[2]/'outputs'
    if outputs.is_dir():
        (outputs/'physics_scene_belief_height_offset_v42_report.md').write_text(pattern.sub(
            lambda m:adapt(m,'../work/physics-scene-belief/height_offset_v42/'),body))
if __name__=='__main__':main()
