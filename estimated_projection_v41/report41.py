"""Assemble the standalone Korean report and update links for the user-facing copy."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
PARTS=('SUMMARY_KO.md','methods41.md','TABLES_KO.md','DIRECTION_AUDIT_KO.md',
       'MECHANISM_KO.md','SELECTOR_AUDIT_KO.md','INTERPRETATION_KO.md','limits41.md','literature41.md','VERIFICATION_KO.md')
def main():
    sections=[]
    for name in PARTS:
        body=(ROOT/name).read_text().strip()
        if name in ('DIRECTION_AUDIT_KO.md','SELECTOR_AUDIT_KO.md') and body.startswith('# '):
            body='#'+body
        if name=='DIRECTION_AUDIT_KO.md':
            body+='\n\n![두 접촉 높이에서 12개 panel의 estimated direction과 oracle 비교](geometry_directions41.png)\n\n그림에서는 방향의 부호를 oracle에 맞춰 표시했다. 실제 projection은 부호에 무관하다.'
        sections.append(body)
    text='\n\n'.join(sections)+'\n'
    link=re.compile(r'\]\(([^)\s]+)\)')
    def adapt(match,prefix):
        url=match.group(1)
        if url.startswith(('https://','http://','#')): return match.group()
        if not (ROOT/url).is_file(): raise FileNotFoundError(url)
        return ']('+prefix+url+')'
    canonical=link.sub(lambda m:adapt(m,''),text)
    (ROOT/'REPORT_KO.md').write_text(canonical)
    workspace=ROOT.parents[2]
    if (workspace/'outputs').is_dir():
        (workspace/'outputs/physics_scene_belief_estimated_projection_v41_report.md').write_text(
            link.sub(lambda m:adapt(m,'../work/physics-scene-belief/estimated_projection_v41/'),text))

if __name__=='__main__': main()
