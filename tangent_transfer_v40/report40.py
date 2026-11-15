"""Build the project report and a separately linked workspace report."""
from pathlib import Path
import re
root=Path(__file__).resolve().parent
workspace=root.parents[2]
parts=('SUMMARY_KO.md','methods40.md','TABLES_KO.md','mechanism40.md','SUPPLEMENT_KO.md','INTERPRETATION_KO.md','literature40.md','REPRODUCE_KO.md')
sections=[]
for name in parts:
    chunk=(root/name).read_text().strip()
    if name=='REPRODUCE_KO.md' and chunk.startswith('# '):
        chunk='#'+chunk
    sections.append(chunk)
body='\n\n'.join(sections)+'\n'
def adapt(match):
    url=match.group(1)
    if url.startswith(('https://','http://','#')):
        return match.group(0)
    if not (root/url).is_file():
        raise FileNotFoundError(root/url)
    return '](../work/physics-scene-belief/tangent_transfer_v40/'+url+')'
workspace_report=re.sub(r'\]\(([^)\s]+)\)',adapt,body)
(root/'REPORT_KO.md').write_text(body)
if (workspace/'outputs').is_dir():
    (workspace/'outputs/physics_scene_belief_tangent_transfer_v40_report.md').write_text(workspace_report)
