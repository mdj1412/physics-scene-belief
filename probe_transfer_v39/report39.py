"""Build standalone Markdown reports using local relative, verified links."""
from pathlib import Path
import re

root=Path(__file__).resolve().parent
workspace=root.parents[2]
parts=('SUMMARY_KO.md','TABLES_KO.md','ANALYSIS_KO.md','LITERATURE_KO.md','REPRODUCE_KO.md')
body='\n\n'.join((root/name).read_text().strip() for name in parts)+'\n'
(root/'REPORT_KO.md').write_text(body)
def rewrite(match):
    path=match.group(1)
    if path.startswith(('https://','http://','#')):
        return match.group(0)
    if not (root/path).exists():
        raise FileNotFoundError(root/path)
    return '](../work/physics-scene-belief/probe_transfer_v39/'+path+')'
out=re.sub(r'\]\(([^)\s]+)\)',rewrite,body)
(workspace/'outputs/physics_scene_belief_probe_transfer_v39_report.md').write_text(out)
