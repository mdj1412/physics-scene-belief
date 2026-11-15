"""Build both server-relative and local, standalone reports with valid asset paths."""
from pathlib import Path
import re

root=Path(__file__).resolve().parent
workspace=root.parents[2]
parts=['RESULTS_KO.md','LITERATURE_KO.md','REPRODUCE_KO.md']
body='\n\n'.join((root/p).read_text().strip() for p in parts)+'\n'
(root/'REPORT_KO.md').write_text(body)
prefix='../work/physics-scene-belief/probe_value_v38/'

def local_target(m):
    target=m.group(1)
    if target.startswith(('https://','http://','#')):
        return m.group(0)
    assert (root/target).is_file(),target
    return ']('+prefix+target+')'

standalone=re.sub(r'\]\(([^)\s]+)\)',local_target,body)
(workspace/'outputs/physics_scene_belief_probe_value_v38_report.md').write_text(standalone)
