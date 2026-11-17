"""Build both report editions and check artifact references."""
import os
from pathlib import Path
import re
root=Path(__file__).resolve().parent
parts=("SUMMARY_KO.md","FINDINGS_KO.md","LITERATURE_KO.md","REPRODUCE_KO.md")
body="\n\n".join((root/p).read_text().strip() for p in parts)+"\n"
out=root.parents[2]/"outputs"
def relocate(match):
    dest=match.group(1)
    if dest.startswith(("http://","https://","#")):return match.group(0)
    path=(root/dest).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    return "]("+os.path.relpath(path,out)+")"
edition=re.sub(r'\]\(([^)\s]+)\)',relocate,body)
(root/"REPORT_KO.md").write_text(body)
if root.parent.parent.name=="work" and out.is_dir():
    (out/"physics_scene_belief_normal_variance_v46_report.md").write_text(edition)
