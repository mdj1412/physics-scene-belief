"""Assemble and link-check local and server editions of the report."""
import os
from pathlib import Path
import re

root=Path(__file__).resolve().parent
parts=("SUMMARY_KO.md","AREA_KO.md","COST_KO.md","EXTERNAL_RESULTS_KO.md","READING_KO.md","REPRODUCE_KO.md")
report="\n\n".join((root/p).read_text().strip() for p in parts)+"\n"
output=root.parents[2]/"outputs"
def relocate(match):
    target=match.group(1)
    if target.startswith(("http://","https://","#")):
        return match.group(0)
    path=(root/target).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return "]("+os.path.relpath(path,output)+")"
other=re.sub(r'\]\(([^)\s]+)\)',relocate,report)
(root/"REPORT_KO.md").write_text(report)
if root.parent.parent.name=="work" and output.is_dir():
    (output/"physics_scene_belief_image_pressure_v45_report.md").write_text(other)
