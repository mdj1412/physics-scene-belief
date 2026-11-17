"""Assemble the Korean report and validate links in both reading locations."""
import os
import re
from pathlib import Path

root = Path(__file__).resolve().parent
parts = ("SUMMARY_KO.md", "FINDINGS_KO.md", "literature44.md", "RECORD_KO.md")
report = "\n\n".join((root / part).read_text().strip() for part in parts) + "\n"
def validate(match):
    url = match.group(1)
    if url.startswith(("https://","http://","#")):
        return match.group(0)
    dest = (root / url).resolve()
    if not dest.is_file():
        raise FileNotFoundError(dest)
    return "](" + os.path.relpath(dest, output) + ")"
output = root.parents[2] / "outputs"
output_copy = re.sub(r'\]\(([^)\s]+)\)', validate, report)
(root / "REPORT_KO.md").write_text(report)
if output.is_dir() and root.parent.parent.name == "work":
    (output / "physics_scene_belief_contact_patch_v44_report.md").write_text(output_copy)
