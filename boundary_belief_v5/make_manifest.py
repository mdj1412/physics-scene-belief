import hashlib,json,platform,time,zipfile
from pathlib import Path

root=Path('.')
scopes=['main','pixel_only','conditional_view','conditional_refined','conditional_convergence','conditional_failure_audit','audit','calibration','validation','real_boundary','analysis']
files=list(root.glob('*.py'))+list(root.glob('*.md'))+list(Path('reports').glob('*.md'))
for name in scopes:files.extend(p for p in (Path('results')/name).rglob('*') if p.is_file())
rows=[]
for p in sorted(set(files)):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest()))
counts={q:len(list((Path('results')/q).glob('case_*.json'))) for q in ['main','pixel_only','conditional_view','conditional_refined','audit']}
assert list(counts.values())==[72,72,72,72,3],counts
assert len(list(Path('results/conditional_failure_audit').glob('case_*.json')))==2
manifest=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=platform.node(),
    remote_root='/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/boundary_belief_v5',
    counts=counts,files=rows,excluded_invalid_archives=['results/main_instantaneous_invalid','results/audit_instantaneous_invalid'],
    dataset_manifest='../contact_model_v4/results/real_data_audit/download_manifest.json')
Path('server_run_manifest.json').write_text(json.dumps(manifest,indent=2))
# Downloadable light package: all case scores, source, calibration and report figures.
# Full RGB, factual banks and per-frame real SDF arrays remain on the server.
with zipfile.ZipFile('review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.write('server_run_manifest.json')
    for p in sorted(set(files)):
        if 'main/images' in str(p) or p.suffix=='.npz' or p.name=='analysis_stdout.json':continue
        z.write(p)
print(json.dumps(dict(counts=counts,manifest_files=len(rows),bundle_bytes=Path('review_bundle.zip').stat().st_size)))
