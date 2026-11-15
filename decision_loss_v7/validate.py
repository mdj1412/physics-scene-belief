import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from experiment import make_world,run_group,Model
from engine import tensor,decode
from decision import load,stability

r=Path('results');old=Path('../tempered_bridge_v6')
hashes={}
for name in ['engine.py','bridge.py','belief.py','base_inference.py','results/calibration/pixel_only.json']:
    h=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert h==hashlib.sha256((old/name).read_bytes()).hexdigest();hashes[name]=h
subprocess.run([sys.executable,'validate_gaussian.py'],check=True)
for case in [0,1]:
    if not (r/'worlds'/f'case_{case:03d}'/'complete.json').exists():make_world(case)
run_group([0,1],'acq_smoke',16,256,3,2)
for case in [0,1]:
    folder=r/'worlds'/f'case_{case:03d}';bank=np.load(folder/'bank.npz');gen=np.load(folder/'generator.npz');outer=np.load(folder/'outer.npz')
    assert len(bank['x'])==16384 and len(gen['x'])==8192 and not np.array_equal(bank['x'][:8192],gen['x'])
    theta=decode(tensor(bank['x'][:5,:4]));b=tensor(bank['x'][:5,4])*.006
    native=Model('corner');precise=Model('corner','precise');mu,cv=native.mean_cov(theta,b,'view');pm,pc=precise.mean_cov(theta,b,'view')
    assert torch.equal(mu,pm) and torch.equal(pc,cv*.25**2)
    for action in ['factual','left']:
        a,c=native.mean_cov(theta,b,action);aa,cc=precise.mean_cov(theta,b,action);assert torch.equal(a,aa) and torch.equal(c,cc)
    # Independent NumPy SNIS reference checks likelihood units/axis order.
    for precision in ['native','precise']:
        model=Model('corner',precision);_,cov=model.mean_cov(theta,b,'view');delta=outer[precision+'_view'][:2,None,:]-bank['mu_view'][None,:4096,:]
        ll=-.5*np.einsum('hnd,dk,hnk->hn',delta,np.linalg.inv(cov.cpu().numpy()),delta);w=np.exp(ll-ll.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        expected=np.einsum('hn,nqd->hqd',w,bank['z'][:4096])
        actual=np.load(r/'acq_smoke'/f'case_{case:03d}'/f'{precision}_view_snis4096.npz')['mean'][:2]
        assert np.allclose(expected,actual,rtol=1e-9,atol=1e-10)
        d=load(r/'acq_smoke'/f'case_{case:03d}',precision);assert d['samples'].shape==(16,6,4) and np.isfinite(d['samples']).all()
        assert len(stability(d['samples'],[0,0,0,0],42)['choice'])==6
result=dict(passed=True,source_hashes=hashes,checks=['bounded Gaussian quadrature','independent generator bank','view-only covariance scaling','NumPy SNIS reference','two-world batched smoke','decision array shapes'],
    runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID')))
(r/'validation/smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
