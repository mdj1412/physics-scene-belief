import json
from pathlib import Path
import numpy as np
import torch
from belief import Forward,BeliefModel


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    forward=Forward()
    velocity_errors=[];contact_errors=[];sensor_errors=[];offgrid=[]
    quadrature_errors=[];refinement_errors=[];agreements=[];actions=[];repeat=[]
    for f in range(32):
        cfg=json.loads(Path(f'data/family_{f:03d}.json').read_text())
        d=np.load(f'data/family_{f:03d}.npz')
        vv,cc=forward.outcomes(d['geometry'][:,None]+np.array(cfg['delta'])[None,:],d['restitution'][:,None])
        velocity_errors.extend(abs(vv-d['truth'][...,:2]).ravel().tolist())
        contact_errors.extend((cc!=d['truth'][...,2]).ravel().tolist())
        mu=forward.observation(d['geometry'],d['restitution'])
        sensor_errors.extend(abs(mu[:,:2]-d['observation_mu'][:,:2]).tolist())
        assert cfg['factual_max_rgb_difference']==0
        model=BeliefModel(cfg,forward)
        assert np.min(abs(d['geometry'][:,None]-np.unique(model.g)[None,:]))>1e-12
        assert np.min(abs(d['restitution'][:,None]-np.unique(model.e)[None,:]))>1e-12
        offgrid.append(True)
        qi=torch.arange(len(cfg['delta']),device='cuda')
        prior=model.prior[None,:].expand(len(qi),-1)
        gain=model.gains(prior,qi)
        action=torch.where(gain.max(-1).values>.005,gain.argmax(-1),-1)
        actions.extend(action.cpu().numpy().tolist());repeat.extend(gain[:,2].cpu().numpy().tolist())
        if f<8:
            fine=BeliefModel(cfg,forward,'fine')
            fg=fine.gains(fine.prior[None,:].expand(len(qi),-1),qi)
            fa=torch.where(fg.max(-1).values>.005,fg.argmax(-1),-1)
            agreements.extend((fa==action).cpu().tolist())
            refinement_errors.extend(abs(fg-gain).cpu().numpy().ravel().tolist())
            higher=BeliefModel(cfg,forward,integration=513)
            hg=higher.gains(higher.prior[None,:].expand(len(qi),-1),qi)
            quadrature_errors.extend(abs(hg-gain).cpu().numpy().ravel().tolist())
        del model
    assert set(actions)>={-1,0,1},'Within-type observation diversity failed'
    assert max(repeat)<1e-8,'Repeat gained hidden information'
    assert max(quadrature_errors)<1e-7,'Observation integration has not converged'
    # Independent particle support and deterministic seeds are checked by constructor.
    result=dict(development_families=32,all_truth_off_inference_grid=all(offgrid),
        physics_lookup_velocity_rmse=float(np.sqrt(np.mean(np.square(velocity_errors)))),
        physics_lookup_velocity_error_p99=float(np.quantile(velocity_errors,.99)),
        physics_lookup_velocity_error_max=max(velocity_errors),
        physics_lookup_contact_disagreement=float(np.mean(contact_errors)),
        sensor_mean_error_max_by_action=np.max(sensor_errors,axis=0).tolist(),
        sensor_mean_error_rmse_by_action=np.sqrt(np.mean(np.square(sensor_errors),axis=0)).tolist(),
        reference_action_counts={str(a):actions.count(a) for a in [-1,0,1,2]},
        max_repeat_gain=max(repeat),quadrature257_vs513_max_gain_error=max(quadrature_errors),
        dense129x33_vs257x65_max_gain_error=max(refinement_errors),
        dense129x33_vs257x65_action_agreement=float(np.mean(agreements)))
    Path('results').mkdir(exist_ok=True)
    Path('results/development_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()
