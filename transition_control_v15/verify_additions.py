from common import *
require();checks={}
for dev in ['gpu','cpu']:
    assert read(R/f'timing_optimized_{dev}_done.json')['source']==sha('optimized_benchmark.py')
    for name,v in read(R/f'timing_optimized_{dev}.json').items():
        assert v['n']==300 and v['max_prediction_error']<1e-9
        if v['max_acquisition_error'] is not None:assert v['max_acquisition_error']<1e-9
checks['optimized_runtime_actions_predictions_scores_verified']=True
v=read(R/'optimized_gpu_worker_exit.json');assert v['code']==0 and v['foreign']==[]
checks['optimized_gpu_worker_success']=True
c=read(R/'mixture_counterexample.json');a=c['uninformative_mixture'];b=c['informative_weak_view'];assert abs(a['expected_Brier']-.25)<1e-12 and a['pair_score']>b['pair_score'] and a['expected_Brier']>b['expected_Brier']
checks['post_outcome_abstract_counterexample_verified']=True
write('ADDITIONAL_INTEGRITY.json',checks);print(checks)
