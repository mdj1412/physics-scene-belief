"""Exercise a labelled schema fixture with missing metrology and a single actuated arm."""
from copy import deepcopy
import json,os
from pathlib import Path
from field_qc65 import validate,sha_file
ROOT=Path(__file__).resolve().parent
fixture=ROOT/'contract_demo65'

def create():
    # These tiny placeholders exercise the record contract, never serve as field observations.
    fixture.mkdir(exist_ok=True)
    def rec(path,role):
        f=fixture/path
        f.write_text(f'schema_test ONLY: {role} {path}\n')
        return {'path':path,'role':role,'sha256':sha_file(f)}
    def missing(why):return {'raw':None,'missing_reason':why}
    slots=[]
    for ix in range(8):
        slot={'slot':ix,'object_id':f'object{ix}','reset_pair':ix//2,
              'reset_electronic_id':None if ix%2==0 else ix%3%2}
        for kind in ('reference','impact'):
            event=f'E{ix}_{kind}'
            slot[kind]={'trigger_id':event,'electronics':{'raw':rec(event+'_elec.txt','policy_electronics')},
                'light':{a:dict(trigger_id=event,lamp_device='led'+a,source='visible'+a,
                   mid_us=[(2*ix+int(kind=='impact'))*20000+j*1000+(a=='B') for j in range(8)],
                   clock_error_us=1,raw=rec(event+a+'.txt','policy_frame')) for a in ('A','B')},
                'audit':{'identity':missing('schema example has no independent identity'),
                         'local_strain':missing('schema example has no local strain')}}
        slots.append(slot)
    pref=rec('three_ref.txt','shared_three_reference_prefix')
    return dict(format='paired-light-v65.1',origin='schema_test',session='demo01',surface_instance='sample',surface_day='d1',
                surface_family='demo',board_id='board1',clock_id='schema-trigger',
                devices=dict(lamp_A='ledA',lamp_B='ledB',policy_sources=['visibleA','visibleB'],
                    identity_sources=['nir_camera'],strain_sources=['bridge']),
                qc_budget=dict(max_exposure_midpoint_offset_us=20,max_clock_error_us=5),calibration_slots=slots,
                deployment_object='held-out-object',deployment_prefix={'raw':pref,'count':3},
                assigned_lamp='A',lamp_A_probability=.5,recommendations={
                    k:dict(choice=v,sealed_us=500000,common_prefix_sha256=pref['sha256']) for k,v in [('A','I'),('B','R')]},
                execution_receipt=dict(raw=rec('receipt.txt','actuator_receipt'),executed='I',executed_us=500020))

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    src=create();(fixture/'session65.json').write_text(json.dumps(src,indent=2)+'\n')
    ref=validate(src,fixture)
    assert ref['observed_terminal_pair'] and ref['paired_recommendation_flip'] and ref['receipt_verified']
    results={'baseline':ref}
    fixes={
        'different_trigger':lambda s:s['calibration_slots'][0]['impact']['light']['B'].update(trigger_id='different'),
        'duplicate_impact':lambda s:s['calibration_slots'][1]['impact'].update(trigger_id='E0_impact'),
        'duplicated_object':lambda s:s['calibration_slots'][1].update(object_id='object0'),
        'prefix_leakage':lambda s:s['recommendations']['B'].update(common_prefix_sha256='other'),
        'post_actuation':lambda s:s['recommendations']['B'].update(sealed_us=900000),
        'different_receipt':lambda s:s['execution_receipt'].update(executed='R'),
        'file_digest':lambda s:s['calibration_slots'][0]['impact']['light']['A']['raw'].update(sha256='corrupt'),
        'audit_passed_as_pixels':lambda s:s['calibration_slots'][0]['impact']['light']['A'].update(source='nir_camera'),
        'shared_verification_sensor':lambda s:s['devices'].update(strain_sources=['nir_camera']),
        'retake_counted':lambda s:s['calibration_slots'].append(s['calibration_slots'][0]),
        'known_wrong_temporal_pair':lambda s:s['calibration_slots'][0]['impact']['light']['B'].update(mid_us=list(range(8))),
        'missing_B_raw':lambda s:(s['calibration_slots'][0]['impact']['light']['B'].update(raw=None,missing_reason='lost recording'),s['recommendations']['B'].update(choice=None,missing_reason='lost B recording')),
        'one_verified_strain':lambda s:s['calibration_slots'][0]['impact']['audit'].update(local_strain={'source':'bridge','trigger_id':'E0_impact','raw':record_strain()}),
        'missing_receipt':lambda s:s.update(execution_receipt={'raw':None,'missing_reason':'actuator log lost'})}
    def record_strain():
        name='demo_gauge.txt';f=fixture/name;f.write_text('schema fixture only\n')
        return dict(role='strain_audit',path=name,sha256=sha_file(f))
    for name,change in fixes.items():
        s=deepcopy(src);change(s)
        try:r=validate(s,fixture)
        except ValueError as e:
            if name in ('known_wrong_temporal_pair','one_verified_strain','missing_B_raw','missing_receipt'):raise
            results[name]={'rejected':str(e)}
        else:
            if name in ('known_wrong_temporal_pair','missing_B_raw'):assert not r['observed_terminal_pair'] and r['paired_recommendation_flip'] is None
            elif name=='one_verified_strain':assert r['paired_recommendation_flip'] and r['verification_available']['local_strain']==1
            elif name=='missing_receipt':assert r['paired_recommendation_flip'] and not r['receipt_verified']
            else:raise AssertionError(name)
            results[name]=r
    (ROOT/'qc65.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
