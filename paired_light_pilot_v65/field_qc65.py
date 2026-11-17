"""Validate one paired bank. All recorded times below use the common trigger clock (microseconds)."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def require(check, why):
    if not check:raise ValueError(why)

def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def validate(meta, base):
    base=Path(base).resolve()
    def artifact(rec,role):
        if rec is None:return False
        require(rec['role']==role, 'PROVENANCE_ROLE')
        path=(base/rec['path']).resolve()
        require(base in path.parents and path.is_file(),'MISSING_OR_EXTERNAL_FILE')
        require(sha_file(path)==rec['sha256'],'HASH_MISMATCH')
        return True
    def missing_reason(item):
        require(isinstance(item.get('missing_reason'),str) and item['missing_reason'].strip(),'MISSING_REASON')
    def avail(a):
        if a.get('raw') is None:
            missing_reason(a)
            return False
        return True
    def policy_signal(raw,role):
        if not avail(raw):return False
        return artifact(raw['raw'],role)
    require(meta['format']=='paired-light-v65.1' and meta['origin'] in ('physical','schema_test'),'FORMAT')
    for key in ('surface_instance','surface_family','surface_day','session','board_id','clock_id'):
        require(bool(meta[key]),'GROUP_OR_CLOCK_MISSING')
    lim=meta['qc_budget']
    for key in ('max_exposure_midpoint_offset_us','max_clock_error_us'):
        require(0<lim[key]<float('inf'),'INVALID_SYNC_BUDGET')
    sensors=meta['devices']
    require(len(set(sensors['identity_sources']+sensors['strain_sources']))==
            len(sensors['identity_sources'])+len(sensors['strain_sources']),'AUDIT_CHANNEL_REUSED')
    require(set(sensors['policy_sources']).isdisjoint(sensors['identity_sources']+sensors['strain_sources']),
            'AUDIT_POLICY_CHANNEL_REUSED')
    require(len({sensors['lamp_A'],sensors['lamp_B']})==2,'ONE_LAMP')
    by_id={}; object_ids=set(); audit_count=dict(identity=0,local_strain=0)
    paired_quality=True; all_evidence={'A':True,'B':True}
    slots=meta['calibration_slots']
    require(len(slots)==8,'BUDGET')
    for pos,s in enumerate(slots):
        require(s['slot']==pos and s['reset_pair']==pos//2,'SLOT_ORDER')
        require(s['object_id'] not in object_ids,'REUSED_CALIBRATION_OBJECT')
        object_ids.add(s['object_id'])
        require(s['reset_electronic_id'] is None if pos%2==0 else s['reset_electronic_id'] in (0,1),'RESET')
        for kind in ('reference','impact'):
            e=s[kind]; event=e['trigger_id']
            require(event and event not in by_id,'DUPLICATE_TRIGGER')
            by_id[event]=1
            electrical=policy_signal(e['electronics'],'policy_electronics')
            times={}
            for lamp in ('A','B'):
                frame=e['light'][lamp]
                require(frame['trigger_id']==event and frame['lamp_device']==sensors['lamp_'+lamp],'UNPAIRED_EXPOSURE')
                require(frame['source'] in sensors['policy_sources'],'POLICY_INPUT_PROVENANCE')
                ok=policy_signal(frame,'policy_frame')
                if ok:
                    ts=frame['mid_us']
                    require(len(ts)==8 and all(math.isfinite(v) for v in ts) and
                            all(x<y for x,y in zip(ts,ts[1:])), 'FRAME_TIMESTAMP')
                    require(math.isfinite(frame['clock_error_us']) and frame['clock_error_us']>=0,'CLOCK_ERROR')
                    paired_quality &= frame['clock_error_us']<=lim['max_clock_error_us']
                    times[lamp]=ts
                all_evidence[lamp] &= ok and electrical
            if len(times)==2:
                paired_quality &= max(abs(x-y) for x,y in zip(times['A'],times['B']))<=lim['max_exposure_midpoint_offset_us']
            else:paired_quality=False
            for typ,role in [('identity','identity_audit'),('local_strain','strain_audit')]:
                a=e['audit'][typ]
                if avail(a):
                    require(a['source'] in sensors['identity_sources' if typ=='identity' else 'strain_sources'],'AUDIT_INPUT_PROVENANCE')
                    require(a['trigger_id']==event,'AUDIT_TRIGGER')
                    artifact(a['raw'],role)
                    audit_count[typ]+=1
    require(meta['deployment_object'] not in object_ids,'REUSED_DEPLOYMENT_OBJECT')
    prefix=meta['deployment_prefix']
    complete=policy_signal(prefix,'shared_three_reference_prefix')
    if complete:
        require(prefix['count']==3,'PREFIX_LENGTH')
    arm=meta['assigned_lamp']; require(arm in ('A','B'),'ASSIGNMENT')
    require(0<meta['lamp_A_probability']<1,'PROPENSITY')
    choices=[]
    for lamp in ('A','B'):
        r=meta['recommendations'][lamp]
        val=r['choice']
        require(val in ('R','I',None),'PURCHASE')
        if val is None:
            missing_reason(r)
        else:
            require(complete and all_evidence[lamp], 'CHOICE_WITHOUT_REPLAYABLE_INPUT')
            require(r['common_prefix_sha256']==prefix['raw']['sha256'],'NONCOMMON_PREFIX')
            require(math.isfinite(r['sealed_us']),'UNSEALED')
        choices.append(val)
    receipt=meta['execution_receipt']
    delivered=avail(receipt)
    if delivered:
        require(artifact(receipt['raw'],'actuator_receipt') and choices['AB'.index(arm)]==receipt['executed'],'RECEIPT_DISAGREES')
        require(all(r['sealed_us']<receipt['executed_us'] for r in meta['recommendations'].values() if r['choice']),'POST_EXECUTION_CHOICE')
    evaluable=paired_quality and None not in choices
    return dict(surface=meta['surface_instance'],session=meta['session'],assigned_lamp=arm,
        impacts=len(slots),references=len(slots),surface_session_count=1,recorded_choices=choices,
        observed_terminal_pair=evaluable,paired_recommendation_flip=(choices[0]!=choices[1]) if evaluable else None,
        pair_sync_qc=paired_quality,acquisition_complete=all_evidence,verification_available=audit_count,
        receipt_verified=delivered,executed_choice=(receipt['executed'] if delivered else None))

if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('session',type=Path)
    parsed=args.parse_args()
    print(json.dumps(validate(json.loads(parsed.session.read_text()),parsed.session.parent),indent=2))
