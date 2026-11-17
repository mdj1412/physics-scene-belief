"""Secondary four-fold grouped within-campaign control after whole-campaign transfer."""
import csv
import json
import os

import numpy as np
import torch

from solver import HERE, RADIUS, campaign


class OriginalIndexWriter:
    def __init__(self, writer, name, fold, src, trg):
        self.writer = writer
        self.name = name
        self.fold = fold
        self.src = src
        self.trg = trg

    def writerow(self, row):
        _,_,_,local,split,method,*values=row
        original = int(self.trg[local] if split=='test' else self.src[local])
        self.writer.writerow([self.name,self.fold,original,split,method,*values])


def main():
    records=[]
    with (HERE/'within_predictions.csv').open('w') as handle:
        writer=csv.writer(handle)
        writer.writerow(['campaign','fold','row','split','method','vx_in','vn_in',
                         'rim_in','vx_out','vn_out','rim_out','vx_pred','rim_pred',
                         'u_pred','p025','p050','p100','p_rev'])
        for name in 'AB':
            data=np.loadtxt(HERE.parent/'data69'/'ESM_2'/f'data_campaign{name}.csv',delimiter=',')*RADIUS
            idx=np.arange(len(data))
            group=(idx//24)%4
            for fold in range(4):
                src,tar=idx[group!=fold],idx[group==fold]
                cal=group[src]==((fold+1)%4)
                rowwriter=OriginalIndexWriter(writer,name,fold,src,tar)
                r=campaign((name,data[src]),(name,data[tar]),rowwriter,cal_mask=cal)
                y=(np.abs(data[src[cal],3]+data[src[cal],5])<=.5)
                yt=(np.abs(data[tar,3]+data[tar,5])<=.5)
                freq=(y.sum()+.5)/(len(y)+1)
                r['marginal_brier']=float(((yt-freq)**2).mean())
                r['fold']=fold
                records.append(r)
    means={}
    for name in 'AB':
        sub=[x for x in records if x['source']==name]
        n=sum(x['n_test'] for x in sub)
        means[name]={'n':n,'source_prevalence_brier':sum(x['marginal_brier']*x['n_test'] for x in sub)/n}
        for method in sub[0]['metrics']:
            means[name][method]={}
            for k in sub[0]['metrics'][method]:
                # To give population RMSE, pool squared fold RMSE before taking square root.
                power=2 if k.startswith('rmse') else 1
                means[name][method][k]=(sum(x['metrics'][method][k]**power*x['n_test'] for x in sub)/n)**(1/power)
    out={'slurm_job':os.environ.get('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
         'folds':records,'means':means}
    (HERE/'within_metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__':
    main()
