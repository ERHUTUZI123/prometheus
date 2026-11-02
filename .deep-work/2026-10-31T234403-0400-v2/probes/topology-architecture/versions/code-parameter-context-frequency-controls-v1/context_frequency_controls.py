"""Bounded TRAIN-only constant/suffix controls for the frozen context data.

No model selection, hidden simulator, task identities at inference, gradient
training or new feedback. Per-stage diagnostic fits are separate tasks, not
a continual retention solution. Report distribution and lookup costs openly.
"""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False);start=time.monotonic();result=[]
    def read(stage,split):
        f=a.data/f'stage{stage}-{split}.npz'
        with np.load(f,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        return d,hashlib.sha256(f.read_bytes()).hexdigest()
    def key(d,i,order):
        length=int(d['lengths'][i]);return (int(d['initial'][i]),*map(int,d['tokens'][i,max(0,length-order):length]))
    for stage in range(3):
        train,h=read(stage,'train');counts=collections.Counter(map(int,train['targets']));major=counts.most_common(1)[0][0];tables={order:{}for order in [1,2]}
        for order,table in tables.items():
            for i,label in enumerate(train['targets']):table.setdefault(key(train,i,order),collections.Counter())[int(label)]+=1
        summary={'stage':stage,'TRAIN_sha256':h,'TRAIN_counts':dict(counts),'TRAIN_majority_class':major,'rows_used':len(train['targets']),'fit_label_reads':len(train['targets'])*3,'scope':'separate per-stage diagnostic fit, not a continual task-ID-free system','lookups':{str(o):{'entries':len(t),'serialized_counter_bytes':len(json.dumps([[list(k),dict(v)]for k,v in t.items()]).encode())}for o,t in tables.items()},'splits':{}}
        for split in ['valid','test_iid','test_long','test_unseen']:
            d,h=read(stage,split);labels=d['targets'];hist=collections.Counter(map(int,labels));v={'sha256':h,'rows':len(labels),'target_counts':dict(hist),'TRAIN_constant_accuracy':float(np.mean(labels==major))}
            for order,t in tables.items():
                pred=[t.get(key(d,i,order),{major:1})for i in range(len(labels))]
                # Deterministic tie breaking by label, never by held-out data.
                ys=np.array([min(c,key=lambda y:(-c[y],y))for c in pred]);v[f'suffix{order}_accuracy']=float(np.mean(ys==labels));v[f'suffix{order}_misses']=sum(key(d,i,order)not in t for i in range(len(labels)))
            summary['splits'][split]=v
        result.append(summary)
    if time.monotonic()-start>30:raise TimeoutError('analysis30s bound')
    (a.out/'summary.json').write_text(json.dumps({'scope':'frozen diagnostic controls; not Primary or general program learning','results':result,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
