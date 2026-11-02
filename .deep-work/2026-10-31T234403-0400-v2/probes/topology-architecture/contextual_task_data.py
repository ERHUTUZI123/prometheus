"""Owned terminal-only context/memory task generator; not imported by learners.

No desired primitive, intermediate state, route, mode or memory value is stored
in the model-input directory. All methods see identical raw tokens and labels.
No task ID accompanies an example. Three stages add new tokens under a common
unchanged transducer. Native fixed-D static models and expanded2D controls are
both legitimate comparisons; this is development data, not frozen Primary.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

TOKENS={'PAD':0,'SET0':1,'SET1':2,'STEP':3,'FLIP':4,'SAVE':5,'LOAD':6,
        'KEY0':7,'KEY1':8,'WRITE':9,'READ':10,'NOP':11,'STEP2':12,'ADDRFLIP':13,'CLEAR':14}

def terminal(initial,word,n):
    q=int(initial);mode=0;saved=0;address=0;cells=[0,0]
    for token in word:
        if token==1:mode=0
        elif token==2:mode=1
        elif token==3:q=(q+1)%n if mode==0 else (3*q+2)%n
        elif token==4:mode=1-mode
        elif token==5:saved=q
        elif token==6:q=saved
        elif token==7:address=0
        elif token==8:address=1
        elif token==9:cells[address]=q
        elif token==10:q=cells[address]
        elif token==12:q=q//2 if mode==0 else (q%2)*(n//2)
        elif token==13:address=1-address
        elif token==14:q=0
        elif token not in (0,11):raise ValueError(token)
    return q

def make_rows(stage,count,n,rng,split):
    allowed=([1,2,3,11] if stage==0 else [1,2,3,4,5,6,11,14] if stage==1 else list(range(1,15)))
    rows=[]
    # TRAIN-only short identification coverage is disclosed and provided to all.
    if split=='train':
        for q in range(n):
            for mode in (1,2):
                for w in ([mode],[mode,3],[mode,11,3],[mode,3,3]):rows.append((q,w))
        if stage>=1:
            for q in range(n):
                for w in ([5,14,6],[1,4,3],[2,4,3],[5,3,6]):rows.append((q,w))
        if stage>=2:
            for q in range(n):
                for key in (7,8):
                    for w in ([key,9,14,10],[key,9,3,10],[1,12],[2,12]):rows.append((q,w))
    while len(rows)<count:
        q=int(rng.integers(n))
        if split=='test_unseen':
            if stage==0:w=[int(rng.choice([1,2])),11,11,3,3,11,3,11,3]
            elif stage==1:w=[int(rng.choice([1,2])),5,4,3,11,3,6,4,3]
            else:w=[7,9,3,8,9,14,7,10,13,10,12]
            for _ in range(int(rng.integers(4,10))):w.insert(int(rng.integers(len(w)+1)),11)
        else:
            length=int(rng.integers(24,49)) if split=='test_long' else int(rng.integers(2,13))
            w=[int(x) for x in rng.choice(allowed,size=length)]
        rows.append((q,w))
    rows=rows[:count]
    initial=np.array([q for q,w in rows],dtype=np.int64)
    lengths=np.array([len(w) for q,w in rows],dtype=np.int64)
    tokens=np.zeros((count,48),dtype=np.int64)
    for i,(q,w) in enumerate(rows):tokens[i,:len(w)]=w
    targets=np.array([terminal(q,w,n) for q,w in rows],dtype=np.int64)
    return dict(initial=initial,tokens=tokens,lengths=lengths,targets=targets)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--states',type=int,default=8);p.add_argument('--seed',type=int,default=0);p.add_argument('--max-seconds',type=float,default=60);a=p.parse_args()
    if a.states not in (8,16,32):raise ValueError('declared finite scales only')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    meta={'scope':'development only, no frozenPrimary','n_states':a.states,'n_tokens':15,'max_length':48,'stages':[],
        'token_ids':TOKENS,'seed':a.seed,'supervision':'initial state, raw tokens, length, terminal class only; no task ID',
        'train_short_coverage':'all initial states with disclosed SET/STEP prefixes; later stages add short SAVE/LOAD and addressed WRITE/READ witnesses',
        'all_methods_same_files':True,'evaluation':'IID, length24-48, held-out structured combinations; source oracle excluded from learner code and readonly inputs'}
    for stage in range(3):
        record={'stage':stage,'files':{},'hashes':{},'duplicates':{}}
        train_words=set()
        for split,count in [('train',4096),('valid',256),('test_iid',512),('test_long',512),('test_unseen',512)]:
            rng=np.random.default_rng(20261102+10000*a.seed+100*stage+{'train':0,'valid':1,'test_iid':2,'test_long':3,'test_unseen':4}[split])
            arrays=make_rows(stage,count,a.states,rng,split);path=a.out/f'stage{stage}-{split}.npz';np.savez_compressed(path,**arrays)
            rows={(int(q),tuple(w[:int(l)])) for q,w,l in zip(arrays['initial'],arrays['tokens'],arrays['lengths'])}
            if split=='train':train_words=rows
            record['files'][split]=path.name;record['hashes'][split]=hashlib.sha256(path.read_bytes()).hexdigest()
            record['duplicates'][split]={'rows':count,'unique':len(rows),'unique_overlap_with_train':len(rows&train_words) if split!='train' else None}
            if time.monotonic()-started>a.max_seconds:raise TimeoutError('bounded data generation')
        meta['stages'].append(record)
    (a.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'out':str(a.out),'stages':3,'wall_seconds':time.monotonic()-started}))
if __name__=='__main__':main()
