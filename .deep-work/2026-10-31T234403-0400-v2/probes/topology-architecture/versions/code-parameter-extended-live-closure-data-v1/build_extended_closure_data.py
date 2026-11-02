"""New fixed DEVELOPMENT continuations, same512 original base cases.

No NN, truth/oracle import or fitting. Four trained-cache words are disclosed;
NOP-inserted, composed and CLEAR continuations remain outside that cache.
Maximum50 tokens; <=48 subset indices fixed by length, not model outcomes.
"""
import argparse,hashlib,json,time,resource
from pathlib import Path
import numpy as np
SUFFIXES=[[],[1,3],[2,3],[14],[1,11,3],[2,11,3],[1,3,2,3]]
NAMES=['empty','SET0_STEP','SET1_STEP','CLEAR','SET0_NOP_STEP','SET1_NOP_STEP','SET0_STEP_SET1_STEP']
def transform(q,s):return(q,(q+1)%8,(3*q+2)%8,0,(q+1)%8,(3*q+2)%8,(3*q+5)%8)[s]
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
 with np.load(a.inputs/'closure.npz',allow_pickle=False)as z:old={k:z[k]for k in z.files}
 ids=np.flatnonzero(old['suffix']==0)
 if len(ids)!=512 or set(old['condition'][ids])!={0,1}:raise ValueError('exact original512 base cases')
 rows=[];tokens=[]
 for i in ids:
  length=int(old['lengths'][i]);q=int(old['targets'][i]);base=old['tokens'][i,:length]
  for suffix,word in enumerate(SUFFIXES):
   value=np.zeros(64,dtype=np.int64);value[:length]=base;value[length:length+len(word)]=word;tokens.append(value)
   rows.append([old['initial'][i],length+len(word),transform(q,suffix),old['source_train_index'][i],old['base_local_index'][i],old['condition'][i],suffix,int(length+len(word)<=48)])
 raw=np.asarray(rows,dtype=np.int64);keys=['initial','lengths','targets','source_train_index','base_local_index','condition','suffix','within_original48']
 arrays={k:raw[:,j]for j,k in enumerate(keys)};arrays['tokens']=np.asarray(tokens)
 if arrays['lengths'].max()>50:raise ValueError('fixed maximal extension')
 path=a.out/'closure.npz'
 with path.open('xb')as f:np.savez_compressed(f,**arrays)
 meta=dict(schema='extended-live-closure-development-v1',rows=3584,base_cases=512,suffixes=SUFFIXES,suffix_names=NAMES,source_sha256=hashlib.sha256((a.inputs/'closure.npz').read_bytes()).hexdigest(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data_bytes=path.stat().st_size,arrays_bytes=sum(v.nbytes for v in arrays.values()),maximum_length=int(arrays['lengths'].max()),within_original48_by_suffix={str(s):int(arrays['within_original48'][arrays['suffix']==s].sum())for s in range(7)},generation_wall_seconds=time.perf_counter()-tick,cpu_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,NN_queries=0,new_external_labels=0,derived_terminal_labels=3584,teacher_costs='original base labels reused from already generated data; fixed old-task suffix semantics evaluator only',cache_training_overlap={'SET0_STEP':True,'SET1_STEP':True,'others':False},scope='new postdesign DEVELOPMENT validation; no replacement of oldTEST; allmodels same max50, fixed <=48 strata report length confound')
 (a.out/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
