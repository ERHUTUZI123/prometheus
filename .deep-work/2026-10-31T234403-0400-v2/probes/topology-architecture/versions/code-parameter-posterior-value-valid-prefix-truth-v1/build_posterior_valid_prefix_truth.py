"""PRIVILEGED offline audit truth ONLY; never training/NN/routing input.

Replay exactly existing512stage3VALID programs. Every new prefixq is checked
against pinned original terminal(). No program generation or model query.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import official_memory_oracle as oracle
EVENTS=('write_first','write_second','after_read','before_mix_after_key_other','at_mix')
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
ORACLE_SHA='8943cd6dd4d638224dcc7be62df82e505b871b32bbcf1be5de3be3afadb92726'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def step(state,t):
    q,mode,saved,address,c0,c1=map(int,state);cells=[c0,c1]
    if t==1:mode=0
    elif t==2:mode=1
    elif t==3:q=(q+1)%8 if mode==0 else(3*q+2)%8
    elif t==4:mode=1-mode
    elif t==5:saved=q
    elif t==6:q=saved
    elif t==7:address=0
    elif t==8:address=1
    elif t==9:cells[address]=q
    elif t==10:q=cells[address]
    elif t==12:q=q//2 if mode==0 else(q%2)*4
    elif t==13:address=1-address
    elif t==14:q=0
    elif t==15:q=(3*q+cells[address])%8
    elif t not in(0,11):raise ValueError('unknown original instruction')
    return(q,mode,saved,address,*cells)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cpu=time.process_time()
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen privilegedsource changed')
        if sha(Path(oracle.__file__))!=ORACLE_SHA or sha(a.inputs/'valid.npz')!=VALID_SHA:raise ValueError('exact original oracle/VALID bytes required')
        with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:
            if set(z.files)!={'initial','tokens','lengths','targets'}:raise ValueError('exact original publicfields required')
            data={k:z[k]for k in z.files}
        if data['tokens'].shape!=(512,128)or any(v.dtype!=np.int64 for v in data.values()):raise ValueError('fixed512int64VALID only')
        states=np.full((512,129,6),-1,dtype=np.int64);positions=np.empty((512,5),dtype=np.int64);events_before=np.empty((512,5,6),dtype=np.int64);events_after=events_before.copy();query=np.empty((512,5),dtype=np.int64);calls=0;transitions=0
        for row in range(512):
            length=int(data['lengths'][row]);word=data['tokens'][row,:length].tolist()
            if not 1<=length<=48 or any(data['tokens'][row,length:])or word[-1]!=15 or word.count(15)!=1 or word.count(9)!=2 or word.count(10)!=1 or word[-4]not in(7,8)or word[-2]!=15-word[-4]:raise ValueError('exact original write/query/MIX publiclayout required')
            writes=[i for i,t in enumerate(word)if t==9];read=word.index(10);positions[row]=[writes[0],writes[1],read,length-2,length-1]
            state=(int(data['initial'][row]),0,0,0,0,0);states[row,0]=state
            if oracle.terminal(int(data['initial'][row]),[])!=state[0]:raise ValueError('official empty semantics mismatch')
            calls+=1
            for column,token in enumerate(word):
                before=state;state=step(state,token);states[row,column+1]=state;transitions+=1
                if oracle.terminal(int(data['initial'][row]),word[:column+1])!=state[0]:raise ValueError('instrumented prefix disagreeswith exactofficialterminal')
                calls+=1
                for event,pos in enumerate(positions[row]):
                    if pos==column:
                        events_before[row,event]=before;events_after[row,event]=state
                        # Query happens before WRITE, but currentKEY has already
                        # selected its address. This is audit-only semantic truth.
                        query[row,event]=before[4+state[3]]
            if state[0]!=int(data['targets'][row]):raise ValueError('originalVALID terminaltarget changed')
            if time.perf_counter()-started>100:raise TimeoutError('bounded privilegedCPUaudit')
        arrays=dict(row_index=np.arange(512,dtype=np.int64),original_initial=data['initial'],original_lengths=data['lengths'],original_targets=data['targets'],event_positions=positions,event_names_json=np.frombuffer(json.dumps(EVENTS).encode(),dtype=np.uint8),true_state_prefix=states,true_state_before_event=events_before,true_state_after_event=events_after,true_selected_memory_at_query=query,query_order=data['tokens'][np.arange(512),data['lengths']-4]-7)
        path=a.out/'prefix-truth.npz'
        with path.open('xb')as f:np.savez_compressed(f,**arrays)
        result=dict(status='PRIVILEGED_VALID_PREFIX_TRUTH_COMPLETE',schema='privileged-VALID-prefix-truth-v1',source_sha256=manifest['source_sha256'],official_oracle_sha256=ORACLE_SHA,VALID_sha256=VALID_SHA,rows=512,events=list(EVENTS),state_columns=['q','mode','saved','address','cell0','cell1'],truth_file=path.name,truth_sha256=sha(path),truth_bytes=path.stat().st_size,truth_array_bytes=sum(v.nbytes for v in arrays.values()),official_terminal_prefix_calls=calls,instrumented_transitions=transitions,all_prefix_q_checked_against_original=True,all512_original_terminal_labels_checked=True,NN_queries=0,SGD_updates=0,new_learner_labels=0,privileged_audit_states=calls,privileged_truth_never_feedback=True,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),scope='offlinePRIVILEGEDvalidationdiagnosis truth only; original512VALID unchanged, no newprograms/training/oracle toNN')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],rows=512,truth_sha256=result['truth_sha256'])))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc()),f,indent=2)
        raise
if __name__=='__main__':main()
