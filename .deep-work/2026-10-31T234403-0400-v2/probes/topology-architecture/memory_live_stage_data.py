"""Offline-only fourth-task generator; never imported or supplied to a learner.

Repairs the old overwritten-first-READ measurement gap prospectively. Both
written values causally affect a new MIXREAD terminal query. Old data is not
read or changed. No source, truth rows, intermediate states or routes enter
the learner's data directory. Finite SAMEV2 development, not frozen Primary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

TOKENS = dict(PAD=0, SET0=1, SET1=2, STEP=3, FLIP=4, SAVE=5, LOAD=6,
              KEY0=7, KEY1=8, WRITE=9, READ=10, NOP=11, STEP2=12,
              ADDRFLIP=13, CLEAR=14, MIXREAD=15)
SPLITS = (("train",4096),("valid",512),("test_iid",512),
          ("test_long",512),("test_unseen",512))
KEYS = ("initial","tokens","lengths","targets")
HELDOUT = (4,14,2,3,1,3)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal(initial, word):
    """Correct semantics lives ONLY in this offline source, never in training."""
    q=int(initial); mode=0; saved=0; address=0; cells=[0,0]
    for t in word:
        if t==1: mode=0
        elif t==2: mode=1
        elif t==3: q=(q+1)%8 if mode==0 else (3*q+2)%8
        elif t==4: mode=1-mode
        elif t==5: saved=q
        elif t==6: q=saved
        elif t==7: address=0
        elif t==8: address=1
        elif t==9: cells[address]=q
        elif t==10: q=cells[address]
        elif t==12: q=q//2 if mode==0 else (q%2)*4
        elif t==13: address=1-address
        elif t==14: q=0
        elif t==15: q=(3*q+cells[address])%8
        elif t not in (0,11): raise ValueError(t)
    return q


def has_heldout(word):
    return any(tuple(word[i:i+len(HELDOUT)])==HELDOUT
               for i in range(len(word)-len(HELDOUT)+1))


def distractor(rng,split):
    # All distractors follow BOTH writes and contain NO memory write. The
    # short held-out composition never occurs in TRAIN/VALID/IID/long.
    if split=="test_unseen":
        pre=[int(x) for x in rng.choice([1,2,3,4,14],size=int(rng.integers(0,5)))]
        post=[int(x) for x in rng.choice([1,2,3,4,14],size=int(rng.integers(0,5)))]
        return pre+list(HELDOUT)+post
    size=int(rng.integers(52,87)) if split=="test_long" else int(rng.integers(6,23))
    while True:
        word=[int(x) for x in rng.choice([1,2,3,4,14],size=size)]
        if not has_heldout(word): return word


def program(v0,v1,noise,order,write_order):
    write0=[7,14,1]+[3]*v0+[9]
    write1=[8,14,1]+[3]*v1+[9]
    suffix=[7+order,10,8-order,15]
    writes=write0+write1 if write_order==0 else write1+write0
    return writes+noise+suffix


def make_rows(count,rng,split):
    if count%256: raise ValueError("balanced 8x8x2x2 groups required")
    rows=[]; truth=[]; pairs=[]
    # Every independently sampled distractor template contains the complete
    # uniformly shuffled value-pair x query-order x write-order grid.
    # Write order and query order vary independently; fixed recency/position
    # cannot replace binding a value to its key. Consequently every
    # suffix/order/template has exactly uniform terminal labels. Varying only
    # v0 OR v1 yields a paired counterfactual with the exact same suffix.
    for group in range(count//256):
        noise=distractor(rng,split); initial=int(rng.integers(8)); lookup={}
        grid=[(v0,v1,order,write_order) for v0 in range(8) for v1 in range(8)
              for order in range(2) for write_order in range(2)]
        for j in rng.permutation(256):
            v0,v1,order,write_order=grid[int(j)]; word=program(v0,v1,noise,order,write_order)
            y=terminal(initial,word); row=len(rows)
            assert y==(3*(v1 if order else v0)+(v0 if order else v1))%8
            rows.append((initial,word,y)); truth.append((group,v0,v1,order,write_order))
            lookup[(v0,v1,order,write_order)]=row
        for v0,v1,order,write_order in grid:
            row=lookup[(v0,v1,order,write_order)]
            pairs.append((row,lookup[((v0+1)%8,v1,order,write_order)],0))
            pairs.append((row,lookup[(v0,(v1+1)%8,order,write_order)],1))
    tokens=np.zeros((count,128),dtype=np.int64)
    for i,(_,word,_) in enumerate(rows): tokens[i,:len(word)]=word
    data=dict(initial=np.asarray([r[0] for r in rows],dtype=np.int64),tokens=tokens,
              lengths=np.asarray([len(r[1]) for r in rows],dtype=np.int64),
              targets=np.asarray([r[2] for r in rows],dtype=np.int64))
    return data,np.asarray(truth,dtype=np.int64),np.asarray(pairs,dtype=np.int64)


def audit(data,truth,pairs,split):
    counts=np.bincount(data["targets"],minlength=8)
    assert len(set(counts.tolist()))==1
    assert np.array_equal(np.bincount(truth[:,3],minlength=2),[len(truth)//2]*2)
    assert np.array_equal(np.bincount(truth[:,4],minlength=2),[len(truth)//2]*2)
    joint_order=np.bincount(2*truth[:,3]+truth[:,4],minlength=4)
    assert np.array_equal(joint_order,[len(truth)//4]*4)
    conditional_counts=[]
    for query_order in range(2):
        for write_order in range(2):
            conditional=np.bincount(data["targets"][(truth[:,3]==query_order)&(truth[:,4]==write_order)],minlength=8)
            assert np.array_equal(conditional,[len(truth)//32]*8)
            conditional_counts.append(conditional.tolist())
    low,high=int(data["lengths"].min()),int(data["lengths"].max())
    assert (64<=low<=high<=112) if split=="test_long" else (high<=48)
    for row,(group,v0,v1,order,write_order) in enumerate(truth):
        word=data["tokens"][row,:data["lengths"][row]].tolist()
        assert word[-4:]==[7+order,10,8-order,15]
        assert word.count(9)==2 and word.count(10)==1 and word.count(15)==1
        noise=word[8+int(v0)+int(v1):-4]
        assert all(t in (1,2,3,4,14) for t in noise)
        assert word==program(int(v0),int(v1),noise,int(order),int(write_order))
        assert has_heldout(noise)==(split=="test_unseen")
        assert terminal(data["initial"][row],word)==int(data["targets"][row])
    changes={0:0,1:0}
    for left,right,changed in pairs:
        a,b=truth[left],truth[right]
        assert a[0]==b[0] and a[3]==b[3] and a[4]==b[4]
        assert a[2-changed]==b[2-changed]
        assert b[1+changed]==(a[1+changed]+1)%8
        assert data["initial"][left]==data["initial"][right]
        wa=data["tokens"][left,:data["lengths"][left]].tolist()
        wb=data["tokens"][right,:data["lengths"][right]].tolist()
        assert wa[-4:]==wb[-4:]
        assert wa[8+int(a[1])+int(a[2]):-4]==wb[8+int(b[1])+int(b[2]):-4]
        assert data["targets"][left]!=data["targets"][right]
        changes[int(changed)]+=1
    return dict(rows=len(truth),counterfactual_pairs=len(pairs),both_written_values_causal=True,
                counterfactual_target_changes_by_written_value={
                    "v0":dict(pairs=int((pairs[:,2]==0).sum()),changed=changes[0]),
                    "v1":dict(pairs=int((pairs[:,2]==1).sum()),changed=changes[1])},
                query_order_counts=np.bincount(truth[:,3],minlength=2).tolist(),
                write_order_counts=np.bincount(truth[:,4],minlength=2).tolist(),
                independent_query_write_order_counts=joint_order.tolist(),
                target_counts_by_query_then_write_order=conditional_counts,
                target_counts=counts.tolist(),min_length=low,max_length=high,
                exact_query_suffix_matched=True,heldout_composition_excluded_outside_unseen=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--max-seconds",type=float,default=60)
    a=p.parse_args()
    if a.seed!=0 or not 0<a.max_seconds<=60: raise ValueError("prospective seed0 bounded generator")
    began=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False)
    data_dir=a.out/"data";data_dir.mkdir();truth_arrays={};receipts={};audits={}
    for index,(split,count) in enumerate(SPLITS):
        rng=np.random.default_rng(np.random.SeedSequence([20261102,a.seed,3,index,1503]))
        data,truth,pairs=make_rows(count,rng,split);audits[split]=audit(data,truth,pairs,split)
        path=data_dir/("stage3-"+split+".npz")
        with path.open("xb") as stream: np.savez_compressed(stream,**data)
        receipts[split]=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,
                             rows=count,array_bytes=sum(v.nbytes for v in data.values()))
        truth_arrays[split+"_values_group_order"]=truth;truth_arrays[split+"_pairs"]=pairs
        if time.perf_counter()-began>a.max_seconds: raise TimeoutError("finite data generation deadline")
    truth_path=a.out/"counterfactual-truth.npz"
    with truth_path.open("xb") as stream: np.savez_compressed(stream,**truth_arrays)
    meta=dict(schema="memory-live-stage-data-v1",formal_Primary=False,n_states=8,n_tokens=16,
              max_length=128,stage=3,seed=a.seed,files=receipts,
              source_sha256=sha(Path(__file__)),counterfactual_truth_sha256=sha(truth_path),
              counts=dict(SPLITS),audits=audits,token_ids=TOKENS,
              labels_created=sum(c for _,c in SPLITS),
              terminal_oracle_calls_total=sum(2*c for _,c in SPLITS),
              terminal_oracle_calls_for_creation=sum(c for _,c in SPLITS),
              terminal_oracle_calls_for_generator_audit=sum(c for _,c in SPLITS),
              labels_reaccessed_in_generator_audit=sum(7*c for _,c in SPLITS),
              generator_target_read_scope="global+conditional bincounts + one row-oracle comparison + four paired-target accesses per row; paired reads are included in audit total",
              counterfactual_label_pair_checks=sum(2*c for _,c in SPLITS),
              counterfactual_target_label_reads=sum(4*c for _,c in SPLITS),
              truth_columns=["template_group","v0","v1","query_order","write_order"],
              counterfactual_pair_columns=["source_row","counterfactual_row","changed_written_value_index"],
              no_rejection_by_label=True,query_write_orders_and_targets_exactly_balanced=True,
              learner_inputs="initial/raw tokens/length/terminal label only; no taskID/routes/intermediates",
              truth_excluded_from_learner_directory=True,
              scope="prospective variable-write-order two-key binding and live-value composition; no arbitrary-key/content-address generalization claim; old datasets/results immutable",
              creation_wall_seconds=time.perf_counter()-began)
    (data_dir/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    (a.out/"generator-receipt.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(dict(status="DATA_AND_COUNTERFACTUAL_AUDIT_PASSED",out=str(a.out),**meta)))


if __name__=="__main__": main()
