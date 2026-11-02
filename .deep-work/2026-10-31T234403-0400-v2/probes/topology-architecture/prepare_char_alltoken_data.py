"""ROOT-RUN offline all-token AR windows from OWN prepared PG100 corpus.

TRAIN-only frozen vocabulary and original contiguous90/5/5 splits are reused.
This is new all-token development data/feedback, not the old prefix endpoint
experiment. No download, model, oracle or import-time work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--max-seconds",type=float,default=90)
    args=p.parse_args()
    if not 1<=args.max_seconds<=100:p.error("bounded preparation<=100seconds")
    args.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+args.max_seconds
    def check():
        if time.perf_counter()>deadline:raise TimeoutError("all-token preparation bound")
    source=json.loads((args.prepared/"manifest.json").read_text());vocab=json.loads((args.prepared/"vocab.json").read_text())
    v=vocab["vocabulary_size"]
    if (not isinstance(v,int) or not 2<=v<=512 or vocab.get("unk_id")!=0
            or len(vocab["characters_by_id_from1"])!=v-1 or source.get("no_targets_cross_split") is not True):
        raise ValueError("exact frozen TRAIN-only vocabulary/original split manifest required")
    receipts={};streams={}
    for leaf in ("train.npy","valid.npy","test.npy","vocab.json"):
        check();path=args.prepared/leaf
        if path.stat().st_size>40<<20 or sha(path)!=source["files"][leaf]:raise ValueError("prepared source changed or exceeds bound")
        receipts[leaf]=sha(path)
    for split in ("train","valid","test"):
        ids=np.load(args.prepared/f"{split}.npy",mmap_mode="r",allow_pickle=False)
        if ids.dtype!=np.uint16 or ids.ndim!=1 or np.any(ids>=v):raise ValueError("invalid owned token stream")
        streams[split]=ids
    endpoints={}
    for code,(split,count,minimum) in enumerate((("train",8192,64),("valid",512,64),("test",512,256))):
        available=len(streams[split])-minimum
        if available<count:raise ValueError("source split too short for unique endpoints")
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,code,0x41524c4d]))
        endpoints[split]=np.sort(rng.choice(available,count,replace=False)+minimum).astype(np.int64)
    specifications=(("train","train",64),("valid","valid",64),("test64","test",64),("test128","test",128),("test256","test",256))
    entries={};numeric_bytes=0
    for split,origin,length in specifications:
        check();ids=streams[origin];pos=endpoints[origin]
        tokens=np.stack([ids[int(end)-length:int(end)] for end in pos]).astype(np.int64)
        targets=np.stack([ids[int(end)-length+1:int(end)+1] for end in pos]).astype(np.int64)
        if not np.array_equal(tokens[:,1:],targets[:,:-1]):raise RuntimeError("shifted-label alignment failed")
        if np.any(pos-length<0) or np.any(pos>=len(ids)):raise RuntimeError("split boundary crossing")
        arrays=dict(initial=np.zeros(len(pos),dtype=np.int64),tokens=tokens,
            lengths=np.full(len(pos),length,dtype=np.int64),targets=targets)
        path=args.out/f"{split}.npz"
        with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
        endpoint_path=args.out/f"{split}-endpoints.npy"
        with endpoint_path.open("xb") as stream:np.save(stream,pos,allow_pickle=False)
        entry=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,endpoint_file=endpoint_path.name,
            endpoint_sha256=sha(endpoint_path),rows=len(pos),context_length=length,source_split=origin,
            endpoint_min=int(pos.min()),endpoint_max=int(pos.max()),array_bytes=sum(a.nbytes for a in arrays.values()),
            target_token_observations=int(targets.size),unknown_input_tokens=int((tokens==0).sum()),unknown_target_tokens=int((targets==0).sum()))
        entries[split]=entry;numeric_bytes+=entry["array_bytes"]
    # Check the promised comparable suffix labels before freezing metadata.
    paired=[]
    for split in ("test64","test128","test256"):
        with np.load(args.out/entries[split]["file"],allow_pickle=False) as data:paired.append(data["targets"][:,-64:].copy())
    if not all(np.array_equal(paired[0],x) for x in paired[1:]):raise RuntimeError("paired suffix labels changed")
    (args.out/"vocab.json").write_bytes((args.prepared/"vocab.json").read_bytes())
    metadata=dict(schema="char-alltoken-ar-development-v1",development_only=True,formal_Primary=False,seed=args.seed,
        n_states=v,n_tokens=v,max_length=256,train_length=64,files=entries,
        source_prepared=str(args.prepared.resolve()),source_manifest_sha256=sha(args.prepared/"manifest.json"),
        source_files_sha256=receipts,raw_corpus_sha256=source["source"]["sha256"],
        source_split_policy=source["split_policy"],vocabulary_policy="same original TRAIN-only vocabulary/UNK0; no refitting",
        alignment="post-token output j predicts next char: input ids[p-L:p], targets ids[p-L+1:p+1]; causal model required",
        window_sampling="unique endpoints, possible overlapping windows; observations charged even when corpus tokens repeat",
        test_pairing="same512TEST endpoints atL64/128/256; last64 target arrays byte-identical",
        all_token_metric="NLL over ALL output positions in sampled windows, not corpus-wide or word PPL",
        paired_suffix_metric="NLL over same last64 labels/endpoints at each context; dependent paired observations",
        training_budget=dict(updates=300,batch_size=32,labels_per_window=64,total_label_observations=614400),
        source_sha256=sha(Path(__file__)),numpy_version=np.__version__,uncompressed_array_bytes=numeric_bytes,
        preparation_wall_seconds=time.perf_counter()-started)
    with (args.out/"metadata.json").open("x") as stream:json.dump(metadata,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(status="prepared_alltoken_AR_development",out=str(args.out),vocabulary=v,entries={k:r["target_token_observations"] for k,r in entries.items()})))


if __name__=="__main__":main()
