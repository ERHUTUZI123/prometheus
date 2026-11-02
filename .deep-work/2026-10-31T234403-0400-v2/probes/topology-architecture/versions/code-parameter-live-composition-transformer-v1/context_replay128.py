"""Bounded terminal TRAIN replay; NumPy/stdlib only, no model/oracle imports.

Reservoir sampling is over ACTUAL committed current-example exposures, including
repeated examples. Capacity counts raw rows, so duplicate source rows consume
capacity and their count is disclosed. Buffers change only at stage boundaries.
Every stored row contains source stage/index/global exposure ordinal. Local RNGs
do not modify training's Python, NumPy-global or Torch RNG streams.
"""
import hashlib
import json
import time
import numpy as np

RAW=("initial","tokens","lengths","targets")
PROVENANCE=("source_stage","source_index","stream_ordinal")


class Replay128:
    def __init__(self,capacity,width,seed,states):
        if not 1<=capacity<=128 or width<1:raise ValueError("bounded nonempty replay required")
        self.capacity,self.width,self.seed,self.states=capacity,width,seed,states
        self.size=0;self.seen=0;self.replacements=0;self.maintenance_seconds=0.;self.draws=0
        self.arrays={key:np.zeros((capacity,width) if key=="tokens" else capacity,dtype=np.int64)
                     for key in RAW+PROVENANCE}
        self.local_rng_states={}

    def digest(self):
        h=hashlib.sha256()
        for key in RAW+PROVENANCE:
            h.update(key.encode());h.update(self.arrays[key][:self.size].tobytes())
        h.update(np.asarray([self.capacity,self.width,self.size,self.seen],dtype=np.int64).tobytes())
        return h.hexdigest()

    def consume(self,data,used_indices,stage,emit=None):
        """Caller passes only indices of successfully completed updates."""
        ids=np.asarray(used_indices)
        if ids.dtype!=np.int64 or ids.ndim!=2 or np.any((ids<0)|(ids>=len(data["initial"]))):
            raise ValueError("actual used TRAIN index matrix required")
        if data["tokens"].shape[1]>self.width:raise ValueError("raw width exceeds replay extent")
        started=time.perf_counter();before=self.digest();previous_seen=self.seen;writes=0
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.states,stage,0x52535652]))
        for source_index in ids.flat:
            ordinal=self.seen;self.seen+=1
            slot=self.size if self.size<self.capacity else int(rng.integers(0,self.seen))
            if slot>=self.capacity:continue
            replaced=slot<self.size;self.size=min(self.capacity,self.size+1)
            row=int(source_index)
            for key in ("initial","lengths","targets"):self.arrays[key][slot]=data[key][row]
            self.arrays["tokens"][slot].fill(0)
            self.arrays["tokens"][slot,:data["tokens"].shape[1]]=data["tokens"][row]
            for key,value in (("source_stage",stage),("source_index",row),("stream_ordinal",ordinal)):
                self.arrays[key][slot]=value
            writes+=1;self.replacements+=int(replaced)
            if emit is not None:emit("replay_reservoir_selection",source_stage=stage,source_index=row,
                stream_ordinal=ordinal,buffer_slot=slot,replaced_existing=replaced)
        self.local_rng_states[str(stage)]=rng.bit_generator.state
        elapsed=time.perf_counter()-started;self.maintenance_seconds+=elapsed
        return dict(previous_seen=previous_seen,new_committed_current_exposures=ids.size,
            seen=self.seen,actual_buffer_writes=writes,before_sha256=before,after_sha256=self.digest(),
            discarded_candidate_exposures=int(ids.size)-writes,
            reservoir_random_draws=max(0,self.seen-max(previous_seen,self.capacity)),
            maintenance_wall_seconds=elapsed,**self.statistics())

    def sample_indices(self,stage,steps,per_batch):
        if self.size<1 or not 1<=per_batch<64:raise ValueError("nonempty old buffer and bounded draw count required")
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.states,stage,0x52504c59]))
        ids=rng.integers(0,self.size,size=(steps,per_batch),dtype=np.int64)
        self.local_rng_states["draws-"+str(stage)]=rng.bit_generator.state
        return ids

    def mix(self,current,current_indices,replay_indices):
        if len(current_indices)+len(replay_indices)!=64:raise ValueError("exact64-row feedback budget required")
        if np.any((replay_indices<0)|(replay_indices>=self.size)):raise ValueError("replay index outside stored buffer")
        width=max(current["tokens"].shape[1],self.width)
        data={key:np.empty((64,width) if key=="tokens" else 64,dtype=np.int64) for key in RAW}
        n=len(current_indices)
        for key in ("initial","lengths","targets"):
            data[key][:n]=current[key][current_indices];data[key][n:]=self.arrays[key][replay_indices]
        data["tokens"].fill(0)
        data["tokens"][:n,:current["tokens"].shape[1]]=current["tokens"][current_indices]
        data["tokens"][n:,:self.width]=self.arrays["tokens"][replay_indices]
        self.draws+=len(replay_indices)
        return data

    def validate_sources(self,train_sources,used_sources):
        """Check buffer bytes and ordinal against real TRAIN exposure evidence."""
        offsets={};offset=0
        for stage in sorted(used_sources):offsets[stage]=offset;offset+=used_sources[stage].size
        if offset!=self.seen:raise AssertionError("reservoir count differs from paid TRAIN exposures")
        for slot in range(self.size):
            stage=int(self.arrays["source_stage"][slot]);row=int(self.arrays["source_index"][slot])
            if stage not in train_sources or stage not in used_sources:raise AssertionError("non-TRAIN/unknown provenance")
            pos=int(self.arrays["stream_ordinal"][slot])-offsets[stage]
            flat=used_sources[stage].reshape(-1)
            if not 0<=pos<len(flat) or int(flat[pos])!=row:raise AssertionError("buffer row was not actually used")
            source=train_sources[stage]
            for key in ("initial","lengths","targets"):
                if self.arrays[key][slot]!=source[key][row]:raise AssertionError("stored TRAIN row differs")
            width=source["tokens"].shape[1]
            if not np.array_equal(self.arrays["tokens"][slot,:width],source["tokens"][row]):
                raise AssertionError("stored TRAIN program differs")
            if np.any(self.arrays["tokens"][slot,width:]):raise AssertionError("invalid added padding")
        return dict(actual_used_TRAIN_only=True,checked_buffer_rows=self.size,used_exposures=self.seen)

    def statistics(self):
        origins=self.arrays["source_stage"][:self.size]
        unique={(int(s),int(i)) for s,i in zip(origins,self.arrays["source_index"][:self.size])}
        return dict(capacity=self.capacity,stored_rows=self.size,distinct_source_rows=len(unique),
            duplicate_source_rows=self.size-len(unique),buffer_array_bytes=sum(v.nbytes for v in self.arrays.values()),
            stored_by_stage={str(int(s)):int((origins==s).sum()) for s in np.unique(origins)},
            local_RNG_state=self.local_rng_states,bookkeeping_counters={"seen":self.seen,"draws":self.draws,
                "replacements":self.replacements},maintenance_wall_seconds_cumulative=self.maintenance_seconds,
            reservoir_unit="committed current-example exposure; repeated source rows are separate stream items")

    def state_dict(self):
        return dict(schema="terminal-replay128-v1",capacity=self.capacity,width=self.width,seed=self.seed,
            states=self.states,size=self.size,seen=self.seen,draws=self.draws,replacements=self.replacements,
            maintenance_seconds=self.maintenance_seconds,local_rng_states=self.local_rng_states,
            arrays={k:v.copy() for k,v in self.arrays.items()},sha256=self.digest())

    @classmethod
    def from_state_dict(cls,state):
        if state.get("schema")!="terminal-replay128-v1":raise ValueError("unknown replay state schema")
        replay=cls(state["capacity"],state["width"],state["seed"],state["states"])
        if set(state["arrays"])!=set(replay.arrays):raise ValueError("replay fields differ")
        for key,array in state["arrays"].items():
            if array.dtype!=np.int64 or array.shape!=replay.arrays[key].shape:raise ValueError("replay array shape/dtype differs")
            replay.arrays[key]=array.copy()
        for key in ("size","seen","draws","replacements"):
            setattr(replay,key,state[key])
        replay.maintenance_seconds=state["maintenance_seconds"];replay.local_rng_states=state["local_rng_states"]
        if not 0<=replay.size<=replay.capacity or replay.digest()!=state["sha256"]:raise ValueError("replay state content hash differs")
        return replay


def self_test():
    """Small deterministic protocol test; no labels from research data or NN."""
    before=np.random.get_state()
    source=dict(initial=np.arange(9,dtype=np.int64),tokens=np.arange(27,dtype=np.int64).reshape(9,3),
        lengths=np.full(9,3,dtype=np.int64),targets=np.arange(9,dtype=np.int64))
    first=np.asarray([[1,4,7,1]],dtype=np.int64);second=np.asarray([[3,5,8,3]],dtype=np.int64)
    a=Replay128(3,3,17,9);b=Replay128(3,3,17,9)
    for replay in (a,b):replay.consume(source,first,0);replay.consume(source,second,1)
    assert a.digest()==b.digest() and a.size==3 and a.seen==8
    audit=a.validate_sources({0:source,1:source},{0:first,1:second})
    ids=a.sample_indices(2,2,16);assert ids.shape==(2,16)
    mixed=a.mix(source,np.zeros(48,dtype=np.int64),ids[0]);assert len(mixed["targets"])==64
    saved=a.arrays["targets"][0];a.arrays["targets"][0]+=1
    try:a.validate_sources({0:source,1:source},{0:first,1:second})
    except AssertionError:pass
    else:raise AssertionError("tampered terminal label not rejected")
    a.arrays["targets"][0]=saved
    restored=Replay128.from_state_dict(a.state_dict());assert restored.digest()==a.digest()
    try:a.consume(source,np.asarray([[99]],dtype=np.int64),2)
    except ValueError:pass
    else:raise AssertionError("unobserved TRAIN row not rejected")
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    return dict(protocol_self_test_passed=True,deterministic_local_RNG=True,capacity_enforced=True,
        exact64_feedback_rows=True,tampered_labels_rejected=True,unused_indices_rejected=True,
        global_NumPy_RNG_unchanged=True,actual_array_state_roundtrip=True,**audit)


if __name__=="__main__":print(json.dumps(self_test(),sort_keys=True))
