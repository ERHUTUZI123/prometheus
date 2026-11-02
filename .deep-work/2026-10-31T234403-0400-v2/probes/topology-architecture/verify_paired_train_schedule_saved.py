"""Independent stdlib reconstruction from saved arrays; no label decode/NN."""
import array,ast,hashlib,json,sys,time,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def npz(p,members=None):
 out={}
 with zipfile.ZipFile(p)as z:
  for name in z.namelist()if members is None else members:
   b=z.read(name);v=b[6];off=10 if v==1 else 12;n=int.from_bytes(b[8:10]if v==1 else b[8:12],'little');h=ast.literal_eval(b[off:off+n].decode());assert h['descr']=='<i8'and not h['fortran_order'];a=array.array('q');a.frombytes(b[off+n:]);out[name[:-4]]=(h['shape'],a)
 return out

def main():
 start=time.perf_counter();cpu=time.process_time();root=P/'runs/parameter-paired-train-indices-n8-stage3-seed0-v1/generate';r=json.loads((root/'result.json').read_text());schedule=root/r['schedule']['file'];assert sha(schedule)==r['schedule']['sha256'];d=npz(schedule);assert set(d)=={'groups','pairdraws','pairorders','indices','indices_same_pool_shuffled'}
 for k,(shape,a)in d.items():assert list(shape)==r['schedule']['arrays'][k]['shape']and hashlib.sha256(a.tobytes()).hexdigest()==r['schedule']['arrays'][k]['sha256_C_payload']
 train=P/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-train.npz';assert sha(train)==r['input_sha256'];raw=npz(train,['initial.npy','tokens.npy','lengths.npy']);by={}
 for row in range(4096):
  length=raw['lengths'][1][row];key=(raw['initial'][1][row],tuple(raw['tokens'][1][row*128:row*128+length-4]));by.setdefault(key,[]).append(row)
 groups=list(by.values());assert len(groups)==2048 and all(len(x)==2 for x in groups);assert sorted(x for g in groups for x in g)==list(range(4096));assert list(d['groups'][1])==[x for g in groups for x in g]
 for l,r0 in groups:
  ll=raw['lengths'][1][l];rr=raw['lengths'][1][r0];assert ll==rr and list(raw['tokens'][1][l*128+ll-4:l*128+ll])!=list(raw['tokens'][1][r0*128+rr-4:r0*128+rr])
 idx=d['indices'][1];control=d['indices_same_pool_shuffled'][1];hits=[0]*4096;chits=[0]*4096;ghits=[0]*2048
 for step in range(600):
  for pair in range(24):
   pos=step*24+pair;g=d['pairdraws'][1][pos];order=d['pairorders'][1][pos];assert 0<=g<2048 and order in(0,1);expected=groups[g][::(-1 if order else 1)];assert list(idx[step*64+2*pair:step*64+2*pair+2])==expected;ghits[g]+=1
  for k in range(48):hits[idx[step*64+k]]+=1;chits[control[step*64+k]]+=1
 assert hits==chits and sum(hits)==28800 and all(x>0 for x in hits);assert all(hits[g[0]]==hits[g[1]]==ghits[i]for i,g in enumerate(groups));exposure=array.array('q',hits);assert hashlib.sha256(exposure.tobytes()).hexdigest()==r['verification']['same_pool_exposure_vector_sha256']
 old=P/'runs/parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2/learn/stage3-common-indices.npz';original=npz(old)['indices'][1];assert len(original)==600*64
 for s in range(600):assert list(idx[s*64+48:s*64+64])==list(control[s*64+48:s*64+64])==list(original[s*64+48:s*64+64])
 source=P/'versions/code-parameter-paired-train-indices-n8-stage3-seed0-v1/generate_paired_train_indices.py';assert sha(source)==r['source_sha256'][source.name];tree=ast.parse(source.read_text());members=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='RAW_MEMBERS'for t in n.targets));assert members==('initial.npy','tokens.npy','lengths.npy')
 out=dict(status='INDEPENDENT_SAVED_PAIR_AND_SAMEPOOL_RECONSTRUCTION_PASS',source_script_sha256=sha(Path(__file__)),schedule_sha256=sha(schedule),schedule_bytes=schedule.stat().st_size,array_payload_bytes=sum(len(v[1])*8 for v in d.values()),raw_TRAIN_sha256=sha(train),original_indices_sha256=sha(old),generator_source_sha256=sha(source),result_sha256=sha(root/'result.json'),groups2048_exact_stable_raw_prefix_partition=True,all4096_rows_once_in_partition=True,all14400_pairdraw_order_reconstructions=True,all28800_current_ids_same_pool_counts_exact=True,all4096_rows_sampled=True,group_draw_min=min(ghits),group_draw_max=max(ghits),unused_tail16_original_exact=True,exposure_vector_sha256=hashlib.sha256(exposure.tobytes()).hexdigest(),target_members_decoded_or_indexed=0,NN_queries=0,SGD_updates=0,analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu,limits='stdlib savedarray algebra/sourcepinnings; noindependentPCG64 regeneration/NNrun. Generator and verifier decodeonly3rawinputmembers; wholefileSHAincludescompressedtargetbytes. Paired/controlchangesbatchcovariance ANDtemporalorder, notcovariance-only causalproof')
 f=P/'paired-train-schedule-independent-raw-checks-v1.json'
 with f.open('x')as h:json.dump(out,h,sort_keys=True,indent=2);h.write('\n')
 print(f.name,sha(f),out)
if __name__=='__main__':main()
