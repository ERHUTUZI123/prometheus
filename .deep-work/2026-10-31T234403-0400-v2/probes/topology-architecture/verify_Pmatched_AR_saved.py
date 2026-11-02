"""Stdlib-only read of actual fixed-P AR300 source/raw/cost; no NN or SSH."""
from pathlib import Path
import argparse,array,ast,hashlib,json,math,sys,zipfile
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def npz(p):
 out={}
 with zipfile.ZipFile(p)as z:
  for leaf in z.namelist():
   raw=z.read(leaf);assert raw[:6]==b'\x93NUMPY';off=10 if raw[6]==1 else 12;n=int.from_bytes(raw[8:off],'little');h=ast.literal_eval(raw[off:off+n].decode());assert not h['fortran_order'];code={'<i8':'q','<f8':'d','<f4':'f','|u1':'B'}[h['descr']];a=array.array(code);a.frombytes(raw[off+n:]);assert len(a)==math.prod(h['shape']);out[leaf.removesuffix('.npy')]=(h['shape'],a,hashlib.sha256(raw[off+n:]).hexdigest())
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--include-graph',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('new evidence artifact only')
 tags=['factorized','gru34','transformer24']+(['factorized-graph']if a.include_graph else[]);report=dict(schema='actual-Pmatched-AR300-raw-cost-v1',no_NN_execution=True,no_SSH=True,arms={});common=None
 for tag in tags:
  name=f'parameter-char-alltoken-{tag}-300-v1';folder=ROOT/'runs'/name/'learn';s=json.loads((folder/'summary.json').read_text());rows=[json.loads(x)for x in(folder/'records.jsonl').read_text().splitlines()];updates=[r for r in rows if r['kind']=='actual_alltoken_update'];assert[r['step']for r in updates]==list(range(1,301));last=updates[-1]['cumulative_costs']
  for key in('optimizer_updates','training_wall_seconds','training_CPU_seconds','training_forward_MAC_proxy','training_label_observations'):assert last[key]==s['costs'][key]
  assert last['optimizer_updates']==300 and last['training_label_observations']==614400
  events=[r['kind']for r in rows];freeze=events.index('ALL_TRAINING_FROZEN_BEFORE_TEST');assert freeze>max(i for i,r in enumerate(rows)if r['kind']=='actual_alltoken_update');assert all(i>freeze for i,r in enumerate(rows)if r['kind']=='FROZEN_TEST_RESULT')
  source={}
  for leaf,digest in s['source_sha256'].items():
   path=ROOT/'versions'/('code-'+name)/leaf;assert sha(path)==digest;ast.parse(path.read_bytes());source[leaf]=digest
  batch=npz(folder/'common-batch-indices.npz')['indices'];assert batch[0]==(300,32)and batch[2]=='da55630fb2576a5c174e4215082965f226d1a37ead1b50f1a616cd6fa1d1ac98'
  outputs={};labels=[]
  for length in(64,128,256):
   split=f'test{length}';saved=s['final'][split]['soft'];file=folder/saved['vectors_file'];assert sha(file)==saved['vectors_sha256'];z=npz(file);shape,losses,_=z['nll'];assert shape==(512,length);targets=z['targets'][1];pred=z['predictions'][1];last64=[losses[i*length+j]for i in range(512)for j in range(length-64,length)];lasttargets=tuple(targets[i*length+j]for i in range(512)for j in range(length-64,length));mean=math.fsum(losses)/len(losses);paired=math.fsum(last64)/len(last64);correct=sum(q==t for q,t in zip(pred,targets))/len(targets)
   assert abs(mean-saved['all_token_NLL'])<2e-7 and abs(paired-saved['last64_paired_NLL'])<2e-7 and correct==saved['all_token_accuracy'];labels.append((tuple(z['endpoints'][1]),lasttargets));outputs[length]=dict(vectors_sha256=sha(file),target_positions=len(losses),paired_target_positions=len(last64),all_token_NLL=mean,all_token_PPL=math.exp(mean),paired_last64_NLL=paired,paired_last64_PPL=math.exp(paired),all_token_accuracy=correct,soft_forward_wall_seconds=saved['wall_seconds'])
  assert labels[0]==labels[1]==labels[2];identity=(labels,s['batch_indices_sha256'],s['metadata_sha256'],s['input_receipts'])
  if common is None:common=identity
  else:assert common==identity
  arm=dict(name=name,summary_sha256=sha(folder/'summary.json'),records_sha256=sha(folder/'records.jsonl'),source_sha256=source,model_configuration=s['model_configuration'],resources=s['resources'],costs=s['costs'],driver_wall_seconds=s['driver_wall_seconds'],checkpoints=s['checkpoints'],outputs=outputs,actual_updates=300,all_common_data_sampler_and_paired_targets=True,accuracy_gate=False)
  for receipt in s['checkpoints']:
   file=folder/receipt['file'];assert sha(file)==receipt['sha256']and file.stat().st_size==receipt['bytes']
  coordinator=ROOT/'runs'/name/'coordinator-exit.json'
  if coordinator.exists():
   x=json.loads(coordinator.read_text());arm['coordinator_file_sha256']=sha(coordinator);arm['coordinator_recorded_wall_seconds']={k:v for k,v in x.items()if 'seconds'in k}
  if tag=='factorized-graph':arm['graph_audit']=s['graph_audit'];arm['driver_CPU_seconds']=s['driver_CPU_seconds']
  report['arms'][tag]=arm
 report.update(verified_models=len(tags),verified_TEST_target_positions=len(tags)*512*(64+128+256),scope='fixedprospective near-P controls selected analytically, seed0/300samebudget; no score gate or tuning; independent rawaccuracy+NLL/PPL verified; parentdata and targets identical; no topology/novelty/fullresource advantage certificate',unmeasured=dict(full_training_FLOPs=None,provider_effective_compute=None,energy=None,GPU_busy_seconds=None))
 with a.out.open('x')as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),models=len(tags),raw_target_positions=report['verified_TEST_target_positions'],P={k:v['resources']['parameters']for k,v in report['arms'].items()})))
if __name__=='__main__':main()
