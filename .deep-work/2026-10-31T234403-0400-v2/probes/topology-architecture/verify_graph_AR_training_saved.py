"""Stdlib numeric check of saved Graph AR currentNN gradients and trajectory."""
import argparse,json,math
from pathlib import Path
from verify_Pmatched_AR_saved import ROOT,npz,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('new receipt only')
 new=ROOT/'runs/parameter-char-alltoken-factorized-graph-300-v1/learn';old=ROOT/'runs/parameter-char-alltoken-factorized-300-v1/learn';s=json.loads((new/'summary.json').read_text());previous=json.loads((old/'summary.json').read_text());g=s['graph_audit'];pairs=[g['admission']]+g['diagnostics'];assert[d['step']for d in g['diagnostics']]==[100,200,300];scalars=0;maximum_output=maximum_gradient=0.;argmax_differences=0
 for pair in pairs:
  f=new/pair['raw_file'];assert sha(f)==pair['raw_sha256'];z=npz(f);names=json.loads(bytes(z['gradient_names_json'][1]).decode());assert len(names)==27;assert z['reference_output'][0]==(32,64,100)
  for left,right,atol,rtol in([('reference_output','candidate_output',2e-5,2e-5),('reference_loss','candidate_loss',2e-5,2e-5)]+[(f'reference_grad_{i:02d}',f'candidate_grad_{i:02d}',2e-5,2e-4)for i in range(27)]):
   l,r=z[left],z[right];assert l[0]==r[0];assert all(math.isfinite(v)and math.isfinite(w)and abs(v-w)<=atol+rtol*abs(w)for v,w in zip(l[1],r[1]));maximum=max(abs(v-w)for v,w in zip(l[1],r[1]));scalars+=len(l[1])
   if left.startswith('reference_grad'):maximum_gradient=max(maximum_gradient,maximum)
   elif left=='reference_output':maximum_output=max(maximum_output,maximum)
  l,r=z['reference_output'][1],z['candidate_output'][1]
  for i in range(2048):
   argmax_differences+=max(range(100),key=lambda j:l[i*100+j])!=max(range(100),key=lambda j:r[i*100+j])
 updates={}
 for name,path in(('graph',new),('eager',old)):
  rows=[json.loads(x)for x in(path/'records.jsonl').read_text().splitlines()];up=[r for r in rows if r['kind']=='actual_alltoken_update'];assert[r['step']for r in up]==list(range(1,301));updates[name]=up
 delta=[x['all_token_train_NLL']-y['all_token_train_NLL']for x,y in zip(updates['graph'],updates['eager'])];absolute=sorted(map(abs,delta))
 def q(p):
  x=(len(absolute)-1)*p;i=int(x);j=min(i+1,len(absolute)-1);return absolute[i]+(absolute[j]-absolute[i])*(x-i)
 out=dict(schema='actual-AR-Graph300-gradients-trajectory-v1',no_NN_or_SSH_execution=True,source_sha256=s['source_sha256'],summary_sha256=sha(new/'summary.json'),original_summary_sha256=sha(old/'summary.json'),all27_same_current_NN_checks=dict(raw_pairs=4,steps=[0,100,200,300],checked_scalars=scalars,max_output_abs=maximum_output,max_gradient_abs=maximum_gradient,argmax_differences=argmax_differences,atol_output=2e-5,rtol_output=2e-5,atol_gradient=2e-5,rtol_gradient=2e-4),trajectory=dict(required_bitwise=False,NLL_max_abs=max(absolute),NLL_RMSE=math.sqrt(math.fsum(v*v for v in delta)/300),NLL_p50_abs=q(.5),NLL_p95_abs=q(.95),NLL_first_difference=delta[0],NLL_final_difference=delta[-1],NLL_exact_equal_steps=sum(v==0 for v in delta),per_step=[dict(step=i+1,NLL_graph=updates['graph'][i]['all_token_train_NLL'],NLL_eager=updates['eager'][i]['all_token_train_NLL'],difference=d)for i,d in enumerate(delta)],scope='same real-valued rule/sample/freshseed initialNN; report actual FP32/Adam trajectory differences, no bitwise or score gate'),costs=dict(graph_training=s['costs'],eager_training=previous['costs'],graph_audit=g,graph_driver_wall_seconds=s['driver_wall_seconds'],graph_driver_CPU_seconds=s['driver_CPU_seconds'],eager_driver_wall_seconds=previous['driver_wall_seconds'],observed_update_loop_ratio=previous['costs']['training_wall_seconds']/s['costs']['training_wall_seconds'],observed_driver_wall_ratio=previous['driver_wall_seconds']/s['driver_wall_seconds'],cost_scope='311actualF/B,22528extraTRAINlabelpositions,1capturedescription0GPUFB; nestedcopy/train/event times not additive; peak snapshots combinebefore/afterreset; no fullFLOPs/busytime/provider-effective claim'),full_training_FLOPs=None,provider_effective_compute=None)
 with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),scalars=scalars,max_output_abs=maximum_output,max_grad_abs=maximum_gradient,NLL_RMSE=out['trajectory']['NLL_RMSE'],NLL_max_abs=max(absolute),update_loop_ratio=out['costs']['observed_update_loop_ratio'])))
if __name__=='__main__':main()
