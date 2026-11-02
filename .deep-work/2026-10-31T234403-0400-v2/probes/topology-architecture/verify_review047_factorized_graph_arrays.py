from pathlib import Path
import ast,array,hashlib,json,math,struct,sys,zipfile
P=Path('/Users/xiaoyangliu/Desktop/prometheus/.deep-work/2026-10-31T234403-0400-v2/probes/topology-architecture')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def npz(path):
 out={}
 with zipfile.ZipFile(path)as z:
  for leaf in z.namelist():
   b=z.read(leaf);assert b[:6]==b'\x93NUMPY';v=b[6];n=int.from_bytes(b[8:10]if v==1 else b[8:12],'little');s=10 if v==1 else 12;h=ast.literal_eval(b[s:s+n].decode());assert not h['fortran_order'];d=h['descr'];typ={'<f4':'f','<f8':'d','<i8':'q','<i4':'i','|u1':'B','|b1':'B'}[d];a=array.array(typ);a.frombytes(b[s+n:]);assert len(a)==math.prod(h['shape']);out[leaf[:-4]]=(h['shape'],a,d)
 return out
result=dict(scope='saved CPU/GPU F/B pairs only; stdlib numeric verification, no model execution',cases={})
for mode,sub in(('cpu','contract'),('gpu','audit')):
 name='parameter-factorized-graph-fb-'+('cpu-admission-v1'if mode=='cpu'else'gpu-audit-v1');folder=P/'runs'/name/sub;x=json.loads((folder/'summary.json').read_text());assert x['completed']and x['NN_unchanged']and x['optimizer_steps']==0 and x['SGD_updates']==0
 for row in x['cases']:
  out={};canonical={};gradmax=outputmax=0;count=0
  for field,receipt in row.items():
   if not field.endswith('_raw'):continue
   path=folder/receipt['file'];assert sha(path)==receipt['sha256'];z=npz(path);assert z['reference_output'][0]==(64,8);gradient_names=json.loads(bytes(z['gradient_names_json'][1]).decode());assert len(gradient_names)==27
   pairs=[('reference_output','candidate_output',2e-5,2e-5),('reference_loss','candidate_loss',2e-5,2e-5)]+[(f'reference_grad_{i:02d}',f'candidate_grad_{i:02d}',2e-5,2e-4)for i in range(27)]
   for left,right,atol,rtol in pairs:
    assert z[left][0]==z[right][0];a,b=z[left][1],z[right][1];assert len(a)==len(b);maximum=max(abs(v-w)for v,w in zip(a,b))
    assert all(math.isfinite(v)and math.isfinite(w)and abs(v-w)<=atol+rtol*abs(w)for v,w in zip(a,b));count+=len(a)
    if left.startswith('reference_grad'):gradmax=max(gradmax,maximum)
    elif left=='reference_output':outputmax=max(outputmax,maximum)
    if left not in canonical:canonical[left]=a.tobytes()
    else:assert canonical[left]==a.tobytes()
   out[field]=dict(sha256=sha(path),all27gradients_present=True,allclose=True)
  def percentile(v):v=sorted(v);return(v[5]+v[6])/2
  for backend,m in row['measurements'].items():
   assert [d['repeat']for d in m['draws']]==list(range(12))
   for channel in('wall_seconds','CUDA_stream_seconds'):assert abs(percentile([d[channel]for d in m['draws']])-m['p50_'+channel])<1e-14
  result['cases'][f'{mode}-B64-L{row["bucket"]}']=dict(raw=out,compared_scalars=count,output_max_abs=outputmax,all27_gradient_max_abs=gradmax,active=row['active_transition_steps'],padded=row['padded_transition_steps'],padding_ratio=row['padded_to_active_ratio'],original_forward_MAC_proxy=row['original_forward_MAC_proxy'],masked_forward_MAC_proxy=row['masked_forward_MAC_proxy'],measurements=row['measurements'],capture=row.get('capture'))
 result[mode]=dict(summary_sha256=sha(folder/'summary.json'),wall_seconds=x['wall_seconds'],CPU_seconds=x['cpu_seconds'],actual_FB_calls=x['actual_FB_calls'],source_sha256=x['source_sha256'],P=x['P'],NN_bytes=x['NN_bytes'],saved_Adam_bytes=x['saved_Adam_tensor_bytes'],live_Adam_bytes=x['live_Adam_tensor_bytes'])
out=P/'review047-factorized-graph-raw-checks.json'
with out.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(str(out),sha(out));print('max',(max(v['output_max_abs']for v in result['cases'].values()),max(v['all27_gradient_max_abs']for v in result['cases'].values())))
