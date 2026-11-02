from pathlib import Path
import json,math
from verify_factorized_graph600_saved import npz,sha,ROOT
folder=ROOT/'runs/parameter-value-interface-two-owned-cpu-v1/check';summary=json.loads((folder/'result.json').read_text());report=dict(schema='actual-initial-value-interface-raw-verification-v1',scope='saved8x8CPUgeometry only; no Torch/model/checkpoint/transition/SSH execution; no newlabels/SGD',result_sha256=sha(folder/'result.json'),models={})
def dot(a,b):return math.fsum(v*w for v,w in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
for name,saved in summary['models'].items():
 path=folder/saved['raw_file'];assert sha(path)==saved['raw_sha256'];z=npz(path);E=z['encoded_values']['values'];W=z['readout_weight']['values'];bias=z['readout_bias']['values'];assert z['encoded_values']['shape']==z['readout_weight']['shape']==(8,32)
 prototypes=[list(E[i*32:(i+1)*32])for i in range(8)];weights=[list(W[i*32:(i+1)*32])for i in range(8)];logp=z['log_probabilities']['values'];reconstructed=[]
 def predict(v):
  logits=[dot(v,w)+bias[j]for j,w in enumerate(weights)];maximum=max(logits);normalizer=math.log(math.fsum(math.exp(t-maximum)for t in logits))+maximum;return[t-normalizer for t in logits]
 def R(v):
  probabilities=[math.exp(t)for t in predict(v)];mixed=[math.fsum(probabilities[j]*prototypes[j][i]for j in range(8))for i in range(32)];scale=norm(v)/max(norm(mixed),1e-6);return[w*scale for w in mixed]
 argmax=[];lpdiff=0.;Rdiff=R2diff=gramdiff=0.;residual=[];second=[]
 for i,e in enumerate(prototypes):
  lp=predict(e);argmax.append(max(range(8),key=lambda j:lp[j]));lpdiff=max(lpdiff,max(abs(v-logp[i*8+j])for j,v in enumerate(lp)));r=R(e);r2=R(r);residual.append(norm([v-w for v,w in zip(r,e)]));second.append(norm([v-w for v,w in zip(r2,r)]));Rdiff=max(Rdiff,max(abs(v-z['hypothetical_R_E']['values'][i*32+j])for j,v in enumerate(r)));R2diff=max(R2diff,max(abs(v-z['hypothetical_R2_E']['values'][i*32+j])for j,v in enumerate(r2)))
  for j,e2 in enumerate(prototypes):gramdiff=max(gramdiff,abs(dot(e,e2)-z['prototype_Gram']['values'][i*8+j]))
 assert argmax==list(z['decoded_argmax']['values'])==saved['decoded_argmax']==list(range(8));assert saved['same_index_argmax_count']==8
 assert max(abs(v-w)for v,w in zip(logp,[v for row in saved['all64_log_probabilities']for v in row]))==0
 report['models'][name]=dict(raw_sha256=sha(path),P=saved['P'],initial_value_argmax=argmax,initial_value_identity_count=8,normalized_prototype_norms=[norm(e)for e in prototypes],prototype_min_pair_L2=min(norm([a-b for a,b in zip(prototypes[i],prototypes[j])])for i in range(8)for j in range(i)),float64_reconstruction_vs_raw=dict(logp_max_abs=lpdiff,R_E_max_abs=Rdiff,R2_E_max_abs=R2diff,Gram_max_abs=gramdiff),R_E_residual_float64=residual,R2_minus_R_residual_float64=second,R2_minus_R_max_float64=max(second))
report['judgment']='Both actualNN alreadydecodeall8initialvaluescorrectly; initialcodecargmaxmismatch is NOTobserved. Tieddecoderidentitylemma isvalidbutdoesnotrepairthisobservedfailure. Offprototype terminalvalue/fullcarryclosure is a separatemeasuredquestion. Neither nearprototypeR(E) norfinite8x8 testisfullclosure oridempotenceproof.'
out=ROOT/'actual-initial-value-interface-independent-raw-checks-v1.json'
with out.open('x')as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(out=str(out),sha256=sha(out),results={k:dict(identity=v['initial_value_identity_count'],R2_max=v['R2_minus_R_max_float64'],reconstruct=v['float64_reconstruction_vs_raw'])for k,v in report['models'].items()})))
