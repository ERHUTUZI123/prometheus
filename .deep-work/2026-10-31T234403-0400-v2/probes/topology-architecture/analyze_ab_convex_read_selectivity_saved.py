"""Literal-write selectivity bounds in saved normalized-memory histories.

No NN/checkpoint/optimizer/SSH execution. Bounds concern the two actual WRITE
history vectors, not all information or what a nonlinear decoder can compute.
"""
import hashlib,json,time
from pathlib import Path
from verify_factorized_graph600_saved import npz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'ab-convex-read-selectivity-saved-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(rows):
    fields=('requested_max','requested_min','requested_minus_other_max','actual_requested','actual_other','coverage_reconstruction_error')
    out={'rows':len(rows)}
    for field in fields:
        v=sorted(r[field] for r in rows)
        out[field]={'mean':sum(v)/len(v),'min':v[0],'median':(v[(len(v)-1)//2]+v[len(v)//2])/2,'max':v[-1]}
    out['max_requested_at_least_0_9']=sum(r['requested_max']>=.9 for r in rows)
    out['max_requested_at_least_0_75']=sum(r['requested_max']>=.75 for r in rows)
    out['no_slot_favors_requested_over_other']=sum(r['requested_minus_other_max']<=0 for r in rows)
    return out
def main():
    if OUT.exists():raise FileExistsError('preserve finite analysis artifact')
    tick=time.perf_counter();cpu=time.process_time()
    publicpath=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz';pub=npz(publicpath)
    assert sha(publicpath)=='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
    paths={'reference_soft':ROOT/'runs/parameter-ab-write-direction-controls-cpu-v1/audit/reference-raw-before-targets.npz',
           'held_soft':ROOT/'runs/parameter-ab-reference-lifetime-cpu-v1/audit/held_soft-raw-before-targets.npz',
           'held_native':ROOT/'runs/parameter-ab-reference-lifetime-cpu-v1/audit/held_native-raw-before-targets.npz'}
    models={}
    for name,path in paths.items():
        z=npz(path);assert z['w']['shape'][0:1]==(512,) and z['w']['shape'][2]==8
        L=z['w']['shape'][1];w=z['w']['values'];cov=z['coverage']['values'];rows=[]
        for row,length in enumerate(pub['lengths']['values']):
            word=pub['tokens']['values'][row*128:row*128+length];writes=[i for i,x in enumerate(word)if x==9];assert len(writes)==2
            read=length-3;assert word[read]==10
            first=word[0];query=word[length-4];assert first in (7,8)and query in (7,8)
            requested=0 if first==query else 1
            coverage=[0.]*8;contrib=[[0.]*8,[0.]*8]
            for t in range(read):
                wt=w[(row*L+t)*8:(row*L+t+1)*8]
                for j in range(8):
                    coverage[j]+=(1-coverage[j])*wt[j]
                    for k in (0,1):contrib[k][j]=(1-wt[j])*contrib[k][j]+(wt[j]if t==writes[k]else 0.)
            coeff=[[contrib[k][j]/max(coverage[j],1e-4)for j in range(8)]for k in (0,1)]
            if 'read_attention'in z:a=z['read_attention']['values'][(row*L+read)*8:(row*L+read+1)*8]
            else:
                wt=w[(row*L+read)*8:(row*L+read+1)*8];assert sum(wt)>0;a=[x/sum(wt)for x in wt]
            assert abs(sum(a)-1)<2e-6 and all(0<=x<=1+2e-6 for x in a)
            saved=cov[(row*L+read-1)*8:(row*L+read)*8]
            r=dict(row=row,requested_write=requested+1,requested_max=max(coeff[requested]),requested_min=min(coeff[requested]),requested_minus_other_max=max(coeff[requested][j]-coeff[1-requested][j]for j in range(8)),actual_requested=sum(a[j]*coeff[requested][j]for j in range(8)),actual_other=sum(a[j]*coeff[1-requested][j]for j in range(8)),coverage_reconstruction_error=max(abs(x-y)for x,y in zip(saved,coverage)))
            assert r['actual_requested']<=r['requested_max']+2e-6
            rows.append(r)
        models[name]=dict(raw_path=str(path),raw_sha256=sha(path),all=stats(rows),first_write_requested=stats([r for r in rows if r['requested_write']==1]),second_write_requested=stats([r for r in rows if r['requested_write']==2]),per_row=rows)
    result=dict(status='SAVED_HISTORY_CONVEX_READ_BOUND_RECONSTRUCTION',source_sha256=sha(Path(__file__)),public_data_sha256=sha(publicpath),models=models,
        rule='p_ij=lambda_write_i,j/max(z_j,1e-4); read=sum_i(sum_j a_j*p_ij)*h_write_i + otherhistory; a in simplex implies requested coefficient <= max_j p_requested,j',
        boundary='Conditional on saved write/coverage history and literal WRITE vectors. Nonlinear h/controller/readout can use mixtures or other information; not an impossibility theorem for task or neural architecture. Original soft attention recovered from positive w/sum(w); held traces provide attention directly. No targets used, no new NN, no training.',
        NN_executions=0,SGD_updates=0,wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,full_FLOPs=None)
    with OUT.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'out':str(OUT),'sha256':sha(OUT),'models':{k:{g:v[g]for g in ('all','first_write_requested','second_write_requested')}for k,v in models.items()},'seconds':result['wall_seconds']}))
if __name__=='__main__':main()
