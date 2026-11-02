"""Independent ROOT stdlib verifier of actual six-arm saved inference records.

Group by saved condition/source/suffix fields, not producer flattening loops.
No Torch/NumPy/model execution or additional labels. Recompute every argmax,
suffix label relationship, accuracy, joint correctness and prediction consistency.
"""
import ast,hashlib,json,math,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array(raw):
    if raw[:6]!=b'\x93NUMPY':raise ValueError('NPY magic')
    major=raw[6];n=struct.unpack('<H'if major==1 else'<I',raw[8:10 if major==1 else 12])[0]
    start=10 if major==1 else 12;header=ast.literal_eval(raw[start:start+n].decode())
    if header['fortran_order']:raise ValueError('C order required')
    fmt={'<i8':'q','<f4':'f','<f8':'d'}[header['descr']];count=math.prod(header['shape'])
    return struct.unpack('<'+str(count)+fmt,raw[start+n:]),header['shape']
def transform(q,s):return(q,(q+1)%8,(3*q+2)%8,0)[s]
def main():
    rows=[]
    for arm in('dynamic','static','gru','transformer','delta','credit'):
        folder=ROOT/'runs'/('parameter-live-composition-'+arm+'-v1')/'evaluate';result=json.loads((folder/'result.json').read_text());p=folder/result['raw_outputs']['file']
        if sha(p)!=result['raw_outputs']['sha256']:raise ValueError('raw SHA mismatch')
        with zipfile.ZipFile(p)as z:values={name[:-4]:array(z.read(name))for name in z.namelist()}
        a={k:v[0]for k,v in values.items()};n=len(a['predictions'])
        if n!=2048 or values['log_probabilities'][1]!=(2048,8):raise ValueError('fixed row/logp shape')
        groups={};maxnll=0.
        for i in range(n):
            lp=a['log_probabilities'][i*8:(i+1)*8];pred=max(range(8),key=lambda k:lp[k])
            if pred!=a['predictions'][i]:raise ValueError('saved argmax differs')
            maxnll=max(maxnll,abs(-lp[a['targets'][i]]-a['terminal_NLL'][i]))
            key=(a['condition'][i],a['source_train_index'][i]);sub=groups.setdefault(key,{})
            if a['suffix'][i]in sub:raise ValueError('duplicate base/suffix')
            sub[a['suffix'][i]]=i
        checks=[]
        for condition,name in enumerate(('MIX15','READ10_control')):
            selected=[sub for (c,source),sub in groups.items()if c==condition]
            if len(selected)!=256 or any(set(sub)!={0,1,2,3}for sub in selected):raise ValueError('complete256grid per condition')
            for suffix,sname in enumerate(('empty','SET0_STEP','SET1_STEP','CLEAR')):
                correct=joint=consistent=basecorrect=0
                for sub in selected:
                    b,j=sub[0],sub[suffix];target=transform(a['targets'][b],suffix)
                    if a['targets'][j]!=target:raise ValueError('terminal suffix label violates supplied task semantics')
                    bc=a['predictions'][b]==a['targets'][b];ok=a['predictions'][j]==a['targets'][j]
                    basecorrect+=bc;correct+=ok;joint+=bc and ok;consistent+=a['predictions'][j]==transform(a['predictions'][b],suffix)
                expected=result['conditions'][name]['suffixes'][sname]
                for key,count in [('suffix_correct_count',correct),('joint_base_and_suffix_correct_count',joint),('consistency_count',consistent),('conditional_denominator_base_correct',basecorrect)]:
                    if expected[key]!=count:raise ValueError('reported count mismatch: '+arm+'/'+name+'/'+sname+'/'+key)
                checks.append(dict(condition=name,suffix=sname,correct=correct,joint=joint,consistent=consistent,denominator=256,own_basecorrect=basecorrect))
        if maxnll>1e-6:raise ValueError('raw NLL mismatch')
        rows.append(dict(model=arm,raw_sha256=sha(p),all2048_argmaxes_checked=True,max_NLL_difference=maxnll,group_based_checks=checks))
    out=ROOT/'live-composition-raw-independent-verification-v1.json'
    with out.open('x')as f:json.dump(dict(status='ALL_SIX_RAW_GROUP_BASED_CHECKS_PASS',no_model_execution=True,no_new_labels=True,rows=rows),f,indent=2,sort_keys=True)
    print(json.dumps(dict(status='ALL_SIX_RAW_GROUP_BASED_CHECKS_PASS',arms=6,actual_rows=12288,artifact=str(out))))
if __name__=='__main__':main()
