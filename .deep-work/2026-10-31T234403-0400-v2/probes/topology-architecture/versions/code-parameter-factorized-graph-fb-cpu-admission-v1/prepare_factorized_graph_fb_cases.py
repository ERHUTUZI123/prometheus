"""NO-RUN deterministic preparation from existing TRAIN data only, no resampling."""
import argparse,hashlib,json
from pathlib import Path


def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('new cases artifact required')
    manifest=json.loads((a.inputs/'manifest.json').read_text());arrays={};receipts=[]
    if manifest['source_sha256']['continuous_primitive_block.py']!='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158':raise ValueError('exact trained fcba lineage inputs required')
    for bucket,stage in((8,0),(16,1),(48,3)):
        entry=manifest['files'][f'{stage}:train'];path=a.inputs/entry['leaf'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=entry['sha256']:raise ValueError('TRAIN source hash changed')
        with np.load(path,allow_pickle=False)as stored:data={key:stored[key]for key in('initial','tokens','lengths','targets')}
        eligible=np.flatnonzero(data['lengths']<=bucket)[:64]
        if len(eligible)!=64:raise ValueError('fixed64 eligible TRAIN rows unavailable; do not weaken bucket')
        selected={key:value[eligible]for key,value in data.items()}
        if np.any(selected['tokens'][:,bucket:]):raise ValueError('cannot drop nonzero padding or active information')
        if np.any(selected['lengths']>bucket):raise ValueError('active extent beyond fixed bucket')
        padded=np.zeros((64,bucket),dtype=np.int64);width=min(bucket,selected['tokens'].shape[1]);padded[:,:width]=selected['tokens'][:,:width]
        for key,value in selected.items():arrays[f'b{bucket}_{key}']=padded if key=='tokens'else value
        receipts.append(dict(bucket=bucket,source_stage=stage,source_sha256=digest,selected_rows=eligible.tolist(),rule='first64 TRAIN rows fitting fixed length, never values/scores',original_stored_columns=data['tokens'].shape[1],active_tokens=int(selected['lengths'].sum()),padded_transition_steps=64*bucket,labels=64))
    arrays['metadata_json']=np.frombuffer(json.dumps(dict(schema='factorized-fixed-graph-FB-cases-v1',TRAIN_only=True,new_teacher_labels=0,receipts=receipts),sort_keys=True).encode(),dtype=np.uint8)
    with a.out.open('xb')as stream:np.savez_compressed(stream,**arrays)
    print(json.dumps(dict(cases=str(a.out),sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),receipts=receipts),sort_keys=True))

if __name__=='__main__':main()
