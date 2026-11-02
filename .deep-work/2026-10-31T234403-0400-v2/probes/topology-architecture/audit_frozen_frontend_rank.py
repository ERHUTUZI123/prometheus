"""ROOT-RUN frozen frontend rank/TRAIN kernel-energy audit, never training.

Use the existing bounded offline CPU runner. No network, SSH or model source
import/forward. Reads one OWN real-feature checkpoint and the matching TRAIN
feature cache; target arrays and TEST/VALID caches are never loaded. The audit
does not establish label relevance or identify the cause of a performance gap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def digest(path: Path) -> str:
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=60.)
    parser.add_argument('--chunk-rows',type=int,default=128)
    args=parser.parse_args()
    if not 1<=args.max_seconds<=120 or not 1<=args.chunk_rows<=256:
        raise ValueError('bounded CPU audit requires <=120s and <=256 rows per chunk')
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size>100<<20:
        raise ValueError('OWN <=100MiB checkpoint required')
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+args.max_seconds
    report={'status':'started','scope':'frozen frontend geometry on TRAIN inputs; not label relevance or a learning result',
        'checkpoint_sha256_before':digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
        'source_sha256':digest(Path(__file__)),'no_training':True,'model_forward_calls':0,
        'optimizer_steps':0,'target_arrays_loaded':False,'VALID_TEST_arrays_loaded':False}
    def check():
        if time.perf_counter()>=deadline:raise TimeoutError('bounded frontend audit deadline')
    try:
        # This restoration occurs only inside ROOT's authorized isolated audit
        # process. The coordinator must copy the OWN file read-only into /opt.
        import numpy as np
        import torch
        torch.set_num_threads(1);check()
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if payload.get('schema')!='real-feature-neural-development-v1' or payload.get('completed_updates')!=300:
            raise ValueError('owned completed300-update real-feature checkpoint required')
        metadata_path=args.data/'metadata.json'
        metadata=json.loads(metadata_path.read_text())
        if metadata.get('schema')!='real-feature-development-v1' or payload['metadata']!=metadata:
            raise ValueError('frozen data/checkpoint metadata mismatch')
        train_entry=metadata['splits']['train'];leaf=train_entry['file']
        if Path(leaf).name!=leaf or not leaf.endswith('.npz'):
            raise ValueError('TRAIN cache must be one owned NPZ leaf')
        train_path=args.data/leaf
        if train_path.stat().st_size>128<<20 or digest(train_path)!=train_entry['sha256']:
            raise ValueError('frozen TRAIN cache hash/size mismatch')
        check()
        state=payload['model_state']
        if 'projection.weight' not in state:raise ValueError('checkpoint has no linear frontend projection')
        weight=state['projection.weight'].detach().cpu()
        if weight.ndim!=2 or not bool(torch.isfinite(weight).all()):raise ValueError('finite projection matrix required')
        matrix=weight.to(torch.float64).numpy().copy()
        output_width,input_width=matrix.shape
        if input_width!=metadata['feature_dimension'] or max(matrix.shape)>512:
            raise ValueError('projection differs from public feature shape/bounds')
        with np.load(train_path,allow_pickle=False) as archive:
            # Deliberately do not access targets or even their array headers.
            features=archive['features']
        if (features.dtype!=np.float32 or list(features.shape)!=train_entry['shape']
                or features.ndim!=3 or features.shape[-1]!=input_width or not np.isfinite(features).all()):
            raise ValueError('TRAIN feature array violates frozen contract')
        check();_,singular,vt=np.linalg.svd(matrix,full_matrices=False)
        largest=float(singular[0]) if len(singular) else 0.
        tol64=np.finfo(np.float64).eps*max(matrix.shape)*largest
        tol32=np.finfo(np.float32).eps*max(matrix.shape)*largest
        rank64=int((singular>tol64).sum());rank32=int((singular>tol32).sum())
        basis=vt[:rank64]
        error=float(np.max(np.abs(basis@basis.T-np.eye(rank64)))) if rank64 else 0.
        if error>1e-8:raise ArithmeticError('SVD row-space basis is not numerically orthonormal')
        mean=features.mean(axis=(0,1),dtype=np.float64)
        total=retained=0.
        chunks=0
        for offset in range(0,len(features),args.chunk_rows):
            check();values=features[offset:offset+args.chunk_rows].reshape(-1,input_width).astype(np.float64)
            values-=mean
            coefficients=values@basis.T
            total+=float(np.sum(values*values,dtype=np.float64))
            retained+=float(np.sum(coefficients*coefficients,dtype=np.float64));chunks+=1
        if total<=0 or retained>total+1e-8*total:raise ArithmeticError('invalid centered feature-energy decomposition')
        residual=max(0.,total-retained)
        after=digest(args.checkpoint)
        if after!=report['checkpoint_sha256_before']:raise RuntimeError('checkpoint file changed during read-only audit')
        report.update(status='completed_frozen_frontend_audit',checkpoint_sha256_after=after,
            checkpoint_unchanged=True,completed_training_updates=payload['completed_updates'],
            dataset=metadata['dataset'],arm=payload['args']['arm'],
            metadata_sha256=digest(metadata_path),TRAIN_cache_sha256=train_entry['sha256'],
            TRAIN_rows=len(features),TRAIN_tokens=int(features.shape[0]*features.shape[1]),
            TRAIN_feature_shape=list(features.shape),TRAIN_feature_array_bytes=int(features.nbytes),
            actual_projection_shape=list(matrix.shape),actual_projection_dtype=str(weight.dtype),
            singular_values=singular.tolist(),rank_float64_tolerance=rank64,rank_float32_tolerance=rank32,
            tolerance_float64=float(tol64),tolerance_float32=float(tol32),
            unavoidable_kernel_dimension=input_width-rank64,basis_orthonormality_max_error=error,
            centered_TRAIN_total_energy=total,centered_TRAIN_rowspace_energy=retained,
            centered_TRAIN_kernel_energy=residual,centered_TRAIN_kernel_energy_fraction=residual/total,
            TRAIN_feature_mean=mean.tolist(),mean_payload_bytes=int(mean.nbytes),chunks=chunks,
            checkpoint_source_sha256=payload['source_sha256'],
            interpretation='kernel energy measures discarded input variation; labels unused, task relevance and causal learning effects remain unresolved')
    except Exception as error:
        report.update(status='failed_or_bounded_incomplete',error=f'{type(error).__name__}: {error}')
        (args.out/'exception.txt').write_text(traceback.format_exc())
    finally:
        report['wall_seconds']=time.perf_counter()-started
        with (args.out/'summary.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({'status':report['status'],'error':report.get('error'),
            'rank':report.get('rank_float64_tolerance'),'kernel_energy_fraction':report.get('centered_TRAIN_kernel_energy_fraction')}))
    if report['status']!='completed_frozen_frontend_audit':raise SystemExit(1)


if __name__=='__main__':main()
