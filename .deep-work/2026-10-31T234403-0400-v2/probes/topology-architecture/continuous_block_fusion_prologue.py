"""Prospective fixed trusted OWN continuous-Block fusion compiler prologue.

No task data/checkpoint, no learning/update, no arbitrary Python path or options.
The existing explicit trusted Torch/CUDA bootstrap precedes this phase; full
compiler socket seccomp is already installed. Same pinned toolchain checks,
filesystem/identity/resources and loader-trust boundary as the verified staged
runner. This primes only the SHA-pinned OWN FP32 soft kernel on fixed synthetic
41,593-element packed weights and B1/32 x L64/128/256, then strict isolation is
irreversibly installed before any real-model audit. No numerical/performance
claim follows from successful priming. Authoring does not execute this file.
"""
from __future__ import annotations

import ctypes
import errno
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

KERNEL_LEAF = 'fused_continuous_block_fp32.py'
KERNEL_SHA256 = '5cf6929bb0da78d8a7dc65cbdfe7d9dcaf0ba9d182845564f3f36c06e6f88d29'
TOOL_COMMANDS = (
    ('ldconfig', '/usr/sbin/ldconfig.real', '-p'),
    ('gcc', '/usr/bin/x86_64-linux-gnu-gcc-13', '--version'),
    ('cc1', '/usr/libexec/gcc/x86_64-linux-gnu/13/cc1', '--version'),
    ('as', '/usr/bin/x86_64-linux-gnu-as', '--version'),
    ('collect2', '/usr/libexec/gcc/x86_64-linux-gnu/13/collect2', '--version'),
    ('ld', '/usr/bin/x86_64-linux-gnu-ld.bfd', '--version'),
    ('ptxas', '/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas', '--version'),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cache(out):
    result = {}
    # Non-overlapping owned cache directories only; exclude logs/checkpoints.
    for name in ('tmp', 'torch', 'cache', 'cuda-cache', 'triton-cache', 'pycache'):
        root = out / name
        files = total = 0
        for folder, directories, leaves in os.walk(root, followlinks=False):
            directories[:] = [n for n in directories if not (Path(folder) / n).is_symlink()]
            for leaf in leaves:
                path = Path(folder) / leaf
                if not path.is_symlink():
                    total += path.stat().st_size; files += 1
                if files > 20000:
                    raise RuntimeError('bounded owned-cache metadata walk exceeded')
        result[name] = dict(files=files, bytes=total)
    return result


def run_prologue(torch, out, cpu_only=False):
    out = Path(out); started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = dict(phase='trusted_fixed_compiler_prologue', completed=False,
                  task_data_used=False, optimizer_updates=0,
                  output_is_training_evidence=False, cpu_only=bool(cpu_only),
                  cache_before=_cache(out), compiler_tools=[],
                  loader_indirect_execution_risk='accepted only for this inspected fixed prologue')
    def save():
        with (out / 'compiler-prologue.json').open('w') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    try:
        # Positive executable checks are fixed; no arbitrary binary/argv input.
        for role, path, option in TOOL_COMMANDS:
            tick = time.perf_counter()
            result = subprocess.run([path, option], stdin=subprocess.DEVNULL,
                                    capture_output=True, timeout=15, check=False)
            entry = dict(role=role, path=path, returncode=result.returncode,
                         wall_seconds=time.perf_counter()-tick,
                         stdout_bytes=len(result.stdout), stderr_bytes=len(result.stderr),
                         stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
                         stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
            record['compiler_tools'].append(entry)
            if result.returncode != 0:
                raise RuntimeError(f'fixed compiler positive check failed: {role}: {result.returncode}')
        # Fixed C input exercises gcc->cc1/as/collect2/ld, and owned-only writes.
        library = out / 'tmp' / 'compiler-policy-probe.so'
        tick = time.perf_counter()
        result = subprocess.run(['/usr/bin/x86_64-linux-gnu-gcc-13', '-shared', '-fPIC', '-O0',
                                 '-x', 'c', '-o', str(library), '-'],
                                input=b'int compiler_policy_value(void) { return 73; }\n',
                                capture_output=True, timeout=20, check=False)
        record['fixed_c_compile'] = dict(returncode=result.returncode,
            wall_seconds=time.perf_counter()-tick,
            stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
            stderr_excerpt=result.stderr[:4096].decode('utf-8', errors='replace'))
        if result.returncode != 0 or not library.is_file():
            raise RuntimeError('fixed gcc compilation failed')
        # dlopen is library loading, not execve; readable libraries remain trusted.
        handle = ctypes.CDLL(str(library)); handle.compiler_policy_value.restype = ctypes.c_int
        if handle.compiler_policy_value() != 73:
            raise RuntimeError('fixed compiler library value mismatch')
        record['fixed_c_compile'].update(bytes=library.stat().st_size, sha256=_sha(library),
                                         library_function_verified=True)
        # Full compiler socket seccomp is already installed before this file.
        try:
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EACCES):
                raise RuntimeError(f'network check wrong errno: {error.errno}') from error
            record['network_socket_denied'] = dict(errno=error.errno)
        else:
            candidate.close(); raise RuntimeError('network socket unexpectedly permitted')
        for label, path, rest in (('shell', '/bin/sh', ['-c', 'exit 73']),
                                  ('python', sys.executable, ['-c', 'raise SystemExit(73)']),
                                  ('shell_ldconfig_wrapper', '/usr/sbin/ldconfig', ['-p'])):
            try:
                result = subprocess.run([path, *rest], stdin=subprocess.DEVNULL,
                                        capture_output=True, timeout=5, check=False)
            except OSError as error:
                if error.errno not in (errno.EPERM, errno.EACCES):
                    raise RuntimeError(f'{label} direct exec wrong errno: {error.errno}') from error
                record[label + '_direct_exec_denied'] = dict(errno=error.errno)
            else:
                raise RuntimeError(f'{label} direct exec unexpectedly admitted: {result.returncode}')
        if cpu_only:
            record['kernel_priming']='explicit CPU policy-check mode; no kernel or GPU fallback'
        else:
            path=Path(__file__).resolve().with_name(KERNEL_LEAF)
            if path.is_symlink()or not path.is_file()or _sha(path)!=KERNEL_SHA256:
                raise RuntimeError('fixed OWN kernel source SHA changed')
            spec=importlib.util.spec_from_file_location('fused_continuous_block_fp32',path)
            if spec is None or spec.loader is None:raise RuntimeError('fixed kernel import unavailable')
            module=importlib.util.module_from_spec(spec)
            sys.modules['fused_continuous_block_fp32']=module
            spec.loader.exec_module(module)
            if module.triton.__version__!='3.4.0'or not str(torch.__version__).startswith('2.8.'):
                raise RuntimeError('existing Triton3.4/Torch2.8 route required; no install/fallback')
            record.update(kernel_sha256=KERNEL_SHA256,kernel_cases=[],
                synthetic_packed_elements=41593,synthetic_packed_bytes=41593*4,
                original_real_model_loaded=False,task_data_used=False,
                numerical_equivalence_verified=False,performance_advantage_verified=False,
                launch=dict(num_warps=4,num_stages=1,enable_fp_fusion=False,autotune=False),
                kernel_artifacts=[])
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(20261102);torch.cuda.manual_seed_all(20261102)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(0)
                packed=torch.randn(41593,device='cuda:0',dtype=torch.float32)*.02
                for length in (64,128,256):
                    for batch in (1,32):
                        tokens=torch.randint(0,100,(batch,length),device='cuda:0',dtype=torch.long)
                        buffers=module.allocate_buffers(tokens)
                        torch.cuda.synchronize();tick=time.perf_counter()
                        with torch.no_grad():compiled=module.launch_packed(tokens,packed,buffers)
                        torch.cuda.synchronize()
                        entry=dict(batch=batch,context=length,wall_seconds=time.perf_counter()-tick,
                            output_bytes=buffers['output'].numel()*4,
                            extra_final_carry_bytes=sum(buffers[n].numel()*4 for n in ('h','keys','values')),
                            kernel_n_regs=getattr(compiled,'n_regs',None),kernel_n_spills=getattr(compiled,'n_spills',None))
                        if not bool(torch.isfinite(buffers['output']).all()):
                            raise RuntimeError('nonfinite fixed synthetic kernel output')
                        record['kernel_cases'].append(entry)
                        # Save one emitted PTX per L when exposed; cache/cubin
                        # artifacts remain in the newly owned Triton cache tree.
                        asm=getattr(compiled,'asm',{})
                        ptx=asm.get('ptx')if isinstance(asm,dict)else None
                        if batch==1 and isinstance(ptx,str):
                            artifact=out/f'fused-soft-fp32-L{length}.ptx'
                            with artifact.open('x')as stream:stream.write(ptx)
                            record['kernel_artifacts'].append(dict(file=artifact.name,
                                bytes=artifact.stat().st_size,sha256=_sha(artifact)))
                        del buffers,tokens
                record.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                del packed
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        record['completed'] = True
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(wall_seconds=time.perf_counter()-started,
            cpu_self_user_seconds=usage.ru_utime-usage_before.ru_utime,
            cpu_self_system_seconds=usage.ru_stime-usage_before.ru_stime,
            cpu_children_user_seconds=children.ru_utime-child_before.ru_utime,
            cpu_children_system_seconds=children.ru_stime-child_before.ru_stime,
            cpu_peak_RSS_bytes=int(usage.ru_maxrss*1024), cache_after=_cache(out))
        save()
    return record
