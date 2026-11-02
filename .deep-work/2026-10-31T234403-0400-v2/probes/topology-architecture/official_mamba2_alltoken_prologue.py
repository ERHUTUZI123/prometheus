"""Fixed trusted official Mamba2 all-token shape compiler prologue.

No task data, optimizer or SGD/update, no CLI options and no arbitrary script.
The only change from the verified numeric-smoke prologue is fixed architecture
and shape coverage for the prospective V100/H64 all-token driver. Random inputs
are independent of TRAIN/VALID/TEST. All priming/compilation/cache/resource work
is charged and never reported as 300-update task evidence.

This full socket-denied compiler phase follows the existing explicit trusted
Torch/CUDA bootstrap. Admitted loader can indirectly run readable code: trust
is only in this pinned prologue, not an adversarial compiler-purpose proof.
"""
from __future__ import annotations

import ctypes
import errno
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

PACKAGE_ROOT = Path('/opt/prometheus-mamba231-r1')
OFFICIAL_FILES = {
    'mamba_ssm/__init__.py': '6d3fee6ff8e8ec4dcab0f7af6c7890ff81f3d9319707865f5a651239aee9ae0a',
    'mamba_ssm/modules/mamba2.py': '605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707',
    'mamba_ssm/models/mixer_seq_simple.py': '13409d7044e930ea3271e4b8ddceaf8155ec49b8e5ac299fba7bb0df6d80cb21',
    'mamba_ssm/models/config_mamba.py': '2a72c1686f775b56547e39ca4406ba10148d12fd7a791c57ce2ba85126010fcd',
    'mamba_ssm/ops/triton/ssd_combined.py': '868540ba0c45e86ac3f610b17b62dc6ede5a80a91c2c8ef24f7e3314be69b626',
}
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
        # Full socket seccomp is installed before this prologue/compiler call.
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
            record['mamba_priming'] = 'explicit CPU policy-test mode; no Mamba or GPU fallback'
        else:
            record['official_source_sha256'] = {}
            for relative, expected in OFFICIAL_FILES.items():
                path = PACKAGE_ROOT / relative
                if not path.is_file() or path.is_symlink() or _sha(path) != expected:
                    raise RuntimeError(f'official Mamba source differs: {relative}')
                record['official_source_sha256'][relative] = expected
            sys.path.insert(0, str(PACKAGE_ROOT))
            tick = time.perf_counter(); package = importlib.import_module('mamba_ssm')
            if package.__version__ != '2.3.1' or not Path(package.__file__).resolve().is_relative_to(PACKAGE_ROOT):
                raise RuntimeError('unexpected actual official Mamba package')
            record['official_import_seconds'] = time.perf_counter()-tick
            config = dict(d_model=64, d_state=16, expand=2, headdim=16,
                          chunk_size=32, use_mem_eff_path=False)
            shapes = (('training_forward_backward',32,64,True),
                      ('causality_forward_only',2,64,False),
                      ('validation_and_reload_forward_only',32,64,False),
                      ('test128_forward_only',32,128,False),
                      ('test256_forward_only',32,256,False))
            record.update(configuration=config, vocabulary=100, dtype='float32',
                          shape_coverage=[], forwards_completed=0, backwards_completed=0,
                          optimizer_updates=0, task_data_used=False,
                          reload_shape='two B32x64 forwards; not B64; covered by validation shape',
                          B1_inference_included=False)
            # Mirrors the exact task wrapper/strides without importing research
            # code or reading any data. Random-token inputs are synthetic only.
            class FixedPrimingLM(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding=torch.nn.Embedding(100,64)
                    self.body=package.Mamba2(**config)
                    self.norm=torch.nn.LayerNorm(64)
                    self.readout=torch.nn.Linear(64,100)
                def forward(self,tokens):
                    value=self.norm(self.body(self.embedding(tokens)))
                    return torch.nn.functional.log_softmax(self.readout(value),-1)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(20261102);torch.cuda.manual_seed_all(20261102)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(0)
                model=FixedPrimingLM().to('cuda:0')
                if any(v.dtype!=torch.float32 for v in model.parameters()):
                    raise RuntimeError('fixed priming model must be FP32')
                record['priming_model_actual_parameters']=sum(v.numel() for v in model.parameters())
                record['priming_model_parameter_bytes']=sum(v.numel()*v.element_size() for v in model.parameters())
                for label,batch,length,backward in shapes:
                    model.train(backward);model.zero_grad(set_to_none=True)
                    tokens=torch.randint(0,100,(batch,length),device='cuda:0',dtype=torch.long)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    with torch.set_grad_enabled(backward):
                        output=model(tokens)
                    torch.cuda.synchronize()
                    entry=dict(purpose=label,input_shape=[batch,length,64],token_shape=[batch,length],
                               requires_backward=backward,forward_seconds=time.perf_counter()-tick)
                    if output.shape!=(batch,length,100)or not bool(torch.isfinite(output).all()):
                        raise RuntimeError('fixed all-token priming shape/finiteness failed: '+label)
                    record['forwards_completed']+=1
                    if backward:
                        tick=time.perf_counter();output.square().mean().backward();torch.cuda.synchronize()
                        entry['backward_seconds']=time.perf_counter()-tick
                        gradients=[v.grad for v in model.parameters()if v.grad is not None]
                        if not gradients or any(not bool(torch.isfinite(v).all())for v in gradients):
                            raise RuntimeError('fixed all-token priming gradient failed')
                        record['backwards_completed']+=1
                    record['shape_coverage'].append(entry)
                    del output,tokens
                record.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                del model
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
