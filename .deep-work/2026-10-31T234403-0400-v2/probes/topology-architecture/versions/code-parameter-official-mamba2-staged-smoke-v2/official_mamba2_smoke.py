"""ROOT-RUN unmodified official Mamba2 import/gradient/reload smoke only.

No training baseline is claimed. No kernel fallback, monkeypatch, package write,
outside-run prewarm, network or subprocess is introduced here. Compiler/exec
denials under the unchanged sandbox are genuine failures and remain recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


PACKAGE_ROOT = Path("/opt/prometheus-mamba231-r1")
OFFICIAL_FILES = {
    "mamba_ssm/__init__.py": "6d3fee6ff8e8ec4dcab0f7af6c7890ff81f3d9319707865f5a651239aee9ae0a",
    "mamba_ssm/modules/mamba2.py": "605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707",
    "mamba_ssm/models/mixer_seq_simple.py": "13409d7044e930ea3271e4b8ddceaf8155ec49b8e5ac299fba7bb0df6d80cb21",
    "mamba_ssm/models/config_mamba.py": "2a72c1686f775b56547e39ca4406ba10148d12fd7a791c57ce2ba85126010fcd",
    "mamba_ssm/ops/triton/ssd_combined.py": "868540ba0c45e86ac3f610b17b62dc6ede5a80a91c2c8ef24f7e3314be69b626",
}
CONFIG = dict(d_model=32, d_state=16, expand=2, headdim=16, chunk_size=32, use_mem_eff_path=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--max-seconds", type=float, default=60)
    args = p.parse_args()
    if not 1 <= args.max_seconds <= 60: p.error("smoke internal bound must be <=60seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); stage = "source_check"; torch = None; model = None
    stream = (args.out / "records.jsonl").open("x", buffering=1)
    facts = dict(status="not_completed", configuration=CONFIG, input_shape=[2,16,32], dtype="float32",
        training_baseline_completed=False, fallback_or_runtime_patch=False, package_root=str(PACKAGE_ROOT),
        driver_sha256=sha(Path(__file__)), forwards_completed=0, backwards_completed=0, optimizer_updates=0,
        primary_sources=["https://github.com/state-spaces/mamba/tree/v2.3.1",
            "https://github.com/state-spaces/mamba/blob/v2.3.1/mamba_ssm/modules/mamba2.py"])
    def emit(kind, **value):
        stream.write(json.dumps(dict(kind=kind, stage=stage, wall=time.perf_counter()-started, **value), sort_keys=True, allow_nan=False)+"\n")
    def timeout(_signum, _frame): raise TimeoutError("official Mamba2 smoke internal60second bound")
    def sync():
        if torch is not None and torch.cuda.is_available(): torch.cuda.synchronize(args.device)
    def cache_snapshot():
        result = {}
        for name in ("TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_HOME", "XDG_CACHE_HOME"):
            text = os.environ.get(name); entry = dict(path=text, files=0, bytes=0)
            if text:
                root = Path(text).resolve()
                if root.is_relative_to(args.out.parent.resolve()) and root.is_dir():
                    for folder, dirs, files in os.walk(root, followlinks=False):
                        dirs[:] = [d for d in dirs if not (Path(folder)/d).is_symlink()]
                        for leaf in files:
                            f = Path(folder)/leaf
                            if not f.is_symlink(): entry["files"] += 1; entry["bytes"] += f.stat().st_size
                        if entry["files"] > 20000: raise RuntimeError("bounded cache metadata walk exceeded")
                    entry["scope"] = "owned outerrun subtree, metadata only; overlapping cache variables not additive"
                else: entry["scope"] = "outside owned outerrun or absent; no traversal"
            result[name] = entry
        return result
    def model_hash(value):
        h = hashlib.sha256()
        for name, tensor in sorted(value.state_dict().items()):
            t = tensor.detach().cpu().contiguous(); h.update(json.dumps([name,str(t.dtype),list(t.shape)]).encode()); h.update(t.numpy().tobytes())
        return h.hexdigest()
    def tensor_bytes(value):
        if torch.is_tensor(value): return value.numel()*value.element_size()
        if isinstance(value, dict): return sum(tensor_bytes(x) for x in value.values())
        if isinstance(value,(list,tuple)): return sum(tensor_bytes(x) for x in value)
        return 0
    def save_json(name, value):
        with (args.out/name).open("x") as dest: json.dump(value,dest,indent=2,sort_keys=True,allow_nan=False);dest.write("\n")
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    try:
        facts["official_source_sha256"] = {}
        for relative, expected in OFFICIAL_FILES.items():
            path = PACKAGE_ROOT/relative
            if not path.is_file() or path.is_symlink() or sha(path)!=expected: raise ValueError(f"official2.3.1 source missing/changed: {relative}")
            facts["official_source_sha256"][relative] = expected
        facts["cache_before_import"] = cache_snapshot(); emit("source_verified", **facts)
        sys.path.insert(0,str(PACKAGE_ROOT))
        stage = "official_full_import"; tick = time.perf_counter()
        torch = importlib.import_module("torch")
        package = importlib.import_module("mamba_ssm")  # full, unmodified package import
        Mamba2 = package.Mamba2
        if package.__version__!="2.3.1" or not Path(package.__file__).resolve().is_relative_to(PACKAGE_ROOT): raise ValueError("wrong actual Mamba package")
        facts["full_import_seconds"] = time.perf_counter()-tick
        facts["versions"] = {}; facts["actual_module_paths"] = {}
        for distribution in ("mamba-ssm","torch","triton","einops","transformers","huggingface-hub","ninja","packaging"):
            try: facts["versions"][distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError: facts["versions"][distribution] = None
        for name in ("torch","triton","mamba_ssm","einops","transformers","huggingface_hub","packaging"):
            module = sys.modules.get(name); facts["actual_module_paths"][name] = getattr(module,"__file__",None)
        module = importlib.import_module("mamba_ssm.modules.mamba2")
        facts["official_optional_causal_conv_available"] = module.causal_conv1d_fn is not None
        facts["torch_CUDA_version"] = torch.version.cuda
        facts["CUDA_available"] = torch.cuda.is_available()
        if not facts["CUDA_available"] or torch.device(args.device).type!="cuda": raise RuntimeError("explicit authorized CUDA required; no CPU fallback")
        facts["GPU"] = torch.cuda.get_device_name(args.device)
        facts["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        facts["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
        emit("official_import_completed", import_seconds=facts["full_import_seconds"], versions=facts["versions"], module_paths=facts["actual_module_paths"])
        stage = "construct"; torch.manual_seed(20261101); torch.cuda.manual_seed_all(20261101)
        model = Mamba2(**CONFIG,device=args.device,dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        x = torch.randn(2,16,32,device=args.device,dtype=torch.float32,requires_grad=True)
        target = torch.randn_like(x)
        if any(t.dtype!=torch.float32 for t in model.parameters()): raise ValueError("actual model dtype differs from fp32")
        facts["parameters"] = sum(t.numel() for t in model.parameters())
        facts["parameter_bytes"] = tensor_bytes(dict(model.named_parameters()))
        facts["buffer_bytes"] = tensor_bytes(dict(model.named_buffers()))
        before = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        facts["initial_model_state_sha256"] = model_hash(model)
        with (args.out/"initial-actual-state.pt").open("xb") as dest: torch.save(dict(config=CONFIG,state_dict=before),dest)
        facts["cache_before_first_forward"] = cache_snapshot(); torch.cuda.reset_peak_memory_stats(args.device)
        stage = "cold_forward"; sync(); tick = time.perf_counter(); output = model(x); sync()
        facts["cold_forward_seconds"] = time.perf_counter()-tick; facts["forwards_completed"] += 1
        if output.shape!=x.shape or output.dtype!=torch.float32 or not bool(torch.isfinite(output).all()): raise FloatingPointError("actual forward shape/dtype/finiteness failed")
        loss = (output-target).square().mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite actual smoke objective")
        facts["loss"] = float(loss.detach()); emit("cold_forward_completed", seconds=facts["cold_forward_seconds"], output_shape=list(output.shape), loss=facts["loss"], cache=cache_snapshot())
        stage = "backward"; sync(); tick = time.perf_counter(); loss.backward(); sync()
        facts["backward_seconds"] = time.perf_counter()-tick; facts["backwards_completed"] += 1
        gradients = {name:None if value.grad is None else dict(norm=float(value.grad.norm()),finite=bool(torch.isfinite(value.grad).all())) for name,value in model.named_parameters()}
        if any(value is not None and not value["finite"] for value in gradients.values()) or x.grad is None or not bool(torch.isfinite(x.grad).all()): raise FloatingPointError("nonfinite/missing actual gradient")
        if not any(value is not None and value["norm"]>0 for value in gradients.values()): raise RuntimeError("no nonzero actual parameter gradient")
        facts["gradients"] = gradients; facts["input_gradient_norm"] = float(x.grad.norm())
        stage = "one_Adam_update"; tick = time.perf_counter(); optimizer.step(); sync()
        facts["optimizer_seconds"] = time.perf_counter()-tick; facts["optimizer_updates"] += 1
        if any(not bool(torch.isfinite(value).all()) for value in model.parameters()): raise FloatingPointError("nonfinite post-update parameter")
        changed = [name for name,value in model.state_dict().items() if not torch.equal(value.detach().cpu(),before[name])]
        if not changed: raise RuntimeError("one Adam proposal did not change any actual state")
        facts["changed_state_names"] = changed; facts["post_update_model_state_sha256"] = model_hash(model)
        facts["optimizer_tensor_bytes"] = tensor_bytes(optimizer.state_dict())
        emit("actual_gradient_and_update_completed", gradients=gradients, changed_state_names=changed, backwards_completed=1, optimizer_updates=1)
        stage = "disk_reload"; checkpoint = args.out/"official-mamba2-actual-step1.pt"; tick=time.perf_counter()
        with checkpoint.open("xb") as dest: torch.save(dict(schema="official-mamba2-smoke-v1",configuration=CONFIG,
            state_dict=model.state_dict(),optimizer=optimizer.state_dict(),optimizer_updates=1,
            package_version=package.__version__,official_source_sha256=OFFICIAL_FILES),dest)
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        other=Mamba2(**payload["configuration"],device=args.device,dtype=torch.float32);other.load_state_dict(payload["state_dict"],strict=True)
        equal=all(torch.equal(value.detach().cpu(),other.state_dict()[name].detach().cpu()) for name,value in model.state_dict().items())
        model.eval();other.eval()
        with torch.no_grad():
            expected=model(x.detach());actual=other(x.detach());sync();facts["forwards_completed"]+=2
        if not bool(torch.isfinite(actual).all()):raise FloatingPointError("nonfinite actual reload forward")
        difference=float((expected-actual).abs().max());outputs_equal=bool(torch.equal(expected,actual));outputs_close=bool(torch.allclose(expected,actual,rtol=1e-5,atol=1e-6))
        if not equal or not outputs_close:raise RuntimeError("actual checkpoint reload mismatch")
        facts["reload_audit"]=dict(tensor_values_equal=equal,output_bitwise_equal=outputs_equal,output_allclose_rtol1e5_atol1e6=outputs_close,
            maximum_output_difference=difference,checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
            wall_seconds=time.perf_counter()-tick,exact_training_resume_tested=False)
        facts.update(status="completed_official_Mamba2_smoke",cache_after=cache_snapshot(),wall_seconds=time.perf_counter()-started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
            cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)))
        save_json("summary.json",facts);emit("completed",status=facts["status"],forwards_completed=facts["forwards_completed"],backwards_completed=1,optimizer_updates=1)
        returncode=0
    except BaseException as error:
        facts.update(status="failed_official_smoke_as_measured",failed_stage=stage,error=f"{type(error).__name__}: {error}",wall_seconds=time.perf_counter()-started)
        (args.out/"exception.txt").write_text(traceback.format_exc())
        try:facts["cache_after_failure"]=cache_snapshot()
        except BaseException as cache_error:facts["cache_snapshot_error"]=str(cache_error)
        save_json("failure.json",facts);emit("failed",error=facts["error"],failed_stage=stage)
        returncode=1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);stream.close()
    print(json.dumps(dict(status=facts["status"],out=str(args.out),returncode=returncode,failed_stage=facts.get("failed_stage"))))
    raise SystemExit(returncode)


if __name__=="__main__":main()
