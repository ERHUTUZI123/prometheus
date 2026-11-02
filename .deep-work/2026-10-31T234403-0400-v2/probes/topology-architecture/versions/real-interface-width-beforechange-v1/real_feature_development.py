"""ROOT-RUN bounded AUDIO/VISION development, never a formal Primary benchmark.

prepare is NumPy/stdlib only. train imports existing Torch locally. One frozen
feature cache and identical300x64 minibatch indices serve all arms. First10
official classes, public35/100-way output; no task ID, correct route, operation,
intermediate state or TEST-based selection. Requires external isolation/time cap.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
import traceback

import numpy as np

from dev_dataset_readers import CIFAR100Binary, SpeechCommands35, log_mel_16k


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def source_hashes(include_core=False):
    names = [Path(__file__).name, "dev_dataset_readers.py"]
    if include_core: names.append("continuous_primitive_block.py")
    return {n: sha(Path(__file__).with_name(n)) for n in names}


def prepare(args):
    started = time.perf_counter(); deadline = started + args.max_seconds
    args.out.mkdir(parents=True, exist_ok=False)
    public = 35 if args.dataset == "speech" else 100
    canonical = "speech35" if args.dataset == "speech" else "cifar100"
    Reader = SpeechCommands35 if args.dataset == "speech" else CIFAR100Binary
    manifest_path = args.raw_root / "prepared" / canonical / "manifest.json"
    metadata = dict(schema="real-feature-development-v1", dataset=args.dataset,
        development_only=True, formal_Primary=False, selected_classes=list(range(10)),
        public_output_classes=public, seed=args.seed,
        original_manifest_sha256=sha(manifest_path), original_manifest=json.loads(manifest_path.read_text()),
        raw_root=str(args.raw_root), sources=source_hashes(), splits={},
        supervision="original terminal class only; full35/100 output head; no task or route input",
        protocol="single static first10-class diagnostic, not continual learning or full-class benchmark",
        normalization="per-feature mean/std from selected TRAIN tokens only; shared across methods",
        test_policy="deterministic feature preparation allowed; no TEST scoring/selection during training")
    raw_data = {}
    for split, cap in (("train", 128), ("valid", 64), ("test", 128)):
        dataset = Reader(args.raw_root, split, classes=range(10))
        if args.dataset == "speech":
            labels = np.asarray([r["label"] for r in dataset.rows], dtype=np.int64)
        else:
            labels = np.asarray(dataset.records[dataset.indices, 1], dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, public, {"train": 0, "valid": 1, "test": 2}[split], 0x5245414c]))
        selected = []
        for label in range(10):
            candidates = np.flatnonzero(labels == label)
            if not len(candidates): raise ValueError(f"official {split} missing selected class {label}")
            selected.extend(rng.permutation(candidates)[:cap].tolist())
        # Cache order fixed independently of any model; preserve its exact IDs.
        selected = np.asarray(selected, dtype=np.int64)
        selected = selected[rng.permutation(len(selected))]
        features = []; targets = []; identities = []
        for j, index in enumerate(selected):
            if time.perf_counter() > deadline: raise TimeoutError("feature preparation bound")
            value, target = dataset[int(index)]
            if args.dataset == "speech":
                feature = log_mel_16k(value)  # exactly98x40, no SciPy/fitted front end
                identity = dataset.rows[int(index)]["path"]
            else:
                # CHW32x32 -> row-major8x8 patches; each patch4x4xRGB.
                feature = value.reshape(3, 8, 4, 8, 4).transpose(1, 3, 2, 4, 0).reshape(64, 48)
                identity = int(dataset.indices[int(index)])
            expected = (98, 40) if args.dataset == "speech" else (64, 48)
            if feature.shape != expected or not np.isfinite(feature).all(): raise ValueError("invalid fixed features")
            features.append(feature.astype(np.float32)); targets.append(int(target)); identities.append(identity)
            if (j + 1) % 256 == 0: print(json.dumps(dict(event="feature_progress", split=split, completed=j + 1)), flush=True)
        raw_data[split] = (np.stack(features), np.asarray(targets, dtype=np.int64), identities)
    train = raw_data["train"][0]
    mean = train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(train.std(axis=(0, 1), dtype=np.float64).astype(np.float32), np.float32(1e-4))
    with (args.out / "train-normalization.npz").open("xb") as stream:
        np.savez_compressed(stream, mean=mean, scale=scale)
    metadata["normalization_sha256"] = sha(args.out / "train-normalization.npz")
    uncompressed = 0
    for split, (features, targets, identities) in raw_data.items():
        features = ((features - mean[None, None, :]) / scale[None, None, :]).astype(np.float32)
        lengths = np.full(len(features), features.shape[1], dtype=np.int64)
        path = args.out / f"{split}.npz"
        with path.open("xb") as stream: np.savez_compressed(stream, features=features, targets=targets, lengths=lengths)
        id_path = args.out / f"{split}-ids.json"
        write_json(id_path, dict(split=split, identities=identities, targets=targets.tolist(), original_official_split=split))
        uncompressed += features.nbytes + targets.nbytes + lengths.nbytes
        metadata["splits"][split] = dict(file=path.name, sha256=sha(path), compressed_bytes=path.stat().st_size,
            rows=len(features), shape=list(features.shape), ids_file=id_path.name, ids_sha256=sha(id_path),
            class_counts={str(k): int((targets == k).sum()) for k in range(10)},
            array_bytes=features.nbytes + targets.nbytes + lengths.nbytes)
    metadata.update(feature_dimension=int(mean.size), sequence_length=int(train.shape[1]),
        uncompressed_feature_cache_bytes=uncompressed,
        normalization_array_bytes=mean.nbytes + scale.nbytes,
        preparation_wall_seconds=time.perf_counter() - started,
        preparation_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)))
    write_json(args.out / "metadata.json", metadata)
    print(json.dumps(dict(status="prepared", out=str(args.out), dataset=args.dataset, rows={k:v["rows"] for k,v in metadata["splits"].items()}, wall_seconds=metadata["preparation_wall_seconds"])))


def tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(x) for x in value)
    return 0


def build_model(arm, metadata, device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from continuous_primitive_block import ContinuousPrimitiveBlock
    classes, dimension, extent = metadata["public_output_classes"], metadata["feature_dimension"], metadata["sequence_length"]

    class FeatureModel(nn.Module):
        def __init__(self):
            super().__init__(); self.arm = arm; self.dimension = dimension; self.extent = extent; self.classes = classes
            if arm in ("block_dynamic", "block_static", "block_detached"):
                self.projection = nn.Linear(dimension, 16)
                self.body = ContinuousPrimitiveBlock(n_states=classes, n_tokens=1, internal_states=64,
                    n_primitives=4, token_dim=16, state_dim=64, memory_dim=64, caller_hidden=32,
                    n_slots=1, routing={"block_dynamic":"dynamic", "block_static":"static", "block_detached":"detached"}[arm],
                    memory_slots=8, key_dim=16, primitive_rank=8)
            elif arm == "gru":
                self.projection = nn.Linear(dimension, 64)
                self.body = nn.GRU(64, 64, num_layers=2, batch_first=True)
                self.head = nn.Linear(64, classes)
            elif arm == "transformer":
                self.projection = nn.Linear(dimension, 64); self.cls = nn.Parameter(torch.randn(1, 1, 64) * .02)
                self.body = nn.ModuleList([nn.TransformerEncoderLayer(64, 4, dim_feedforward=128,
                    dropout=0., batch_first=True, norm_first=True) for _ in range(3)])
                self.head = nn.Linear(64, classes)
                position = torch.arange(extent + 1, dtype=torch.float32)[:, None]
                frequency = torch.exp(torch.arange(0, 64, 2, dtype=torch.float32) * (-math.log(10000.) / 64))
                encoding = torch.zeros(extent + 1, 64); encoding[:, 0::2] = torch.sin(position * frequency); encoding[:, 1::2] = torch.cos(position * frequency)
                self.register_buffer("positions", encoding)
            elif arm == "cnn":
                if metadata["dataset"] == "speech":
                    self.body = nn.Sequential(nn.Conv1d(dimension, 64, 5, padding=2), nn.GELU(), nn.Conv1d(64, 64, 5, padding=2), nn.GELU())
                else:
                    self.body = nn.Sequential(nn.Conv2d(dimension, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 64, 3, padding=1), nn.GELU())
                self.head = nn.Linear(64, classes)
            else: raise ValueError("unknown architecture")

        def forward(self, features, lengths, native=False):
            if self.arm.startswith("block_"):
                embedded = self.projection(features)
                initial = torch.zeros(len(features), device=features.device, dtype=torch.long)
                return self.body.forward_embedded(initial, embedded, lengths, execution="native" if native else "soft", output="logits")
            if self.arm == "gru":
                hidden, _ = self.body(self.projection(features)); output = hidden[torch.arange(len(features), device=features.device), lengths - 1]
            elif self.arm == "transformer":
                hidden = torch.cat((self.cls.expand(len(features), -1, -1), self.projection(features)), 1) + self.positions[None, :features.shape[1] + 1]
                padding = torch.arange(hidden.shape[1], device=features.device)[None] > lengths[:, None]
                for layer in self.body: hidden = layer(hidden, src_key_padding_mask=padding)
                output = hidden[:, 0]
            elif metadata["dataset"] == "speech":
                output = self.body(features.transpose(1, 2)).mean(-1)
            else:
                output = self.body(features.transpose(1, 2).reshape(len(features), dimension, 8, 8)).mean((-1, -2))
            return F.log_softmax(self.head(output), -1)

        def proxy(self, count, length, native=False):
            # Forward MAC proxies only, excluding nonlinearities, normalization,
            # indexing, backward, optimizer, feature extraction and dispatch.
            if self.arm.startswith("block_"):
                core = self.body.resource_report(batch_size=count, sequence_length=length,
                    active_steps=count * length, execution="native" if native else "soft")["proxy_forward_macs"]
                return core + count * length * dimension * 16
            if self.arm == "gru": return count * length * (dimension * 64 + 2 * 3 * (64 * 64 + 64 * 64)) + count * 64 * classes
            if self.arm == "transformer":
                q = length + 1
                return count * (length * dimension * 64 + 3 * (4 * q * 64 * 64 + 2 * q * q * 64 + 2 * q * 64 * 128) + 64 * classes)
            if metadata["dataset"] == "speech": return count * length * (dimension * 64 * 5 + 64 * 64 * 5) + count * 64 * classes
            return count * 64 * (dimension * 32 * 9 + 32 * 64 * 9) + count * 64 * classes

    return FeatureModel().to(device)


def train(args):
    import torch
    from torch.nn import functional as F
    started = time.perf_counter(); deadline = started + args.max_seconds
    args.out.mkdir(parents=True, exist_ok=False)
    stream = (args.out / "records.jsonl").open("x", buffering=1)
    def emit(kind, **value):
        stream.write(json.dumps(dict(kind=kind, wall=time.perf_counter() - started, **value), sort_keys=True, allow_nan=False) + "\n")
    def check():
        if time.perf_counter() > deadline: raise TimeoutError("real-data training/evaluation bound")
    device = torch.device(args.device)
    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def load(split):
        entry = metadata["splits"][split]; path = args.data / entry["file"]
        if path.stat().st_size > 128 << 20 or sha(path) != entry["sha256"]: raise ValueError("feature cache differs from freeze")
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {"features", "targets", "lengths"}: raise ValueError("unexpected feature keys")
            data = {k: stored[k] for k in stored.files}
        x, y, length = data["features"], data["targets"], data["lengths"]
        if (x.dtype != np.float32 or x.shape != tuple(entry["shape"]) or not np.isfinite(x).all()
                or y.dtype != np.int64 or y.shape != (len(x),) or length.dtype != np.int64
                or length.shape != y.shape or np.any(length != metadata["sequence_length"])
                or np.any(y < 0) or np.any(y >= 10)): raise ValueError("feature/label contract")
        return data
    def batch(data, indices):
        return tuple(torch.as_tensor(data[k][indices], device=device) for k in ("features", "lengths", "targets"))
    def resources():
        result = dict(parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parameter_bytes=tensor_bytes(dict(model.named_parameters())), buffer_bytes=tensor_bytes(dict(model.named_buffers())),
            optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()), replay_bytes=0,
            feature_cache_array_bytes=metadata["uncompressed_feature_cache_bytes"],
            normalization_array_bytes=metadata["normalization_array_bytes"],
            cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)))
        if device.type == "cuda": result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device), cuda_reserved_bytes=torch.cuda.memory_reserved(device), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return result
    @torch.no_grad()
    def evaluate(data):
        was = model.training; model.eval(); total = correct = native_correct = 0; nll = 0.; work = native_work = 0
        sync(); t = time.perf_counter()
        try:
            for start in range(0, len(data["targets"]), 64):
                check(); x, lengths, targets = batch(data, np.arange(start, min(start + 64, len(data["targets"]))))
                logp = model(x, lengths)
                if not bool(torch.isfinite(logp).all()): raise FloatingPointError("nonfinite evaluation")
                nll += float(F.nll_loss(logp, targets, reduction="sum")); correct += int((logp.argmax(-1) == targets).sum()); total += len(targets)
                work += model.proxy(len(x), x.shape[1])
                if args.arm.startswith("block_"):
                    native = model(x, lengths, native=True); native_correct += int((native.argmax(-1) == targets).sum())
                    native_work += model.proxy(len(x), x.shape[1], True)
            sync()
        finally: model.train(was)
        result = dict(rows=total, terminal_NLL=nll / total, accuracy=correct / total,
            native_hard_CALL_accuracy=native_correct / total if args.arm.startswith("block_") else None,
            forward_MAC_proxy=work, native_forward_MAC_proxy=native_work, wall_seconds=time.perf_counter() - t)
        costs["evaluation_forward_MAC_proxy"] += work + native_work
        costs["evaluation_wall_seconds"] += result["wall_seconds"]
        costs["evaluation_examples"] += total * (2 if args.arm.startswith("block_") else 1)
        return result
    def checkpoint(name):
        path = args.out / name; t = time.perf_counter()
        state = dict(schema="real-feature-neural-development-v1", args=vars(args).copy(), metadata=metadata,
            model_state=model.state_dict(), model_core_config=model.body.configuration() if args.arm.startswith("block_") else None,
            optimizer=optimizer.state_dict(), completed_updates=completed, batch_sha256=batch_sha,
            rng=dict(python=random.getstate(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []),
            source_sha256=source_hashes(True), costs=copy.deepcopy(costs),
            scope="single10class development; no formalPrimary or exacttrainingresume claim")
        # Owned checkpoint contains safe primitive metadata/paths and tensors.
        state["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in state["args"].items()}
        with path.open("xb") as dest: torch.save(state, dest)
        if path.stat().st_size > 100 << 20: raise RuntimeError("owned checkpoint cap")
        receipt = dict(file=name, sha256=sha(path), bytes=path.stat().st_size, completed_updates=completed, wall_seconds=time.perf_counter() - t)
        costs["checkpoint_wall_seconds"] += receipt["wall_seconds"]; receipts.append(receipt); emit("actual_checkpoint", **receipt)
        return path
    model = optimizer = None; completed = 0; curves = []; receipts = []; metadata = None
    costs = dict(optimizer_updates=0, training_examples=0, training_feature_tokens=0, training_forward_MAC_proxy=0,
        training_wall_seconds=0., training_CPU_seconds=0., batch_assembly_wall_seconds=0., evaluation_forward_MAC_proxy=0,
        evaluation_examples=0, evaluation_wall_seconds=0., checkpoint_wall_seconds=0., reload_wall_seconds=0.)
    try:
        if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested GPU unavailable; no silent fallback")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
        metadata = json.loads((args.data / "metadata.json").read_text())
        if metadata["schema"] != "real-feature-development-v1" or metadata["selected_classes"] != list(range(10)): raise ValueError("frozen10class development metadata required")
        train_data, valid = load("train"), load("valid")
        batches = np.random.default_rng(np.random.SeedSequence([args.seed, metadata["public_output_classes"], 300, 0x46454154])).integers(0, len(train_data["targets"]), size=(300, 64), dtype=np.int64)
        with (args.out / "common-batch-indices.npz").open("xb") as dest: np.savez_compressed(dest, indices=batches)
        batch_sha = sha(args.out / "common-batch-indices.npz")
        model = build_model(args.arm, metadata, device); optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        emit("configuration", args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            metadata_sha256=sha(args.data / "metadata.json"), source_sha256=source_hashes(True), batch_sha256=batch_sha,
            resources=resources(), fixed_budget=dict(updates=300, batch=64, optimizer="Adam", lr=.003, clip_norm=1.),
            no_test_selection=True, formal_Primary=False, validity="implementation/info/prospective protocol; never achievedaccuracy threshold",
            supervision="terminal originalclass only", Block_native_scope="hardCALL only; continuous state/operations/readout", torch_version=str(torch.__version__))
        for step in range(301):
            check()
            if step % 50 == 0:
                report = dict(step=step, TRAIN=evaluate(train_data), VALID=evaluate(valid)); curves.append(report); emit("TRAIN_VALID_curve", **report)
                if step and step < 300: checkpoint(f"{args.arm}-step-{step:03d}.pt")
            if step == 300: break
            bt = time.perf_counter(); x, lengths, targets = batch(train_data, batches[step]); costs["batch_assembly_wall_seconds"] += time.perf_counter() - bt
            model.train(); sync(); t = time.perf_counter(); cpu = time.process_time()
            optimizer.zero_grad(set_to_none=True); logp = model(x, lengths); loss = F.nll_loss(logp, targets)
            if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite trainloss")
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()): raise FloatingPointError("nonfinite parameters")
            sync(); completed = step + 1
            costs["training_wall_seconds"] += time.perf_counter() - t; costs["training_CPU_seconds"] += time.process_time() - cpu
            costs["training_forward_MAC_proxy"] += model.proxy(len(x), x.shape[1]); costs["optimizer_updates"] += 1
            costs["training_examples"] += len(x); costs["training_feature_tokens"] += int(lengths.sum())
            emit("actual_update", step=completed, terminal_loss=float(loss.detach()), train_batch_accuracy=float((logp.argmax(-1)==targets).float().mean()), preclip_gradient_norm=float(norm), cumulative_costs=costs)
        final_path = checkpoint(f"{args.arm}-frozen-final.pt"); emit("ALL_TRAINING_FROZEN_BEFORE_TEST", completed_updates=completed, checkpoint=str(final_path))
        # First loading/scoring any TEST cache happens after the final disk freeze.
        test = load("test"); final = dict(TRAIN=curves[-1]["TRAIN"], VALID=curves[-1]["VALID"], TEST=evaluate(test))
        rt = time.perf_counter(); payload = torch.load(final_path, map_location="cpu", weights_only=True)
        other = build_model(args.arm, metadata, device); other.load_state_dict(payload["model_state"], strict=True)
        same = all(torch.equal(v, other.state_dict()[k]) for k,v in model.state_dict().items())
        x, lengths, _ = batch(valid, np.arange(min(64, len(valid["targets"])))); model.eval(); other.eval()
        with torch.no_grad(): equal_outputs = torch.equal(model(x,lengths), other(x,lengths))
        sync(); costs["reload_wall_seconds"] += time.perf_counter() - rt
        if not same or not equal_outputs: raise RuntimeError("actual neural reload mismatch")
        reload = dict(actual_tensor_values_equal=same, actual_sameinput_output_equal=equal_outputs, checked_rows=len(x),
            forward_examples=2*len(x), forward_MAC_proxy=2*model.proxy(len(x),x.shape[1]), exact_training_resume_tested=False)
        result = dict(status="completed300updates_realfeature_development", arm=args.arm, dataset=metadata["dataset"],
            completed_updates=completed, final=final, curves=curves, resources=resources(), costs=costs,
            checkpoints=receipts, reload_audit=reload, batch_sha256=batch_sha, metadata_sha256=sha(args.data/"metadata.json"),
            formal_Primary=False, scope="first10classes static dev; fullpublichead; notcontinual/fullbenchmark",
            driver_wall_seconds=time.perf_counter()-started,
            cost_scope="nestedtimings do notsum; forwardMACproxies excludebackward/optimizer; shared featurecache/extraction charged separately; concurrency preventsisolatedthroughput claim")
        write_json(args.out / "summary.json", result); emit("completed", final=final, resources=result["resources"])
        print(json.dumps(dict(status=result["status"], arm=args.arm, completed_updates=completed, out=str(args.out))))
    except Exception as error:
        emit("failure", error=f"{type(error).__name__}: {error}", completed_updates=completed)
        (args.out / "exception.txt").write_text(traceback.format_exc())
        write_json(args.out / "failure.json", dict(error=f"{type(error).__name__}: {error}", completed_updates=completed, costs=costs, curves=curves, checkpoints=receipts))
        if model is not None:
            with (args.out / "partial-actual-neural-state.pt").open("xb") as dest: torch.save(dict(model_state=model.state_dict(), completed_updates=completed), dest)
        raise
    finally: stream.close()


def main():
    p = argparse.ArgumentParser(); commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare"); prep.add_argument("--dataset", choices=("speech", "cifar"), required=True)
    prep.add_argument("--raw-root", type=Path, required=True); prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=0); prep.add_argument("--max-seconds", type=float, default=600)
    run = commands.add_parser("train"); run.add_argument("--data", type=Path, required=True); run.add_argument("--out", type=Path, required=True)
    run.add_argument("--arm", choices=("block_dynamic", "block_static", "block_detached", "gru", "transformer", "cnn"), required=True)
    run.add_argument("--device", default="cuda:0"); run.add_argument("--seed", type=int, default=0); run.add_argument("--max-seconds", type=float, default=850)
    args = p.parse_args()
    if not 1 <= args.max_seconds <= 850: p.error("finite <=850second internal bound required")
    if args.command == "prepare": prepare(args)
    else: train(args)


if __name__ == "__main__": main()
