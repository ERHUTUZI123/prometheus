"""Inference-only length diagnosis of OWN completed stage0 Transformers.

Read only task0 TRAIN and VALID terminal examples, and one or two actual
stage0-step600 Transformer64/96 checkpoints. No training, test split, new label,
architecture change or checkpoint overwrite. Fixed B64, internal <=110 seconds;
the coordinator supplies the existing GPU isolation and external120s limit.
MAIN=TOPOLOGY human-locked; Transformer primary; M0 unpassed; M3 ultimate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_projection_numeric import restore_parameter_layouts
from integrate_native_repair import Recorder, sha256, synchronize, MAX_FILE_BYTES
from parameter_transformer_baselines import GrowingTaskTransformer


BATCH = 64
GROUPS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2),
          ("3", 3, 3), ("4-8", 4, 8))
ARRAY_KEYS = ("tokens", "starts", "task_ids", "targets")


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(np.asarray(item.shape, dtype="<i8").tobytes())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def read_data(directory, split, n):
    # Literal names deliberately exclude every test split and future task.
    path = directory/f"task0-{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key], dtype=np.int64) for key in ARRAY_KEYS}
    size = len(data["starts"])
    if (data["tokens"].shape != (size, 18)
            or any(data[key].shape != (size,) for key in ARRAY_KEYS[1:])
            or size != (8192 if split == "train" else 512)
            or np.any(data["task_ids"] != 0)
            or np.any((data["tokens"] < 0) | (data["tokens"] > 6))
            or np.any((data["starts"] < 0) | (data["starts"] >= n))
            or np.any((data["targets"] < 0) | (data["targets"] >= n))):
        raise ValueError("actual task0 TRAIN/VALID schema or bounds differ")
    lengths = np.count_nonzero(data["tokens"], axis=1)
    if np.any(lengths > 8):
        raise ValueError("task0 TRAIN/VALID must use the frozen <=8-word domain")
    if not np.array_equal(data["tokens"] != 0, np.arange(18)[None, :] < lengths[:, None]):
        raise ValueError("actual diagnostic requires the original right-PAD format")
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        digest.update(key.encode())
        digest.update(np.asarray(data[key], dtype="<i8").tobytes())
    return data, lengths, dict(split=split, file=str(path), file_sha256=sha256(path),
        four_array_sha256=digest.hexdigest(), rows=size,
        length_counts={str(k): int(np.count_nonzero(lengths == k)) for k in range(9)})


@torch.no_grad()
def diagnose(model, data, lengths, split, tag, device, recorder):
    total = len(lengths)
    logits_all = np.empty((total, model.n_states), dtype=np.float32)
    nll_all = np.empty(total, dtype=np.float32)
    predictions = np.empty(total, dtype=np.int64)
    synchronize(device)
    began = time.perf_counter()
    for start in range(0, total, BATCH):
        recorder.check()
        end = min(start+BATCH, total)
        batch = {key: torch.as_tensor(value[start:end], device=device, dtype=torch.long)
                 for key, value in data.items()}
        logits = model(batch["tokens"], batch["starts"], batch["task_ids"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite diagnostic logits")
        nll = F.cross_entropy(logits, batch["targets"], reduction="none")
        pred = logits.argmax(-1)
        logits_all[start:end] = logits.detach().cpu().float().numpy()
        nll_all[start:end] = nll.detach().cpu().float().numpy()
        predictions[start:end] = pred.detach().cpu().numpy()
        recorder.emit("actual_inference_batch", arm=tag, split=split,
            first_source_row=start, last_source_row_exclusive=end, rows=end-start,
            terminal_CE_sum=float(nll.sum().item()),
            correct=int((pred == batch["targets"]).sum().item()),
            actual_word_length_counts={str(k): int(np.count_nonzero(lengths[start:end] == k))
                                      for k in range(9)})
    synchronize(device)
    inference_seconds = time.perf_counter()-began
    correct = predictions == data["targets"]
    bins = {}
    for label, low, high in GROUPS:
        selected = (lengths >= low) & (lengths <= high)
        count = int(np.count_nonzero(selected))
        bins[label] = dict(rows=count, correct=int(correct[selected].sum()),
            accuracy=float(correct[selected].mean()) if count else None,
            terminal_CE=float(nll_all[selected].astype(np.float64).mean()) if count else None)
    raw_path = recorder.out/f"{tag}-{split}-raw-inference.npz"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    np.savez_compressed(raw_path, source_rows=np.arange(total, dtype=np.int64),
        lengths=lengths, starts=data["starts"], targets=data["targets"],
        predictions=predictions, terminal_NLL=nll_all, logits=logits_all)
    if raw_path.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError("diagnostic raw output exceeds existing file limit")
    result = dict(rows=total, correct=int(correct.sum()), accuracy=float(correct.mean()),
        terminal_CE=float(nll_all.astype(np.float64).mean()), by_length=bins,
        inference_seconds=inference_seconds, batches=(total+BATCH-1)//BATCH,
        raw_file=raw_path.name, raw_sha256=sha256(raw_path), raw_bytes=raw_path.stat().st_size)
    recorder.emit("length_diagnostic_result", arm=tag, split=split, **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=110.)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    if (args.out.exists() or not 1 <= len(args.checkpoint) <= 2
            or not 0 < args.max_seconds <= 110):
        parser.error("new output, one/two owned checkpoints and <=110seconds required")
    recorder = Recorder(args.out, args.max_seconds)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required; no silent CPU fallback")
        device = torch.device(args.device)
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.3, device)
        recorder.emit("configuration", MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_PASSED", M3="ultimate",
            scope="existing stage0 Transformer checkpoints; TRAIN/VALID length-stratified inference only",
            batch_size=BATCH, gradient_updates=0, new_external_labels=0,
            test_splits_read=False, selected_repair_or_checkpoint=False,
            torch_version=torch.__version__, CUDA_version=torch.version.cuda,
            device_name=torch.cuda.get_device_name(device),
            checkpoint_paths=[str(path) for path in args.checkpoint])
        results, tags, data_cache = [], set(), {}
        for path in args.checkpoint:
            recorder.check()
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("owned checkpoint missing or exceeds existing100MiB limit")
            source = torch.load(path, map_location="cpu", weights_only=False)
            saved = source.get("args", {})
            tag, n = saved.get("arm"), saved.get("scale")
            if (source.get("schema") != "parameter-growth-neural-checkpoint-v1"
                    or tag not in ("transformer64", "replay64", "transformer96", "replay96")
                    or n not in (8, 16) or source.get("stage") != 0 or source.get("step") != 600
                    or not source.get("stage_training_complete") or tag in tags):
                raise ValueError("completed actual stage0-step600 Transformer64/96 checkpoint required")
            tags.add(tag)
            if n not in data_cache:
                data_cache[n] = {split: read_data(args.datasets, split, n) for split in ("train", "valid")}
            model = GrowingTaskTransformer(n, vocab=6, n_tasks=6, max_len=18,
                d_model=96 if "96" in tag else 64).to(device)
            model.load_state_dict(source["model"], strict=True)
            layouts = restore_parameter_layouts(model, source["model"])
            model.eval()
            original_hash = state_hash(source["model"])
            if state_hash(model.state_dict()) != original_hash:
                raise RuntimeError("exact saved model/buffer values failed restoration")
            recorder.emit("actual_checkpoint_restored", arm=tag, checkpoint=str(path),
                checkpoint_sha256=sha256(path), parameter_layouts=layouts,
                source_state_sha256=original_hash, all_parameter_and_buffer_bits_equal=True,
                parameter_count=sum(p.numel() for p in model.parameters()),
                source_stage=0, source_step=600, no_optimizer_or_training_execution=True)
            measured = {}
            for split, (data, lengths, provenance) in data_cache[n].items():
                recorder.emit("actual_dataset_read", arm=tag, **provenance)
                measured[split] = diagnose(model, data, lengths, split, tag, device, recorder)
            if state_hash(model.state_dict()) != original_hash:
                raise RuntimeError("inference modified actual checkpoint model state")
            result = dict(arm=tag, scale=n, checkpoint_sha256=sha256(path),
                unchanged_actual_state_sha256=original_hash, splits=measured,
                parameters=sum(p.numel() for p in model.parameters()),
                actual_parameter_state_unchanged=True)
            results.append(result)
            del model, source
            torch.cuda.empty_cache()
        recorder.json("summary.json", dict(status="ACTUAL_CHECKPOINT_LENGTH_DIAGNOSIS_COMPLETE",
            results=results, gradient_updates=0, new_labels=0, test_splits_read=False,
            wall_seconds=time.perf_counter()-recorder.started, M0="NOT_PASSED"))
    except BaseException as error:
        recorder.emit("run_failed", error=type(error).__name__, reason=str(error))
        recorder.json("failure.json", dict(error=type(error).__name__, reason=str(error),
            gradient_updates=0, M0="NOT_PASSED"))
        raise
    finally:
        recorder.stream.close()


if __name__ == "__main__":
    main()
