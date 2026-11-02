"""Minimal existing-NumPy/Torch development readers; no IO/network at import.

Reader items are (input,target), never task/phase IDs. Phase class sets are
loader metadata only. Official sources/splits and TRAIN-only text vocabulary
come from trusted_dev_data.py; downloaded code and pickle are never loaded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import wave

import numpy as np


def manifest(root: Path, dataset: str) -> tuple[Path, dict]:
    folder = Path(root) / "prepared" / dataset
    return folder, json.loads((folder / "manifest.json").read_text())


class CIFAR100Binary:
    def __init__(self, root: Path, split: str, classes=None):
        folder, self.metadata = manifest(root, "cifar100")
        if split not in ("train", "valid", "test"):
            raise ValueError("fixed train/valid/test split required")
        count = 10000 if split == "test" else 50000
        file = "test.bin" if split == "test" else "train.bin"
        path = Path(root) / "extracted" / "cifar100" / "cifar-100-binary" / file
        if path.stat().st_size != count * 3074:
            raise ValueError("CIFAR binary source length mismatch")
        self.records = np.memmap(path, mode="r", dtype=np.uint8, shape=(count, 3074))
        self.indices = np.load(folder / f"{split}-indices.npy", allow_pickle=False)
        if self.indices.dtype != np.int64 or self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= count):
            raise ValueError("invalid fixed CIFAR indexes")
        if classes is not None:
            self.indices = self.indices[np.isin(self.records[self.indices, 1], list(classes))]

    def __len__(self):return len(self.indices)

    def __getitem__(self, index):
        record = self.records[int(self.indices[index])]
        return record[2:].reshape(3, 32, 32).astype(np.float32) / 255., int(record[1])


class SpeechCommands35:
    def __init__(self, root: Path, split: str, classes=None):
        folder, self.metadata = manifest(root, "speech35")
        if split not in ("train", "valid", "test"):
            raise ValueError("official train/valid/test split required")
        allowed = None if classes is None else set(classes)
        self.rows = []
        with (folder / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if allowed is None or row["label"] in allowed:self.rows.append(row)
        self.raw = (Path(root) / "extracted" / "speech35").resolve()

    def __len__(self):return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index];path = (self.raw / row["path"]).resolve()
        if not path.is_relative_to(self.raw):raise ValueError("unsafe indexed audio path")
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16000 or audio.getcomptype() != "NONE":
                raise ValueError("expected original16kHz mono PCM16 audio")
            count = audio.getnframes()
            if not 0 < count <= 16000:raise ValueError("word recording duration outside original <=1second scope")
            raw = audio.readframes(count)
        if len(raw) != count * 2:raise ValueError("truncated PCM payload")
        result = np.zeros(16000, dtype=np.float32)
        result[:count] = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.
        return result, int(row["label"])


def log_mel_16k(waveform, n_fft=512, window=400, hop=160, n_mels=40):
    """Fixed NumPy front end, no fitted/global/test statistics or augmentation.

    Output [frames,mels], natural log power. Padding/parameters must be common
    to competitors. Feature caching, if used, is extra charged data storage.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.shape != (16000,) or not 1 <= window <= n_fft <= 2048 or not 1 <= hop <= window or not 1 <= n_mels <= 128:
        raise ValueError("bounded fixed one-second feature configuration required")
    frames = np.lib.stride_tricks.sliding_window_view(x, window)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(window), n=n_fft, axis=-1)) ** 2
    hz_to_mel = lambda f:2595. * np.log10(1. + f / 700.)
    points = 700. * (10. ** (np.linspace(0., hz_to_mel(8000.), n_mels + 2) / 2595.) - 1.)
    frequencies = np.linspace(0., 8000., n_fft // 2 + 1)
    filters = np.stack([np.maximum(0., np.minimum((frequencies - left) / (mid - left),
                        (right - frequencies) / (right - mid))) for left, mid, right in zip(points, points[1:], points[2:])])
    return np.log(np.maximum(spectrum @ filters.T, 1e-10)).astype(np.float32)


class CharLMStream:
    def __init__(self, root: Path, split: str):
        folder, self.metadata = manifest(root, "shakespeare")
        if split not in ("train", "valid", "test"):
            raise ValueError("fixed contiguous split required")
        self.ids = np.load(folder / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        self.vocab = json.loads((folder / "vocab.json").read_text())
        if self.ids.ndim != 1 or self.ids.dtype != np.uint16 or len(self.ids) < 2:
            raise ValueError("bounded frozen character-token stream required")
        self.vocabulary_size = self.vocab["vocabulary_size"]
        if np.any(self.ids >= self.vocabulary_size):raise ValueError("token outside TRAIN-only vocabulary")

    def block(self, start: int, length: int):
        if start < 0 or length < 1 or start + length >= len(self.ids):
            raise IndexError("next-token input and target must stay inside their own split")
        return np.asarray(self.ids[start:start + length], dtype=np.int64), np.asarray(self.ids[start + 1:start + length + 1], dtype=np.int64)

    def iter_evaluation_blocks(self, length=256):
        """Each target except split's first token counted once; no boundary leak."""
        if not 1 <= length <= 8192:raise ValueError("bounded context length required")
        for start in range(0, len(self.ids) - 1, length):
            yield self.block(start, min(length, len(self.ids) - 1 - start))


class NextTokenMetrics:
    def __init__(self):self.nll_sum = 0.;self.tokens = 0;self.correct = 0

    def add(self, logits, targets):
        """Existing Torch tensors [B,T,V]/[B,T] or [T,V]/[T]; no training."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            if logits.shape[:-1] != targets.shape or targets.numel() == 0:
                raise ValueError("logit/next-token shape mismatch")
            value = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long(), reduction="sum"))
            if not math.isfinite(value):raise ValueError("nonfinite next-token loss")
            self.nll_sum += value;self.tokens += targets.numel()
            self.correct += int((logits.argmax(-1) == targets).sum())

    def result(self):
        if not self.tokens:raise ValueError("no evaluated next-token targets")
        mean = self.nll_sum / self.tokens
        return {"nll_sum": self.nll_sum, "target_tokens": self.tokens, "cross_entropy_nats": mean,
                "character_perplexity": math.exp(mean) if mean < 700 else None,
                "perplexity_overflow": mean >= 700,
                "next_token_accuracy": self.correct / self.tokens,
                "scope": "same frozen TRAIN-only character vocabulary and split; not word-level perplexity"}
