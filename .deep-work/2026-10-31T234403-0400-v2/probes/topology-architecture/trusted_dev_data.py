"""ROOT-RUN trusted public-data preparation; no network or work at import.

Development data, not a completed Primary benchmark. Uses only Python stdlib
for fetch/extraction and existing NumPy for fixed indexes/token arrays. No
pickle, downloaded code, tar.extractall, dependency install, or training.
Run in a network-enabled DATA coordinator, not an experimental world process.
All archive input is hash-checked before bounded manual regular-file extraction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import tarfile
import time
import urllib.request
from urllib.parse import urlparse
import uuid


SPEECH_WORDS = tuple(sorted("backward bed bird cat dog down eight five follow forward four go happy house learn left marvin nine no off on one right seven sheila six stop three tree two up visual wow yes zero".split()))
ALLOWED_HOSTS = {"www.cs.toronto.edu", "cave.cs.toronto.edu", "storage.googleapis.com",
                 "www.gutenberg.org", "gutenberg.org"}
SOURCE_REFERENCES = {
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "speech35": "https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/speech_commands/checksums.tsv",
    "speech_split": "https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/datasets/speechcommands.py",
    "speech_paper": "https://arxiv.org/html/1804.03209v1",
    "shakespeare": "https://www.gutenberg.org/ebooks/100",
}


@dataclass(frozen=True)
class Asset:
    url: str
    filename: str
    maximum_bytes: int
    minimum_bytes: int
    sha256: str | None = None
    md5: str | None = None
    exact_bytes: int | None = None


ASSETS = {
    "cifar100": Asset("https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
                       "cifar-100-binary.tar.gz", 200 << 20, 100 << 20,
                       md5="03b5dce01913d631647c71ecec9e9cb8"),
    "speech35": Asset("https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                       "speech_commands_v0.02.tar.gz", 2500 << 20, 2000 << 20,
                       sha256="af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
                       exact_bytes=2428923189),
    "shakespeare": Asset("https://www.gutenberg.org/cache/epub/100/pg100.txt",
                          "pg100.txt", 16 << 20, 500000),
}


def check_time(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("bounded data-coordinator wall deadline reached")


def hash_file(path: Path, algorithm: str = "sha256", deadline: float | None = None) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            if deadline is not None:
                check_time(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checked_url(url: str) -> str:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("only credential-free HTTPS official-source URLs are allowed")
    return url


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def verify_asset(path: Path, spec: Asset, deadline: float, text_sha256: str | None = None) -> dict:
    size = path.stat().st_size
    if not spec.minimum_bytes <= size <= spec.maximum_bytes or (spec.exact_bytes is not None and size != spec.exact_bytes):
        raise ValueError(f"unexpected official asset size: {size}")
    actual_sha = hash_file(path, deadline=deadline)
    expected_sha = spec.sha256 or text_sha256
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("SHA256 mismatch; source bytes preserved, no extraction")
    actual_md5 = hash_file(path, "md5", deadline) if spec.md5 else None
    if spec.md5 and actual_md5 != spec.md5:
        raise ValueError("official published MD5 mismatch; no extraction")
    return {"bytes": size, "sha256": actual_sha, "md5": actual_md5,
            "expected_sha256": expected_sha, "expected_md5": spec.md5,
            "publisher_digest_available": bool(spec.sha256 or spec.md5),
            "verification": "published digest verified" if spec.sha256 or spec.md5 else
                "official HTTPS source; actual SHA frozen; no publisher digest located"}


def fetch(spec: Asset, destination: Path, deadline: float, *, resume: bool,
          text_sha256: str | None = None) -> dict:
    started = time.perf_counter()
    if destination.exists():
        receipt = verify_asset(destination, spec, deadline, text_sha256)
        return {**receipt, "url": spec.url, "reused": True, "new_network_bytes": 0,
                "wall_seconds": time.perf_counter() - started}
    partial = destination.with_name(destination.name + ".part")
    partial_exists = partial.exists()
    offset = partial.stat().st_size if partial_exists else 0
    if partial_exists and not resume:
        raise FileExistsError(f"partial source retained at {partial}; use --resume for digest-pinned archives")
    if offset and not (spec.sha256 or spec.md5 or text_sha256):
        raise ValueError("resuming mutable text without an expected digest is forbidden")
    if offset > spec.maximum_bytes:
        raise ValueError("partial file exceeds asset bound")
    if spec.exact_bytes is not None and offset == spec.exact_bytes:
        receipt = verify_asset(partial, spec, deadline, text_sha256)
        partial.rename(destination)
        return {**receipt, "url": spec.url, "reused": True,
                "verified_complete_partial": True, "new_network_bytes": 0,
                "wall_seconds": time.perf_counter() - started}
    headers = {"User-Agent": "Prometheus-Research-Data/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), OfficialRedirect(),
                                        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request = urllib.request.Request(checked_url(spec.url), headers=headers)
    check_time(deadline)
    received = 0
    with opener.open(request, timeout=min(30., max(1., deadline - time.monotonic()))) as response:
        checked_url(response.geturl())
        if offset:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                raise ValueError("server did not honor safe Range resume; existing partial retained")
        elif response.status != 200:
            raise ValueError("unexpected HTTP status")
        content_length = response.headers.get("Content-Length")
        if content_length and offset + int(content_length) > spec.maximum_bytes:
            raise ValueError("HTTP size exceeds asset bound")
        metadata = {"requested_url": spec.url, "final_url": response.geturl(),
                    "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
        with partial.open("ab" if partial_exists else "xb") as stream:
            while chunk := response.read(1 << 20):
                check_time(deadline)
                received += len(chunk)
                if offset + received > spec.maximum_bytes:
                    raise ValueError("stream exceeds asset bound")
                stream.write(chunk)
    receipt = verify_asset(partial, spec, deadline, text_sha256)
    if destination.exists():
        raise FileExistsError("a concurrent source writer appeared")
    partial.rename(destination)
    return {**receipt, **metadata, "reused": False, "resumed_bytes": offset,
            "new_network_bytes": received, "wall_seconds": time.perf_counter() - started}


def safe_member_name(name: str) -> str:
    if "\\" in name or "\x00" in name or len(name) > 512:
        raise ValueError("invalid archive member name")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("archive path traversal rejected")
    return str(path)


def allowed_member(name: str, dataset: str) -> bool:
    parts = PurePosixPath(name).parts
    if dataset == "cifar100":
        return len(parts) == 2 and parts[0] == "cifar-100-binary" and parts[1] in {
            "train.bin", "test.bin", "fine_label_names.txt", "coarse_label_names.txt"}
    if len(parts) == 1:
        return parts[0] in {"validation_list.txt", "testing_list.txt", "README.md", "LICENSE"}
    return len(parts) == 2 and ((parts[0] in SPEECH_WORDS and parts[1].endswith(".wav"))
                              or (parts[0] == "_background_noise_" and (parts[1].endswith(".wav") or parts[1] == "README.md")))


def extract_data(archive: Path, destination: Path, dataset: str, deadline: float) -> dict:
    complete = destination / "extraction-complete.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        if result["archive_sha256"] != hash_file(archive, deadline=deadline):
            raise ValueError("existing extracted source has different archive provenance")
        return result
    destination.mkdir(parents=True, exist_ok=False)
    maximum_total = (6 << 30) if dataset == "speech35" else (200 << 20)
    maximum_member = (8 << 20) if dataset == "speech35" else (160 << 20)
    maximum_count = 120000 if dataset == "speech35" else 32
    total = written = count = skipped = 0
    seen = set()
    log_path = destination / "extracted-members.jsonl"
    with log_path.open("x") as log, tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            check_time(deadline)
            name = safe_member_name(member.name)
            count += 1
            if count > maximum_count or member.size < 0 or member.size > maximum_member:
                raise ValueError("archive member/count bound exceeded")
            total += member.size
            if total > maximum_total or (not member.isdir() and not member.isfile()):
                raise ValueError("archive volume/link/device/special entry rejected")
            if member.isdir():
                continue
            if name in seen:
                raise ValueError("duplicate archive file rejected")
            seen.add(name)
            if not allowed_member(name, dataset):
                skipped += 1
                continue
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("extraction escaped its fresh root")
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("regular member missing data")
            digest = hashlib.sha256();actual = 0
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as output:
                while chunk := source.read(1 << 20):
                    check_time(deadline)
                    output.write(chunk);digest.update(chunk);actual += len(chunk)
                    if actual > member.size:
                        raise ValueError("member exceeds declared size")
            if actual != member.size:
                raise ValueError("truncated archive member")
            log.write(json.dumps({"path": name, "bytes": actual, "sha256": digest.hexdigest()}) + "\n")
            log.flush();written += actual
    result = {"archive_sha256": hash_file(archive, deadline=deadline), "archive_members": count,
              "extracted_bytes": written, "skipped_nondata_files": skipped,
              "member_manifest": str(log_path), "member_manifest_sha256": hash_file(log_path, deadline=deadline),
              "security": "new regular files only; no links/devices/mode restoration/pickle/code"}
    write_json(complete, result)
    return result


def prepare_cifar(raw: Path, out: Path, seed: int, deadline: float) -> dict:
    import numpy as np
    source = raw / "cifar-100-binary"
    for name, count in (("train.bin", 50000), ("test.bin", 10000)):
        path = source / name
        if path.stat().st_size != count * 3074:
            raise ValueError("official binary CIFAR record count mismatch")
    records = np.memmap(source / "train.bin", mode="r", dtype=np.uint8, shape=(50000, 3074))
    labels = np.asarray(records[:, 1]);test = np.memmap(source / "test.bin", mode="r", dtype=np.uint8, shape=(10000, 3074))
    if not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 500)) or not np.array_equal(np.bincount(test[:, 1], minlength=100), np.full(100, 100)):
        raise ValueError("CIFAR fine-label counts differ from official source")
    names = (source / "fine_label_names.txt").read_text().splitlines()
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError("CIFAR class names invalid")
    rng = np.random.default_rng(seed);train, valid = [], []
    for label in range(100):
        check_time(deadline);indices = rng.permutation(np.flatnonzero(labels == label))
        valid.extend(indices[:50]);train.extend(indices[50:])
    out.mkdir(parents=True, exist_ok=False)
    for name, values in (("train", sorted(train)), ("valid", sorted(valid)), ("test", range(10000))):
        np.save(out / f"{name}-indices.npy", np.asarray(values, dtype=np.int64), allow_pickle=False)
    return {"dataset": "cifar100_binary", "raw_root": str(source.resolve()), "classes": names,
            "numpy_version": np.__version__, "split_rng": "numpy.default_rng; frozen actual indexes supplied",
            "split_seed": seed, "counts": {"train": 45000, "valid": 5000, "test": 10000},
            "class_order": list(range(100)), "phase_classes": [list(range(i, i + 10)) for i in range(0, 100, 10)],
            "input_shape": [3, 32, 32], "inference_task_id": False,
            "files": {p.name: hash_file(p, deadline=deadline) for p in sorted(out.glob("*.npy"))}}


def list_split(path: Path) -> set[str]:
    result = set()
    for line in path.read_text().splitlines():
        name = safe_member_name(line.strip())
        if len(PurePosixPath(name).parts) != 2 or not allowed_member(name, "speech35"):
            raise ValueError("invalid official speech split member")
        if name in result:
            raise ValueError("duplicate official speech split entry")
        result.add(name)
    return result


def prepare_speech(raw: Path, out: Path, deadline: float) -> dict:
    valid, test = list_split(raw / "validation_list.txt"), list_split(raw / "testing_list.txt")
    if valid & test:
        raise ValueError("official speech split lists overlap")
    paths = sorted(p for label in SPEECH_WORDS for p in (raw / label).glob("*.wav"))
    if len(paths) != 105829:
        raise ValueError("expected all105829 official35-class utterances")
    observed = {p.relative_to(raw).as_posix() for p in paths}
    if not valid <= observed or not test <= observed:
        raise ValueError("official split references missing utterances")
    out.mkdir(parents=True, exist_ok=False)
    streams = {key: (out / f"{key}.jsonl").open("x") for key in ("train", "valid", "test")}
    counts = Counter();histograms = {key: Counter() for key in streams};speakers = {key: set() for key in streams}
    try:
        for path in paths:
            check_time(deadline);relative = path.relative_to(raw).as_posix()
            split = "valid" if relative in valid else "test" if relative in test else "train"
            label = path.parent.name;speaker = path.name.split("_nohash_", 1)[0]
            if "_nohash_" not in path.name:
                raise ValueError("unexpected speech speaker filename")
            streams[split].write(json.dumps({"path": relative, "label": SPEECH_WORDS.index(label), "speaker": speaker}) + "\n")
            counts[split] += 1;histograms[split][label] += 1;speakers[split].add(speaker)
    finally:
        for stream in streams.values():stream.close()
    if any(speakers[a] & speakers[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("speaker crossing detected in official speech splits")
    return {"dataset": "speech_commands_v0.02_35words", "raw_root": str(raw.resolve()), "classes": list(SPEECH_WORDS),
            "counts": dict(counts), "class_histograms": {k: dict(v) for k, v in histograms.items()},
            "phase_classes": [list(range(i, i + 5)) for i in range(0, 35, 5)],
            "sample_rate": 16000, "inference_task_id": False,
            "split_policy": "original validation_list/testing_list; remainder TRAIN; no12-label remapping",
            "official_split_hashes": {name: hash_file(raw / name, deadline=deadline) for name in ("validation_list.txt", "testing_list.txt")},
            "files": {p.name: hash_file(p, deadline=deadline) for p in sorted(out.glob("*.jsonl"))}}


def prepare_text(raw: Path, out: Path, deadline: float) -> dict:
    import numpy as np
    text = raw.read_bytes().decode("utf-8-sig", errors="strict")
    # Retain the raw ebook/terms. Only the unambiguous body enters tokenization.
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.IGNORECASE)
    if not start or not end or end.start() <= start.end():
        raise ValueError("ebook body markers absent or ambiguous; do not guess a corpus")
    body = text[start.end():end.start()]
    n = len(body);cut1, cut2 = n * 90 // 100, n * 95 // 100
    segments = {"train": body[:cut1], "valid": body[cut1:cut2], "test": body[cut2:]}
    characters = sorted(set(segments["train"]))
    if len(characters) >= 65535 or n < 500000:
        raise ValueError("bounded char corpus/vocabulary check failed")
    stoi = {char: i + 1 for i, char in enumerate(characters)}
    out.mkdir(parents=True, exist_ok=False)
    vocab = {"kind": "TRAIN-only Unicode characters", "unk_id": 0, "characters_by_id_from1": characters,
             "vocabulary_size": len(characters) + 1, "normalization": "UTF8 strict; BOM removed; body otherwise unchanged"}
    write_json(out / "vocab.json", vocab)
    unknown = {}
    for name, segment in segments.items():
        check_time(deadline);ids = np.fromiter((stoi.get(c, 0) for c in segment), dtype=np.uint16, count=len(segment))
        unknown[name] = int((ids == 0).sum());np.save(out / f"{name}.npy", ids, allow_pickle=False)
    return {"dataset": "shakespeare_project_gutenberg_100_char_dev", "raw_source": str(raw.resolve()),
            "body_characters": n, "raw_body_character_span": [start.end(), end.start()],
            "split_character_boundaries": [0, cut1, cut2, n], "split_policy": "contiguous90/5/5 before fitting vocabulary",
            "counts": {k: len(v) for k, v in segments.items()}, "unknown_tokens": unknown,
            "vocabulary_size": len(characters) + 1, "no_targets_cross_split": True,
            "metric": "next-character NLL and exp(totalNLL/number_of_targets); not word-level PPL or TinyShakespeare comparability",
            "files": {p.name: hash_file(p, deadline=deadline) for p in sorted(out.iterdir())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(ASSETS), default=["shakespeare", "cifar100", "speech35"])
    parser.add_argument("--seconds", type=float, default=1800.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--text-sha256", default=None, help="optional previously frozen SHA; never guessed")
    parser.add_argument("--min-free-gib", type=float, default=10.)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 7200 or not 0 <= args.min_free_gib <= 1000:
        raise ValueError("finite coordinator timeout/free-space requirement needed")
    if args.text_sha256 and not re.fullmatch(r"[a-f0-9]{64}", args.text_sha256):
        raise ValueError("expected text SHA256 must be64 lowercase hex characters")
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < args.min_free_gib * (1 << 30):
        raise OSError("insufficient requested free space; full selection requires about10GiB reserve")
    lock = args.root / ".trusted-data-preparation.lock"
    with lock.open("x") as stream:stream.write(str(os.getpid()))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    receipt = {"status": "started", "development_only": True, "run_id": run_id,
               "source_references": SOURCE_REFERENCES, "datasets": {}, "no_training": True,
               "coordinator_sha256": hash_file(Path(__file__)),
               "incomplete_extraction_policy": "partial trees preserved; inspect/quarantine before rerun, never silently overwrite"}
    started = time.perf_counter();deadline = time.monotonic() + args.seconds
    try:
        archives = args.root / "sources";archives.mkdir(exist_ok=True)
        prepared = args.root / "prepared";prepared.mkdir(exist_ok=True)
        for name in dict.fromkeys(args.datasets):
            check_time(deadline);out = prepared / name;manifest_path = out / "manifest.json"
            spec = ASSETS[name];asset_path = archives / spec.filename
            source = fetch(spec, asset_path, deadline, resume=args.resume, text_sha256=args.text_sha256 if name == "shakespeare" else None)
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text())
                if existing["source"]["sha256"] != source["sha256"]:
                    raise ValueError("prepared dataset source changed; choose a new root, do not overwrite a freeze")
                if name == "cifar100" and existing["split_seed"] != args.split_seed:
                    raise ValueError("existing fixed split seed differs")
                for filename, sha in existing["files"].items():
                    if hash_file(out / filename, deadline=deadline) != sha:
                        raise ValueError("prepared data/index hash mismatch")
                receipt["datasets"][name] = {"manifest": str(manifest_path), "reused": True, "source": source}
                continue
            if name == "shakespeare":
                info = prepare_text(asset_path, out, deadline)
            else:
                raw = args.root / "extracted" / name
                extraction = extract_data(asset_path, raw, name, deadline)
                info = prepare_cifar(raw, out, args.split_seed, deadline) if name == "cifar100" else prepare_speech(raw, out, deadline)
                info["extraction"] = extraction
            info.update(source=source, development_only=True)
            write_json(manifest_path, info)
            receipt["datasets"][name] = {"manifest": str(manifest_path), "manifest_sha256": hash_file(manifest_path, deadline=deadline), "counts": info["counts"], "source": source}
            print(json.dumps({"event": "dataset_prepared", "dataset": name, "manifest": str(manifest_path), "counts": info["counts"]}), flush=True)
        receipt["status"] = "prepared_verified_sources"
    except Exception as error:
        receipt.update(status="preparation_failed_or_incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["unverified_partial_downloads"] = {p.name: p.stat().st_size
            for p in (args.root / "sources").glob("*.part")}
        write_json(args.root / f"data-receipt-{run_id}.json", receipt)
        lock.unlink()


if __name__ == "__main__":
    main()
