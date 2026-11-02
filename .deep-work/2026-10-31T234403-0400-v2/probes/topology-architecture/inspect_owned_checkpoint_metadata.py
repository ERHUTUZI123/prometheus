"""Data-only inspection of owned Torch ZIP metadata, without Torch/NumPy imports.

This is NOT a general checkpoint loader. Unknown pickle globals are rejected;
known array/tensor constructors are replaced with inert local descriptors. No
saved model code, parameter computation, or storage tensors are executed/read.
JSON byte counts describe compact, sorted, UTF-8 serialization of model_extra,
not Python RAM, Torch archive allocation, or a minimum representation.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import struct
import zipfile


class DType:
    def __init__(self, name, *_):
        self.name = name
        self.state = None

    def __setstate__(self, state):
        self.state = state


class Array:
    def __setstate__(self, state):
        version, self.shape, self.dtype, self.fortran, self.data = state
        if version not in (0, 1) or self.fortran:
            raise ValueError("unsupported ndarray state")

    def tolist(self):
        names = {'i8': 'q', 'i4': 'i', 'u8': 'Q', 'u4': 'I', 'f8': 'd',
                 'f4': 'f', 'u1': 'B', 'i1': 'b', 'b1': '?'}
        name = self.dtype.name
        endian = '<'
        if name[:1] in '<>=|':
            endian, name = name[0], name[1:]
        elif self.dtype.state and self.dtype.state[1] in '<>':
            endian = self.dtype.state[1]
        if name not in names or len(self.data) > 16 * 1024 * 1024:
            raise ValueError("unsupported or oversized array")
        fmt = ('<' if endian in '=|' else endian) + names[name]
        size = struct.calcsize(fmt)
        if len(self.data) % size:
            raise ValueError("bad array bytes")
        flat = [v[0] for v in struct.iter_unpack(fmt, self.data)]
        count = 1
        for d in self.shape:
            count *= d
        if count != len(flat) or count > 1_000_000:
            raise ValueError("bad array shape")
        def nest(shape, offset):
            if not shape:
                return flat[offset], offset + 1
            out = []
            for _ in range(shape[0]):
                value, offset = nest(shape[1:], offset)
                out.append(value)
            return out, offset
        return nest(self.shape, 0)[0]


class Tensor:
    def __init__(self, *args):
        self.description = args


class Storage:
    pass


def reconstruct(*_):
    return Array()


def encode(value, encoding):
    if encoding not in ('latin1', 'latin-1') or len(value) > 16 * 1024 * 1024:
        raise ValueError("unsupported encoded bytes")
    return value.encode(encoding)


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        known = {
            ('collections', 'OrderedDict'): OrderedDict,
            ('_codecs', 'encode'): encode,
            ('numpy', 'dtype'): DType,
            ('numpy', 'ndarray'): Array,
            ('numpy._core.multiarray', '_reconstruct'): reconstruct,
            ('numpy.core.multiarray', '_reconstruct'): reconstruct,
            ('torch._utils', '_rebuild_tensor_v2'): Tensor,
            ('torch', 'FloatStorage'): Storage,
            ('torch', 'ByteStorage'): Storage,
            ('pathlib', 'PosixPath'): PurePosixPath,
        }
        try:
            return known[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"rejected global {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or not pid or pid[0] != 'storage':
            raise pickle.UnpicklingError("rejected persistent object")
        return ('INERT_STORAGE_REFERENCE', *pid[2:])


def plain(value):
    if isinstance(value, Array):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"non-JSON metadata object: {type(value).__name__}")


def inspect(path):
    path = Path(path)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("checkpoint over cap")
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1 or z.getinfo(names[0]).file_size > 32 * 1024 * 1024:
            raise ValueError("unexpected archive layout")
        raw = z.read(names[0])
    state = DataOnlyUnpickler(io.BytesIO(raw)).load()
    if state.get('schema') != 'parameter-growth-neural-checkpoint-v1':
        raise ValueError("not the owned parameter-growth format")
    extra = plain(state['model_extra'])
    serialized = json.dumps(extra, sort_keys=True, separators=(',', ':'),
                            allow_nan=False).encode('utf-8')
    return dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_bytes=path.stat().st_size, stage=state['stage'], step=state['step'],
                model_extra_compact_json_bytes=len(serialized),
                model_extra_compact_json_sha256=hashlib.sha256(serialized).hexdigest(),
                model_extra_top_level_keys=sorted(extra) if isinstance(extra, dict) else None,
                model_extra_field_json_bytes={k: len(json.dumps(v, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode('utf-8'))
                    for k, v in extra.items()} if isinstance(extra, dict) else None,
                counting_scope='standalone compact sorted UTF-8 JSON; shared references duplicated by value; excludes all learned tensors and optimizer; not Python RAM or Torch archive bytes',
                inspection='restricted inert pickle descriptors; no Torch/NumPy/model imports; tensor storage not read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = [inspect(p) for p in args.checkpoint]
    with open(args.out, 'x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps([dict(stage=r['stage'], bytes=r['model_extra_compact_json_bytes']) for r in result]))
