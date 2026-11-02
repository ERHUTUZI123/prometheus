"""Two-stage Linux x86_64 GPU runner for inspected official Mamba compilation.

Current human override prioritizes capabilities and complete resource costs
against strong architectures; Monodromy is optional. This is a new derivative;
the old verified no-exec runner is unchanged.

Stage 1 runs ONLY a fixed SHA-pinned prologue file beside this runner. uid65534,
NO_NEW_PRIVS, Landlock filesystem restrictions, no-network seccomp, and resource
limits already apply before torch/CUDA import. It admits fixed root-owned,
SHA-pinned compiler ELF inodes and the necessary pinned ELF loader. The loader
could indirectly execute readable code: this trusted fixed phase is NOT an
adversarial compiler-purpose proof. It cannot run an arbitrary user script or
choose a prologue via argv. Its fixed random Mamba forward/backward, compiler
probes, time and caches are charged priming; no task data or optimizer is used.

Stage 2 first stacks exec/fork-denying seccomp with TSYNC across all threads,
then a zero-allow EXECUTE Landlock layer on the current/future threads. Existing
threads' exec is denied by TSYNC; Landlock has no TSYNC API. A root-supervisor
pipe gate kills/reaps every residual stage-1 process in the same immutable
process group before ACK. Only then may the inspected research script run.
Cache misses fail; no automatic return to compiler permissions is possible.

No shell/Python/cache/output inode receives Landlock EXECUTE directly. Forked
Python can compute in trusted stage 1, and Python/ctypes/dlopen/JIT remain code
execution mechanisms: this is NOT a policy for untrusted evolutionary programs.
GPU fraction limits the torch caching allocator, not arbitrary raw driver
allocations. AS/CPU are per-process, FSIZE per-file; NPROC per UID and external
wall bound remain. No credentials are accepted/copied/embedded.

Root, Linux x86_64, Landlock ABI>=4 required. No isolation fallback. Own-process
/proc/<PID>/task WRITE_FILE|TRUNCATE exception and bounded process-group cleanup
remain. No network is deferred until after trusted CUDA initialization.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import select
import runpy
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback


UID = GID = 65534
GIB = 1024 ** 3
LANDLOCK_CREATE, LANDLOCK_ADD, LANDLOCK_RESTRICT = 444, 445, 446
LL_EXEC, LL_WRITE, LL_READ, LL_READ_DIR = 1, 2, 4, 8
LL_REMOVE_DIR, LL_REMOVE_FILE = 1 << 4, 1 << 5
LL_MAKE_DIR, LL_MAKE_REG = 1 << 7, 1 << 8
LL_REFER, LL_TRUNCATE = 1 << 13, 1 << 14
LL_HANDLED_FS = (1 << 15) - 1  # ABI 3+ filesystem rights; ioctl is ABI 5.
LL_READ_TREE = LL_READ | LL_READ_DIR  # NEVER grants EXECUTE to a tree.
LL_OUT = (LL_READ_TREE | LL_WRITE | LL_REMOVE_DIR | LL_REMOVE_FILE
          | LL_MAKE_DIR | LL_MAKE_REG | LL_REFER | LL_TRUNCATE)
SYSTEM_TREES = ("/usr", "/lib", "/lib64", "/opt", "/proc", "/sys")
ETC_FILES = ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/nsswitch.conf",
             "/etc/passwd", "/etc/group", "/etc/localtime", "/etc/os-release",
             "/etc/hosts", "/etc/nvidia/nvidia-application-profiles-rc.d/10-container.conf")

# Fixed host inventory independently read by ROOT, 2026-11-02. No CLI extension.
# /usr/sbin/ldconfig is a shell wrapper and is intentionally NOT admitted.
COMPILER_TOOLS = (
    ("ldconfig", "/usr/sbin/ldconfig.real", "d879c9a8a41240aeb7fd0f3221116fc4cf41cc6c64c24e10541d1bafbe537ab1"),
    ("gcc", "/usr/bin/x86_64-linux-gnu-gcc-13", "1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26"),
    ("cc1", "/usr/libexec/gcc/x86_64-linux-gnu/13/cc1", "5d1679131184e2de4435b426eb264bf13472fe026db8e5c6bc97445814e8e2f4"),
    ("as", "/usr/bin/x86_64-linux-gnu-as", "21aff249b692b5c31a44007491f922dcb49f41323e362c57d2ada3f52eddb7f0"),
    ("collect2", "/usr/libexec/gcc/x86_64-linux-gnu/13/collect2", "4d1f341ae5b763b513258ee2812422a45e063c30a2f1924a0cf63d3699f3a158"),
    ("ld", "/usr/bin/x86_64-linux-gnu-ld.bfd", "e9ceb054c12207970f2726dfc07e9a66b411602748628baf27399f02a9bbb31b"),
    ("ptxas", "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas", "c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca"),
)
STAGE1_LOADER = ("loader", "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
                 "c20a2dc8917c755f02b94049356320fe1f62ac7d9f8994731f807d9df39302da")
PROLOGUE_LEAF = "official_mamba2_compiler_prologue.py"
PROLOGUE_SHA256 = "584e4b1e6b8cb42e698178cb64fd491666baa68c06e714ff026d1796a83025c3"
TRITON_LIBCUDA_DIRECTORY = "/lib/x86_64-linux-gnu"
ALLOWED_CLONE_FLAGS = (0x3d0f00, 0x1200011, 0x4111, 0x11)


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64),
                ("handled_access_net", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class SockProgram(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


def _libc():
    lib = ctypes.CDLL(None, use_errno=True)
    lib.syscall.restype = ctypes.c_long
    lib.prctl.restype = ctypes.c_int
    return lib


def _checked(result, operation):
    if result == -1:
        number = ctypes.get_errno()
        raise OSError(number, f"{operation}: {os.strerror(number)}")
    return result


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _platform_required():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("requires Linux x86_64; syscall numbers are architecture-specific")
    if os.getuid() != 0 or os.geteuid() != 0:
        raise RuntimeError("root supervisor required; refusing an unverified uid-drop substitute")


def _code_paths(script):
    if script.suffix != ".py" or not script.is_file():
        raise ValueError("--script must resolve to an inspected regular .py file")
    if script == Path("/root") or Path("/root") in script.parents:
        raise ValueError("scripts under /root are not admitted")
    directory = script.parent
    info = directory.stat()
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError("script directory must be root-owned and not group/world writable")
    paths = []
    for path in sorted(directory.glob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source helper must be a regular non-symlink file: {path.name}")
        info = path.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(f"source helper must be root-owned and read-only to child: {path.name}")
        paths.append(path)
    if script not in paths:
        raise ValueError("script was not included in inspected source-file set")
    return paths


def _root_immutable_path(path, *, directory=False, executable=True):
    """Canonical root-owned inode and every ancestor; no child-writable swap."""
    path = Path(path)
    resolved = path.resolve(strict=True)
    if not path.is_absolute() or path != resolved:
        raise ValueError(f"fixed path is not canonical: {path}")
    for current in (path, *path.parents):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(f"fixed path/ancestor must be root-owned and non-writable: {current}")
    info = path.stat()
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"fixed library path is not a directory: {path}")
    elif not stat.S_ISREG(info.st_mode) or (executable and not info.st_mode & 0o111):
        raise ValueError(f"compiler tool is not a regular executable file: {path}")
    return path


def _elf_interpreter(path):
    # Read-only ELF64 little-endian x86_64 inspection. Never executes readelf.
    with Path(path).open("rb") as stream:
        head = stream.read(64)
        if len(head) != 64 or head[:7] != b"\x7fELF\x02\x01\x01":
            raise ValueError(f"fixed tool is not ELF64 little-endian: {path}")
        if struct.unpack_from("<H", head, 18)[0] != 62:
            raise ValueError(f"fixed tool is not x86_64: {path}")
        offset = struct.unpack_from("<Q", head, 32)[0]
        size, count = struct.unpack_from("<HH", head, 54)
        if size != 56 or count > 1024:
            raise ValueError(f"unexpected ELF program-header table: {path}")
        for index in range(count):
            stream.seek(offset + index * size)
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError(f"truncated ELF header: {path}")
            entry = struct.unpack("<IIQQQQQQ", raw)
            if entry[0] == 3:  # PT_INTERP; only the independently pinned loader is admitted.
                if not 1 < entry[5] <= 4096:
                    raise ValueError(f"unexpected ELF interpreter size: {path}")
                stream.seek(entry[2]); value = stream.read(entry[5])
                if len(value) != entry[5] or not value.endswith(b"\0"):
                    raise ValueError(f"malformed ELF interpreter: {path}")
                return value[:-1].decode("utf-8", errors="strict")
    return None


def _compiler_paths():
    paths, record = [], []
    for role, name, expected in (*COMPILER_TOOLS, STAGE1_LOADER):
        path = _root_immutable_path(name)
        if _sha(path) != expected:
            raise ValueError(f"fixed compiler SHA mismatch: {role}")
        interpreter = _elf_interpreter(path)
        if interpreter is not None and Path(interpreter).resolve(strict=True) != Path(STAGE1_LOADER[1]):
            raise ValueError(f"tool interpreter differs from pinned loader: {role}")
        if role != "loader" and ("ld-linux" in path.name or path.name.startswith(("ld-", "python", "sh", "bash"))):
            raise ValueError(f"interpreter executable cannot be whitelisted: {path}")
        paths.append(path)
        record.append(dict(role=role, path=str(path), sha256=expected,
                           elf_interpreter=interpreter,
                           interpreter_execute_granted=(role != "loader" and interpreter is not None)))
    return paths, record


def _libcuda_directory():
    # /lib may be a distro symlink; resolve the one fixed directory, then check
    # its canonical ancestors. This does not forward an inherited env value.
    path = Path(TRITON_LIBCUDA_DIRECTORY).resolve(strict=True)
    if not any(path == Path(base) or Path(base) in path.parents for base in ("/usr", "/lib", "/lib64")):
        raise ValueError("fixed TRITON_LIBCUDA_PATH resolved outside system trees")
    _root_immutable_path(path, directory=True)
    library = (path / "libcuda.so.1").resolve(strict=True)
    _root_immutable_path(library, executable=False)
    return str(path)


def _prologue_path():
    # No CLI path or filename expansion. Both supervisor and root child verify
    # the same inspected file before dropping privilege.
    path = Path(__file__).resolve(strict=True).parent / PROLOGUE_LEAF
    _root_immutable_path(path, executable=False)
    if _sha(path) != PROLOGUE_SHA256:
        raise ValueError("fixed trusted compiler prologue SHA mismatch")
    return path


def _device_paths():
    found = set()
    for pattern in ("/dev/nvidia*", "/dev/nvidia-caps/nvidia-cap*"):
        for name in glob.glob(pattern):
            path = Path(name)
            if not (re.fullmatch(r"nvidia[0-9]+", path.name)
                    or path.name in {"nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"}
                    or path.parent == Path("/dev/nvidia-caps")):
                continue
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink GPU device: {name}")
            if stat.S_ISCHR(path.stat().st_mode):
                found.add(path)
    if not Path("/dev/nvidiactl") in found:
        raise RuntimeError("/dev/nvidiactl unavailable; no CPU or unsandboxed fallback")
    for name in glob.glob("/dev/dri/renderD*") + glob.glob("/dev/dri/card*"):
        path = Path(name)
        info = path.stat()
        vendor = Path(f"/sys/dev/char/{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}/device/vendor")
        if (not path.is_symlink() and stat.S_ISCHR(info.st_mode) and vendor.is_file()
                and vendor.read_text().strip() == "0x10de"):
            found.add(path)
    return sorted(found)


def _environment(out):
    # Construct from scratch: no SSH_AUTH_SOCK, tokens, cloud config, PYTHONPATH,
    # LD_PRELOAD, inherited HOME, or arbitrary environment forwarding.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "CC": "/usr/bin/x86_64-linux-gnu-gcc-13",
        "TRITON_PTXAS_PATH": "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas",
        "TRITON_LIBCUDA_PATH": _libcuda_directory(),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": str(out),
        "TMPDIR": str(out / "tmp"), "TMP": str(out / "tmp"),
        "TEMP": str(out / "tmp"), "TORCH_HOME": str(out / "torch"),
        "XDG_CACHE_HOME": str(out / "cache"),
        "CUDA_CACHE_PATH": str(out / "cuda-cache"),
        "TRITON_CACHE_DIR": str(out / "triton-cache"),
        "PYTHONPYCACHEPREFIX": str(out / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE", "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "CUDA_MODULE_LOADING": "LAZY", "PYTORCH_NVML_BASED_CUDA_CHECK": "0",
        "SANDBOX_OUT": str(out),
    }
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if library_path:
        parts = []
        for value in library_path.split(":"):
            if not value:
                raise ValueError("empty LD_LIBRARY_PATH component is not permitted")
            path = Path(value).resolve(strict=True)
            if not path.is_dir() or not any(path == Path(base) or Path(base) in path.parents
                                            for base in ("/usr", "/lib", "/lib64", "/opt")):
                raise ValueError("LD_LIBRARY_PATH must contain only existing system-library directories")
            parts.append(str(path))
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def _limits(args):
    def hard(kind, value):
        old_hard = resource.getrlimit(kind)[1]
        if old_hard != resource.RLIM_INFINITY:
            value = min(value, old_hard)
        resource.setrlimit(kind, (value, value))
    hard(resource.RLIMIT_CPU, math.ceil(args.timeout))
    hard(resource.RLIMIT_AS, args.address_space_gib * GIB)
    hard(resource.RLIMIT_FSIZE, args.file_limit_mib * 1024 ** 2)
    hard(resource.RLIMIT_NOFILE, 256)
    hard(resource.RLIMIT_NPROC, 128)  # Includes threads for this uid on Linux.
    hard(resource.RLIMIT_CORE, 0)


def _drop_identity(lib):
    _checked(lib.prctl(38, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
    _checked(lib.prctl(8, 0, 0, 0, 0), "PR_SET_KEEPCAPS(0)")
    os.setgroups([])
    os.setresgid(GID, GID, GID)
    os.setresuid(UID, UID, UID)
    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class CapData(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]
    header, caps = CapHeader(0x20080522, 0), (CapData * 2)()
    _checked(lib.syscall(126, ctypes.byref(header), ctypes.byref(caps)), "capset(clear)")
    _checked(lib.syscall(125, ctypes.byref(header), ctypes.byref(caps)), "capget(verify)")
    if any(item.effective or item.permitted or item.inheritable for item in caps):
        raise RuntimeError("capabilities did not clear")
    if os.getresuid() != (UID, UID, UID) or os.getresgid() != (GID, GID, GID) or os.getgroups():
        raise RuntimeError("uid/gid/group drop failed")
    _checked(lib.prctl(4, 0, 0, 0, 0), "PR_SET_DUMPABLE(0)")
    if _checked(lib.prctl(39, 0, 0, 0, 0), "PR_GET_NO_NEW_PRIVS") != 1:
        raise RuntimeError("NO_NEW_PRIVS verification failed")


def _landlock(lib, out, script, sources, devices, compiler_paths, prologue):
    abi = _checked(lib.syscall(LANDLOCK_CREATE, 0, 0, 1), "Landlock ABI query")
    if abi < 4:
        raise RuntimeError(f"Landlock ABI {abi} < 4; isolation not weakened")
    ruleset = Ruleset(LL_HANDLED_FS, 3)  # Handle and deny all TCP bind/connect.
    fd = _checked(lib.syscall(LANDLOCK_CREATE, ctypes.byref(ruleset),
                              ctypes.sizeof(ruleset), 0), "Landlock create")
    try:
        def add(path, rights):
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, path_fd)
                _checked(lib.syscall(LANDLOCK_ADD, fd, 1, ctypes.byref(rule), 0),
                         f"Landlock add {path}")
            finally:
                os.close(path_fd)
        for base in SYSTEM_TREES:
            if Path(base).exists():
                add(base, LL_READ_TREE)
        for tool in compiler_paths:
            add(tool, LL_EXEC | LL_READ)
        for name in ETC_FILES + tuple(glob.glob("/etc/ld.so.conf.d/*.conf")):
            if Path(name).exists():
                add(name, LL_READ)
        for directory in (Path("/etc/nvidia"), Path("/etc/nvidia/nvidia-application-profiles-rc.d")):
            if directory.exists():
                add(directory, LL_READ_DIR)
        # Directory enumeration alone does not grant reading arbitrary siblings.
        add(script.parent, LL_READ_DIR)
        for path in sources:
            add(path, LL_READ)
        add(prologue, LL_READ)
        add(out, LL_OUT)
        # Explicit current-user authorization: only this process's task subtree.
        # Kernel-created thread comm files need WRITE_FILE|TRUNCATE, not MAKE.
        own_tasks = Path(f"/proc/{os.getpid()}/task")
        add(own_tasks, LL_WRITE | LL_TRUNCATE)
        for directory in (Path("/dev"), Path("/dev/nvidia-caps"), Path("/dev/dri"), Path("/dev/dri/by-path")):
            if directory.exists():
                add(directory, LL_READ_DIR)  # Enumeration only, no generic device read/write.
        for device in devices + [Path("/dev/null")]:
            add(device, LL_READ | LL_WRITE)
        add("/dev/urandom", LL_READ)
        _checked(lib.syscall(LANDLOCK_RESTRICT, fd, 0), "Landlock restrict_self")
    finally:
        os.close(fd)
    return abi


def _deny_all_execute(lib):
    # Stacked layer has handled EXECUTE and zero allow rules. Applies to this
    # thread and future descendants; existing threads are sealed by TSYNC.
    ruleset = Ruleset(LL_EXEC, 0)
    fd = _checked(lib.syscall(LANDLOCK_CREATE, ctypes.byref(ruleset),
                              ctypes.sizeof(ruleset), 0), "Landlock no-exec create")
    try:
        _checked(lib.syscall(LANDLOCK_RESTRICT, fd, 0), "Landlock no-exec restrict_self")
    finally:
        os.close(fd)


def _seccomp(lib, *, strict=False):
    # Explicitly x86_64. Reject other ABIs and x32 before using syscall numbers.
    LD, JEQ, JGE, RET, AND = 0x20, 0x15, 0x35, 0x06, 0x54
    KILL, ALLOW, ERRNO = 0x80000000, 0x7FFF0000, 0x00050000
    deny = ERRNO | errno.EPERM
    instructions = [(LD, 0, 0, 4), (JEQ, 1, 0, 0xC000003E), (RET, 0, 0, KILL),
                    (LD, 0, 0, 0), (JGE, 0, 1, 0x40000000),
                    (RET, 0, 0, ERRNO | errno.ENOSYS)]
    blocked = {
        # All socket creation/network operations, including UNIX socketpairs.
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 288, 299, 307,
        # execve is checked by Landlock; execveat/anonymous exec routes stay denied.
        322, 319, 134,
        # Process-group escape, ptrace and cross-process memory/fds.
        109, 112, 101, 310, 311, 438,
        # Namespace/mount, kernel/key/perf/module and io_uring escape surfaces.
        155, 161, 165, 166, 167, 168, 169, 175, 176, 246, 248, 249, 250,
        272, 298, 300, 308, 313, 321, 323, 425, 426, 427, 428, 429, 430,
        431, 432, 433, 442,
        # No identity/capability changes after the verified drop.
        105, 106, 113, 114, 116, 117, 119, 122, 123, 126,
    }
    if strict:
        blocked.update((57, 58, 59))  # fork, vfork, execve; execveat always denied.
    for number in sorted(blocked):
        instructions.extend([(JEQ, 0, 1, number), (RET, 0, 0, deny)])
    # clone3 cannot inspect its pointed-to structure with classic BPF. Return
    # ENOSYS so libc falls back to the four explicitly admitted clone modes.
    instructions.extend([(JEQ, 0, 1, 435), (RET, 0, 0, ERRNO | errno.ENOSYS)])
    clone_body = [(LD, 0, 0, 20), (JEQ, 1, 0, 0), (RET, 0, 0, deny),
                  (LD, 0, 0, 16)]
    for flags in ((0x3d0f00,) if strict else ALLOWED_CLONE_FLAGS):
        clone_body.extend([(JEQ, 0, 1, flags), (RET, 0, 0, ALLOW)])
    clone_body.append((RET, 0, 0, deny))
    instructions.append((JEQ, 0, len(clone_body), 56))
    instructions.extend(clone_body)
    # Stage-1 fork/vfork inherit all confinement; stage 2 explicitly denies them.
    instructions.append((RET, 0, 0, ALLOW))
    filters = (SockFilter * len(instructions))(*(SockFilter(*row) for row in instructions))
    program = SockProgram(len(instructions), filters)
    result = _checked(lib.syscall(317, 1, 1, ctypes.byref(program)), "seccomp BPF TSYNC install")
    if result != 0:
        raise RuntimeError(f"seccomp TSYNC failed on thread {result}; refusing partial confinement")


def _child(args):
    _platform_required()
    script, out = Path(args.script).resolve(strict=True), Path(args.out).resolve(strict=True)
    sources = _code_paths(script)
    prologue = _prologue_path()
    compiler_paths, compiler_record = _compiler_paths()
    devices = [] if args.cpu_only else _device_paths()
    if out == script.parent or out in script.parents:
        raise ValueError("writable output cannot contain inspected source")
    for descriptor in (args._gate_request_fd, args._gate_ack_fd):
        if descriptor < 3:
            raise ValueError("supervisor transition pipe missing")
        os.set_inheritable(descriptor, False)
    lib = _libc()
    _limits(args)
    _drop_identity(lib)
    os.chdir(out)
    abi = _landlock(lib, out, script, sources, devices, compiler_paths, prologue)
    _seccomp(lib, strict=False)  # No network from BEFORE the first torch import.
    print(json.dumps({"event": "trusted_compiler_phase_ready", "uid": os.getuid(),
                      "gid": os.getgid(), "groups": os.getgroups(),
                      "landlock_abi": abi, "seccomp": "no_network_tsync",
                      "no_new_privs": True, "compiler_tools": compiler_record,
                      "prologue_sha256": PROLOGUE_SHA256,
                      "loader_risk": "indirect readable-code execution; trusted fixed prologue only",
                      "fork_purpose_separation": False,
                      "authorized_proc_write": {"path": f"/proc/{os.getpid()}/task",
                                                "rights": ["WRITE_FILE", "TRUNCATE"]},
                      "effective_rlimits": {
                          name: list(resource.getrlimit(getattr(resource, name)))
                          for name in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE",
                                       "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_CORE")
                      }}), flush=True)
    priming_started = time.perf_counter()
    priming_usage_before = resource.getrusage(resource.RUSAGE_SELF)
    priming_children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    import torch
    torch_import_seconds = time.perf_counter() - priming_started
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cuda_started = time.perf_counter()
    if not args.cpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable inside sandbox; no fallback")
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, 0)
    cuda_initialization_seconds = time.perf_counter() - cuda_started
    fixed = runpy.run_path(str(prologue), run_name="_fixed_compiler_prologue")
    result = fixed["run_prologue"](torch, out, cpu_only=args.cpu_only)
    if not result.get("completed"):
        raise RuntimeError("fixed compiler prologue not completed")
    priming_usage = resource.getrusage(resource.RUSAGE_SELF)
    priming_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["whole_compiler_stage_including_torch_cuda"] = dict(
        wall_seconds=time.perf_counter()-priming_started,
        torch_import_seconds=torch_import_seconds,
        cuda_initialization_seconds=cuda_initialization_seconds,
        cpu_self_user_seconds=priming_usage.ru_utime-priming_usage_before.ru_utime,
        cpu_self_system_seconds=priming_usage.ru_stime-priming_usage_before.ru_stime,
        cpu_children_user_seconds=priming_children.ru_utime-priming_children_before.ru_utime,
        cpu_children_system_seconds=priming_children.ru_stime-priming_children_before.ru_stime,
        cpu_peak_RSS_bytes=int(priming_usage.ru_maxrss*1024),
        cuda_memory_scope="torch allocator peak/reserved in prologue; raw driver/context memory not hard-capped or measured")
    with (out / "compiler-stage-total.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    # No untrusted script has run. The filter only tightens and is TSYNC across
    # all CUDA/library threads before requesting descendant cleanup.
    _seccomp(lib, strict=True)
    _deny_all_execute(lib)
    print(json.dumps({"event": "strict_filter_installed", "exec_fork": "denied_all_threads",
                      "seccomp": "enforced_tsync", "landlock_execute": "zero_allow_current_future_threads"}), flush=True)
    if os.write(args._gate_request_fd, b"C") != 1:
        raise RuntimeError("transition request pipe failed")
    os.close(args._gate_request_fd)
    if os.read(args._gate_ack_fd, 1) != b"S":
        raise RuntimeError("supervisor did not verify compiler-descendant cleanup")
    os.close(args._gate_ack_fd)
    print(json.dumps({"event": "sandbox_ready", "phase": "strict_research",
                      "compiler_descendants_cleanup_verified": True,
                      "uid": os.getuid(), "landlock_abi": abi,
                      "priming_costs": result}), flush=True)
    sys.path.insert(0, str(script.parent))
    script_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    sys.argv = [str(script), *script_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as finished:
        if finished.code not in (None, 0):
            raise
    if not args.cpu_only:
        torch.cuda.synchronize()


def _group_processes(pgid):
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rindex(")") + 2:].split()
            if int(fields[2]) == pgid:
                result[int(entry.name)] = fields[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def _reap_group(pgid):
    # Skip the leader, whose return code belongs exclusively to Popen.
    for pid in _group_processes(pgid):
        if pid == pgid:
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass


def _clear_compiler_descendants(process, deadline):
    # The leader has already installed strict TSYNC and cannot fork. Stop all
    # existing compiler descendants, then kill/reap until the group has only the
    # waiting leader. This gate is bounded and cannot acknowledge partial work.
    deadline = min(deadline, time.monotonic() + 5)
    seen = set()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("leader exited before strict transition gate")
        children = {pid: state for pid, state in _group_processes(process.pid).items()
                    if pid != process.pid}
        if not children:
            return sorted(seen)
        seen.update(children)
        for pid in children:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        # Killing a parent can reparent a nested child to this subreaper; repeat
        # scans until no same-group process remains, including adopted zombies.
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _reap_group(process.pid)
        time.sleep(0.01)
    raise RuntimeError("compiler descendants not cleared within transition budget")


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_session(process):
    # Run on success/failure/timeout: a normally exiting leader may leave a
    # compiler descendant. setsid/setpgid/CLONE_PARENT/namespaces remain denied.
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        process.poll()
        _reap_group(process.pid)
        if not _group_exists(process.pid):
            return process.wait(timeout=1)
        time.sleep(0.02)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        _reap_group(process.pid)
        if not _group_exists(process.pid):
            return process.wait(timeout=1)
        time.sleep(0.02)
    raise RuntimeError("process group remains after SIGKILL/reap; cleanup is NOT verified")


def _supervisor(args):
    supervisor_started = time.monotonic()
    supervisor_usage_before = resource.getrusage(resource.RUSAGE_SELF)
    _platform_required()
    script = Path(args.script).resolve(strict=True)
    sources = _code_paths(script)
    _, compiler_record = _compiler_paths()
    prologue = _prologue_path()
    # Adopt orphaned descendants so process-group emptiness can be verified.
    _checked(_libc().prctl(36, 1, 0, 0, 0), "PR_SET_CHILD_SUBREAPER")
    if not args.cpu_only:
        _device_paths()
    out = Path(args.out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError("--out must be a NEW directory; existing output is never overwritten")
    parent = out.parent.resolve(strict=True)
    if parent.stat().st_uid != 0 or parent.stat().st_mode & 0o022:
        raise ValueError("output parent must be root-owned and not group/world writable")
    out = parent / out.name
    if out == script.parent or out in script.parents or out == Path("/root") or Path("/root") in out.parents:
        raise ValueError("output must be outside /root and must not contain source")
    out.mkdir(mode=0o700)
    os.chown(out, UID, GID)
    for name in ("tmp", "torch", "cache", "cuda-cache", "triton-cache", "pycache"):
        path = out / name
        path.mkdir(mode=0o700)
        os.chown(path, UID, GID)
    env = _environment(out)
    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    runner = Path(__file__).resolve(strict=True)
    command = [sys.executable, "-s", "-B", str(runner), "--_child", "--script", str(script),
               "--out", str(out), "--timeout", str(args.timeout),
               "--address-space-gib", str(args.address_space_gib),
               "--file-limit-mib", str(args.file_limit_mib),
               "--gpu-fraction", str(args.gpu_fraction)]
    if args.cpu_only:
        command.append("--cpu-only")
    if args.script_args:
        command.extend(["--", *([*args.script_args][1:] if args.script_args[:1] == ["--"] else args.script_args)])
    record = {"status": "starting", "runner_sha256": _sha(runner), "script_sha256": _sha(script),
              "source_sha256": {path.name: _sha(path) for path in sources},
              "timeout_seconds": args.timeout, "address_space_gib": args.address_space_gib,
              "file_limit_mib": args.file_limit_mib, "gpu_allocator_fraction": args.gpu_fraction,
              "cpu_only": args.cpu_only,
              "compiler_tools": compiler_record,
              "compiler_manifest_cli_extensible": False,
              "dynamic_loader_execute": "trusted fixed prologue only; permanently denied before research",
              "prologue_path": str(prologue), "prologue_sha256": PROLOGUE_SHA256,
              "fork_purpose_separation": False,
              "resource_scope": "AS/CPU per-process; FSIZE per-file; NPROC per UID; supervisor wall-bound",
              "network_enforced_before_torch_import": True,
              "authorized_proc_write": {"scope": "child /proc/<PID>/task only",
                                        "rights": ["WRITE_FILE", "TRUNCATE"], "make_rights": False},
              "environment_keys": sorted(env), "out": str(out)}
    started = time.monotonic()
    request_read, request_write = os.pipe()
    ack_read, ack_write = os.pipe()
    command.extend(["--_gate-request-fd", str(request_write), "--_gate-ack-fd", str(ack_read)])
    # Hidden supervisor options MUST precede argparse.REMAINDER script args.
    if "--" in command:
        split = command.index("--")
        gates = command[-4:]; del command[-4:]
        command[split:split] = gates
    with open(out / "stdout.log", "xb", buffering=0) as stdout, open(out / "stderr.log", "xb", buffering=0) as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   env=env, close_fds=True, pass_fds=(request_write, ack_read),
                                   start_new_session=True, cwd=str(parent))
        os.close(request_write); os.close(ack_read)
        deadline = started + args.timeout
        gate_seen = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, args.timeout)
                if not gate_seen:
                    ready, _, _ = select.select([request_read], [], [], min(0.1, remaining))
                    if ready:
                        request = os.read(request_read, 1)
                        if request == b"C":
                            cleared = _clear_compiler_descendants(process, deadline)
                            os.write(ack_write, b"S")
                            gate_seen = True
                            record.update(strict_transition_verified=True,
                                          residual_compiler_pids_killed=cleared,
                                          strict_transition_wall_seconds=time.monotonic()-started)
                            os.close(request_read); request_read = -1
                            os.close(ack_write); ack_write = -1
                        elif request:
                            raise RuntimeError("invalid trusted transition request")
                else:
                    time.sleep(min(0.05, remaining))
                code = process.poll()
                if code is not None:
                    record.update(status="completed" if code == 0 and gate_seen else "failed", returncode=code)
                    _terminate_session(process)
                    record["process_group_cleanup_verified"] = True
                    break
        except subprocess.TimeoutExpired:
            record.update(status="wall_timeout", returncode=_terminate_session(process),
                          process_group_cleanup_verified=True)
        except BaseException:
            _terminate_session(process)
            raise
        finally:
            for descriptor in (request_read, ack_write):
                if descriptor >= 0:
                    os.close(descriptor)
    record["elapsed_seconds"] = time.monotonic() - started
    supervisor_usage = resource.getrusage(resource.RUSAGE_SELF)
    reaped_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(supervisor_wall_seconds_including_preflight=time.monotonic()-supervisor_started,
                  supervisor_cpu_user_seconds=supervisor_usage.ru_utime-supervisor_usage_before.ru_utime,
                  supervisor_cpu_system_seconds=supervisor_usage.ru_stime-supervisor_usage_before.ru_stime,
                  supervisor_reaped_children_cpu_user_seconds=reaped_usage.ru_utime,
                  supervisor_reaped_children_cpu_system_seconds=reaped_usage.ru_stime,
                  cost_additivity="reaped children totals include child processes; do not add them again to prologue child CPU")
    # The authoritative supervisor result lives outside the child's writable tree.
    receipt = parent / (out.name + ".runner-status.json")
    with open(receipt, "x", encoding="utf-8") as stream:
        os.chmod(receipt, 0o600)
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--address-space-gib", type=int, default=512)
    parser.add_argument("--file-limit-mib", type=int, default=100)
    parser.add_argument("--gpu-fraction", type=float, default=0.3)
    parser.add_argument("--cpu-only", action="store_true", help="explicit CPU execution; isolation remains enforced")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_gate-request-fd", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--_gate-ack-fd", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 900:
        parser.error("timeout must be finite and in (0, 900] seconds")
    if not 8 <= args.address_space_gib <= 512 or not 1 <= args.file_limit_mib <= 100:
        parser.error("address space must be 8..512 GiB; file size must be 1..100 MiB")
    if not math.isfinite(args.gpu_fraction) or not 0 < args.gpu_fraction <= 0.3:
        parser.error("gpu fraction must be in (0, 0.3]")
    try:
        if args._child:
            _child(args)
            return 0
        return _supervisor(args)
    except BaseException as error:
        print(json.dumps({"event": "runner_failure", "error_type": type(error).__name__,
                          "error": str(error), "isolation_fallback": False}), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
