"""Linux x86_64 GPU sandbox launcher; standard-library supervisor.

Same V2: topology MAIN human-locked, M0 not passed, M3 final ambition.
No credentials are accepted, copied, logged, or embedded by this module.

Requires root for the supervisor and Landlock ABI >= 4. Fails closed on setup
errors. The supervisor launches exactly one current-Python child; after uid/gid
drop, resource limits, Landlock, and seccomp, that child uses runpy, not execve.
The child imports existing torch only after confinement to set its normal CUDA
caching-allocator budget. That budget is not a hard cap on arbitrary raw driver
allocations. RLIMIT_AS bounds host address space, not GPU memory. No fallback
removes a limit if CUDA initialization or a policy check fails.

Current user authorization permits only this child process /proc/<PID>/task
subtree WRITE_FILE|TRUNCATE in addition to the existing output/device writes.
No MAKE rights or other proc writes are granted. No metadata shim is loaded.
Explicit --cpu-only remains available with all other isolation/resource limits.

Kernel API references used when authoring, without executing this file:
https://docs.kernel.org/userspace-api/landlock.html
https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html
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
import runpy
import signal
import stat
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
LL_READ_TREE = LL_EXEC | LL_READ | LL_READ_DIR
LL_OUT = (LL_READ_TREE | LL_WRITE | LL_REMOVE_DIR | LL_REMOVE_FILE
          | LL_MAKE_DIR | LL_MAKE_REG | LL_REFER | LL_TRUNCATE)
SYSTEM_TREES = ("/usr", "/lib", "/lib64", "/opt", "/proc", "/sys")
ETC_FILES = ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/nsswitch.conf",
             "/etc/passwd", "/etc/group", "/etc/localtime", "/etc/os-release",
             "/etc/hosts", "/etc/nvidia/nvidia-application-profiles-rc.d/10-container.conf")

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
        "PATH": "/usr/local/bin:/usr/bin:/bin",
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


def _landlock(lib, out, script, sources, devices):
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


def _seccomp(lib):
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
        # exec/fork, process-group escape, ptrace and cross-process memory/fds.
        57, 58, 59, 322, 109, 112, 101, 310, 311, 438,
        # Namespace/mount, kernel/key/perf/module and io_uring escape surfaces.
        155, 161, 165, 166, 167, 168, 169, 175, 176, 246, 248, 249, 250,
        272, 298, 300, 308, 313, 321, 323, 425, 426, 427, 428, 429, 430,
        431, 432, 433, 442,
        # No identity/capability changes after the verified drop.
        105, 106, 113, 114, 116, 117, 119, 122, 123, 126,
    }
    for number in sorted(blocked):
        instructions.extend([(JEQ, 0, 1, number), (RET, 0, 0, deny)])
    # libc pthread creation falls back from clone3 ENOSYS to thread-only clone.
    instructions.extend([(JEQ, 0, 1, 435), (RET, 0, 0, ERRNO | errno.ENOSYS)])
    required = 0x100 | 0x800 | 0x10000  # CLONE_VM | CLONE_SIGHAND | CLONE_THREAD.
    instructions.extend([
        (JEQ, 0, 7, 56), (LD, 0, 0, 20), (JEQ, 1, 0, 0), (RET, 0, 0, deny),
        (LD, 0, 0, 16), (AND, 0, 0, required), (JEQ, 1, 0, required),
        (RET, 0, 0, deny), (RET, 0, 0, ALLOW),
    ])
    filters = (SockFilter * len(instructions))(*(SockFilter(*row) for row in instructions))
    program = SockProgram(len(instructions), filters)
    result = _checked(lib.syscall(317, 1, 1, ctypes.byref(program)), "seccomp BPF TSYNC install")
    if result != 0:
        raise RuntimeError(f"seccomp TSYNC failed on thread {result}; refusing partial confinement")


def _child(args):
    _platform_required()
    script, out = Path(args.script).resolve(strict=True), Path(args.out).resolve(strict=True)
    sources = _code_paths(script)
    devices = [] if args.cpu_only else _device_paths()
    if out == script.parent or out in script.parents:
        raise ValueError("writable output cannot contain inspected source")
    lib = _libc()
    _limits(args)
    _drop_identity(lib)
    os.chdir(out)
    abi = _landlock(lib, out, script, sources, devices)
    if not args.cuda_init_before_seccomp:
        _seccomp(lib)
    print(json.dumps({"event": "filesystem_identity_ready" if args.cuda_init_before_seccomp else "sandbox_ready",
                      "uid": os.getuid(), "gid": os.getgid(),
                      "groups": os.getgroups(), "landlock_abi": abi,
                      "seccomp": "pending_trusted_cuda_init" if args.cuda_init_before_seccomp else "enforced_tsync",
                      "no_new_privs": True,
                      "authorized_proc_write": {"path": f"/proc/{os.getpid()}/task",
                                                "rights": ["WRITE_FILE", "TRUNCATE"]},
                      "effective_rlimits": {
                          name: list(resource.getrlimit(getattr(resource, name)))
                          for name in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE",
                                       "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_CORE")
                      }}), flush=True)
    # Normal PyTorch operations only: no compiler, subprocess or network service.
    print(json.dumps({"event": "torch_import_begin"}), flush=True)
    import torch
    print(json.dumps({"event": "torch_import_complete"}), flush=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.cpu_only:
        print(json.dumps({"event": "explicit_cpu_only", "automatic_fallback": False}), flush=True)
    else:
        print(json.dumps({"event": "cuda_availability_begin"}), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable inside sandbox; refusing CPU/unsandboxed fallback")
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, 0)
        print(json.dumps({"event": "cuda_allocator_budget_set", "fraction": args.gpu_fraction,
                          "scope": "torch caching allocator; not arbitrary driver allocations"}), flush=True)
    if args.cuda_init_before_seccomp:
        _seccomp(lib)
        print(json.dumps({"event": "sandbox_ready", "seccomp": "enforced_tsync",
                          "uid": os.getuid(), "landlock_abi": abi}), flush=True)
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


def _terminate_session(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise RuntimeError("child not reaped after SIGKILL; do not report successful termination")


def _supervisor(args):
    _platform_required()
    script = Path(args.script).resolve(strict=True)
    sources = _code_paths(script)
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
    if args.cuda_init_before_seccomp:
        command.append("--cuda-init-before-seccomp")
    if args.script_args:
        command.extend(["--", *([*args.script_args][1:] if args.script_args[:1] == ["--"] else args.script_args)])
    record = {"status": "starting", "script_sha256": _sha(script),
              "source_sha256": {path.name: _sha(path) for path in sources},
              "timeout_seconds": args.timeout, "address_space_gib": args.address_space_gib,
              "file_limit_mib": args.file_limit_mib, "gpu_allocator_fraction": args.gpu_fraction,
              "cpu_only": args.cpu_only,
              "cuda_init_before_seccomp": args.cuda_init_before_seccomp,
              "authorized_proc_write": {"scope": "child /proc/<PID>/task only",
                                        "rights": ["WRITE_FILE", "TRUNCATE"], "make_rights": False},
              "environment_keys": sorted(env), "out": str(out)}
    started = time.monotonic()
    with open(out / "stdout.log", "xb", buffering=0) as stdout, open(out / "stderr.log", "xb", buffering=0) as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   env=env, close_fds=True, start_new_session=True, cwd=str(parent))
        try:
            code = process.wait(timeout=args.timeout)
            record.update(status="completed" if code == 0 else "failed", returncode=code)
        except subprocess.TimeoutExpired:
            record.update(status="wall_timeout", returncode=_terminate_session(process))
        except BaseException:
            _terminate_session(process)
            raise
    record["elapsed_seconds"] = time.monotonic() - started
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
    parser.add_argument("--address-space-gib", type=int, default=128)
    parser.add_argument("--file-limit-mib", type=int, default=1024)
    parser.add_argument("--gpu-fraction", type=float, default=0.5)
    parser.add_argument("--cpu-only", action="store_true", help="explicit CPU execution; isolation remains enforced")
    parser.add_argument("--cuda-init-before-seccomp", action="store_true",
                        help="trusted CUDA bootstrap after uid/Landlock, then TSYNC all threads before script")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800:
        parser.error("timeout must be finite and in (0, 1800] seconds")
    if not 8 <= args.address_space_gib <= 512 or not 1 <= args.file_limit_mib <= 1024:
        parser.error("address space must be 8..512 GiB; file size must be 1..1024 MiB")
    if not math.isfinite(args.gpu_fraction) or not 0 < args.gpu_fraction <= 0.9:
        parser.error("gpu fraction must be in (0, 0.9]")
    if args.cpu_only and args.cuda_init_before_seccomp:
        parser.error("CPU-only and trusted CUDA bootstrap are mutually exclusive")
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
