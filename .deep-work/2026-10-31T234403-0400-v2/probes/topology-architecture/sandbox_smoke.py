"""UNEXECUTED sandbox acceptance probe; run only through inspected remote_runner.

Prints a small CUDA forward/backward and explicit denial/success checks.
Any unexpected permission or CUDA failure makes the probe fail; no fallback.
"""

import errno
import json
import os
from pathlib import Path
import socket

import torch


def denied(name, operation):
    try:
        operation()
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EACCES):
            raise RuntimeError(f"{name}: not a demonstrated permission denial: {error}") from error
        print(json.dumps({"check": name, "denied": True, "errno": error.errno}), flush=True)
    else:
        raise RuntimeError(f"{name}: unexpectedly allowed")


if os.getuid() != 65534 or os.getgid() != 65534 or os.getgroups():
    raise RuntimeError("identity/group confinement not observed")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU fallback")

x = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4).requires_grad_()
weight = torch.full((4, 2), 0.125, device="cuda")
prediction = x @ weight
loss = prediction.square().mean()
loss.backward()
torch.cuda.synchronize()
if x.grad is None or not torch.isfinite(x.grad).all().item():
    raise RuntimeError("CUDA backward did not produce finite gradients")
print(json.dumps({"check": "cuda_forward_backward", "device": str(x.device),
                  "prediction": prediction.detach().cpu().tolist(), "loss": loss.item(),
                  "gradient_sum": x.grad.sum().item()}), flush=True)


def try_socket():
    descriptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    descriptor.close()


def try_root_ssh():
    # Directory open demonstrates denial even if no particular key filename exists.
    descriptor = os.open("/root/.ssh", os.O_RDONLY | os.O_DIRECTORY)
    os.close(descriptor)


def try_outside_write():
    # A new name avoids truncating an existing host file if isolation is defective.
    path = Path("/tmp") / f"sandbox-outside-{os.getpid()}-must-not-exist"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    path.unlink()


denied("socket", try_socket)
denied("root_ssh_read", try_root_ssh)
denied("outside_tmp_write", try_outside_write)

own_comm = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/comm")
old_name = own_comm.read_text(encoding="utf-8")
with open(own_comm, "w", encoding="utf-8") as stream:
    stream.write("prom-smoke\n")
if own_comm.read_text(encoding="utf-8").strip() != "prom-smoke":
    raise RuntimeError("authorized own-process thread comm write failed")
with open(own_comm, "w", encoding="utf-8") as stream:
    stream.write(old_name)
print(json.dumps({"check": "authorized_own_comm_write", "succeeded": True,
                  "path": str(own_comm)}), flush=True)


def try_other_process_comm():
    descriptor = os.open("/proc/1/task/1/comm", os.O_WRONLY | os.O_TRUNC)
    os.close(descriptor)


denied("other_process_comm_write", try_other_process_comm)
out = Path(os.environ["SANDBOX_OUT"])
path = out / "smoke-write.txt"
with open(path, "x", encoding="utf-8") as stream:
    stream.write("sandbox output write succeeded\n")
if path.read_text(encoding="utf-8") != "sandbox output write succeeded\n":
    raise RuntimeError("output write/readback mismatch")
print(json.dumps({"check": "outdir_write", "succeeded": True, "path": str(path)}), flush=True)
print(json.dumps({"smoke": "passed"}), flush=True)
