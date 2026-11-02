"""Strict-phase negative checks; uses only a clearly named synthetic fixture.

Run via the staged runner, after its fixed prologue/irreversible transition.
No real credentials or secret paths are inspected. Unexpected permitted exec
replaces this process with a harmless nonzero exit; absence of a passing JSON
therefore always fails the coordinator's check, even if an exec target exits 0.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-denied-path', type=Path, required=True)
    args = parser.parse_args()
    out = Path(os.environ['SANDBOX_OUT']).resolve(strict=True)
    denied = args.synthetic_denied_path.absolute()
    if denied.name != '.compiler-policy-synthetic-denied.txt':
        raise ValueError('only the explicitly synthetic fixture basename is accepted')
    if denied.is_relative_to(out) or denied.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('synthetic denied fixture must be outside owned output/source')
    if any(denied.is_relative_to(Path(v)) for v in ('/usr','/lib','/lib64','/opt','/proc','/sys')):
        raise ValueError('synthetic denied fixture must be outside readable system trees')
    record = dict(strict_phase=True, checks=[], completed=False)
    def expect_denied(label, action, expected=(errno.EPERM, errno.EACCES)):
        try:
            action()
        except OSError as error:
            if error.errno not in expected:
                raise RuntimeError(f'{label}: wrong failure errno {error.errno}') from error
            record['checks'].append(dict(check=label, denied=True, errno=error.errno))
        else:
            raise RuntimeError(f'{label}: unexpectedly permitted')
    lib = ctypes.CDLL(None, use_errno=True); lib.syscall.restype = ctypes.c_long
    def syscall(number, *values):
        ctypes.set_errno(0)
        answer = lib.syscall(number, *values)
        if answer == -1:
            value = ctypes.get_errno(); raise OSError(value, os.strerror(value))
        return answer
    def fork_test():
        child = os.fork()
        if child == 0:
            os._exit(75)
        os.waitpid(child, 0)
    def clone_test():
        child = syscall(56, ctypes.c_ulong(17), 0, 0, 0, 0)
        if child == 0:
            os._exit(75)
        os.waitpid(child, 0)
    expect_denied('fork', fork_test)
    expect_denied('clone_process_SIGCHLD', clone_test)
    expect_denied('clone3_ENOSYS', lambda: syscall(435, 0, 0), (errno.ENOSYS,))
    expect_denied('memfd_create', lambda: syscall(319, ctypes.c_char_p(b'compiler-policy-synthetic'), 0))
    for kind in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
        def create_socket(kind=kind):
            candidate = socket.socket(kind, socket.SOCK_STREAM)
            candidate.close()
        expect_denied('socket_' + str(kind), create_socket)
    expect_denied('synthetic_protected_read', lambda: denied.read_bytes())
    # Minimal static ELF64 exit(75), authored here rather than compiled elsewhere.
    code = bytes.fromhex('b83c000000bf4b0000000f05')
    ident = b'\x7fELF\x02\x01\x01' + bytes(9)
    header = struct.pack('<16sHHIQQQIHHHHHH', ident, 2, 62, 1, 0x400078,
                         64, 0, 0, 64, 56, 1, 0, 0, 0)
    segment = struct.pack('<IIQQQQQQ', 1, 5, 0, 0x400000, 0x400000,
                          120 + len(code), 120 + len(code), 4096)
    binary = out / 'cache' / 'compiler-policy-static-elf'
    with binary.open('xb') as stream:
        stream.write(header + segment + code)
    binary.chmod(0o700)
    targets = (
        ('shell_execve', '/bin/sh', ['sh','-c','exit 74']),
        ('python_execve', sys.executable, ['python','-c','raise SystemExit(76)']),
        ('pinned_compiler_execve', '/usr/bin/x86_64-linux-gnu-gcc-13', ['gcc','--compiler-policy-must-deny']),
        ('output_cache_ELF_execve', str(binary), [str(binary)]),
        ('loader_execve', '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2',
            ['ld-linux','/bin/sh','-c','exit 78']),
    )
    for label, path, argv in targets:
        expect_denied(label, lambda path=path, argv=argv: os.execve(path, argv, {}))
    argv = (ctypes.c_char_p * 4)(b'sh', b'-c', b'exit 77', None)
    env = (ctypes.c_char_p * 1)(None)
    expect_denied('execveat_shell', lambda: syscall(322, -100, ctypes.c_char_p(b'/bin/sh'), argv, env, 0))
    with binary.open('rb') as stream:
        expect_denied('execveat_output_AT_EMPTY_PATH', lambda: syscall(322, stream.fileno(),
                      ctypes.c_char_p(b''), argv, env, 0x1000))
    # Thread creation remains available without enabling process clones.
    event = threading.Event(); thread = threading.Thread(target=event.set)
    thread.start(); thread.join(timeout=2)
    if not event.is_set() or thread.is_alive():
        raise RuntimeError('strict pthread creation did not complete')
    record['checks'].append(dict(check='pthread', completed=True))
    status = Path('/proc/self/status').read_text()
    fields = {line.split(':',1)[0]: line.split(':',1)[1].strip()
              for line in status.splitlines() if ':' in line}
    if fields.get('NoNewPrivs') != '1' or fields.get('Seccomp') != '2':
        raise RuntimeError('NO_NEW_PRIVS/seccomp status not enforced')
    if os.getresuid() != (65534,65534,65534) or os.getresgid() != (65534,65534,65534):
        raise RuntimeError('unexpected identity')
    record.update(completed=True, uid=os.getuid(), no_new_privs=fields['NoNewPrivs'],
                  seccomp=fields['Seccomp'], secret_data_read=False)
    with (out / 'strict-policy-checks.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
