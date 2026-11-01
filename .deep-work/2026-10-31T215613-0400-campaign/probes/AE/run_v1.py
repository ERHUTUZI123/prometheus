"""Outer-enforced no-network, scrubbed environment, write-limited finite probe."""
import json, os, resource, subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
PY=D.parent/'infrastructure/venv/bin/python'
profile='(version 1)(allow default)(deny network*)(deny file-write* (require-not (subpath '+json.dumps(str(D))+')))'
def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(165,170))
env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
cmd=['/usr/bin/sandbox-exec','-p',profile,'/usr/bin/env','-i',*['%s=%s'%x for x in env.items()],str(PY),str(D/'probe_v1.py')]
with (D/'execution_v1.log').open('x') as f:
    f.write(json.dumps({'command':cmd,'wall_timeout_seconds':180,'cpu_limits_seconds':[165,170],'environment_keys':list(env),'network':'deny network*','write_root':str(D)})+'\n');f.flush()
    try:p=subprocess.run(cmd,cwd=D,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180,preexec_fn=limits)
    except subprocess.TimeoutExpired:
        f.write('WALL_TIMEOUT\n');raise SystemExit(124)
raise SystemExit(p.returncode)
