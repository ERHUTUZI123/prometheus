#!/usr/bin/env python3
"""Run inspected local numerical code with no credential env, no network, bounded wall time and run-only writes. Not an agent-context or evaluation-data sandbox."""
import argparse, json, os, subprocess, sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('script',type=Path);ap.add_argument('--timeout',type=float,default=120);ap.add_argument('--log',type=Path,required=True);args=ap.parse_args()
run=Path(__file__).resolve().parents[2];script=args.script.resolve();log=args.log.resolve()
if run not in script.parents or run not in log.parents:raise SystemExit('script/log must stay in run')
if args.timeout<=0:raise SystemExit('positive timeout required')
tmp=run/'probes/infrastructure/tmp';tmp.mkdir(exist_ok=True)
profile='(version 1)(allow default)(deny network*)(deny file-write* (require-not (subpath '+json.dumps(str(run))+')))' 
env={'PATH':'/usr/bin:/bin','TMPDIR':str(tmp),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
python=Path(__file__).parent/'venv/bin/python'
cmd=['/usr/bin/sandbox-exec','-p',profile,str(python),str(script)]
with log.open('x') as f:
 try:
  p=subprocess.run(cmd,cwd=run,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=args.timeout)
 except subprocess.TimeoutExpired:
  print('WALL_TIMEOUT',file=f);raise SystemExit(124)
raise SystemExit(p.returncode)
