import pathlib,json,hashlib,stat
base=pathlib.Path('/workspace/prometheus-v2-m0-architecture')
names=['gpu-native-v6-n8-seed0-stage2-'+x for x in ['transformer','replay','topo','topo_off']]
groups={}
for key,root,roots in [('runs',base/'outputs',names),('versions',base,['code-native-v6'])]:
 files=[];dirs=[]
 for name in roots:
  d=root/name
  if not d.is_dir() or d.is_symlink():raise RuntimeError('missing or unsafe directory '+str(d))
  dirs.append(name)
  for q in sorted(d.rglob('*')):
   s=q.lstat()
   if stat.S_ISLNK(s.st_mode):raise RuntimeError('symlink '+str(q))
   if stat.S_ISDIR(s.st_mode):dirs.append(str(q.relative_to(root)))
   elif stat.S_ISREG(s.st_mode):files.append({'path':str(q.relative_to(root)),'original_path':str(q),'bytes':s.st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'mode':oct(stat.S_IMODE(s.st_mode)),'uid':s.st_uid})
   else:raise RuntimeError('unexpected file '+str(q))
  if key=='runs':
   q=root/(name+'.runner-status.json')
   dstatus=json.loads(q.read_text())
   if dstatus.get('status')!='completed' or dstatus.get('returncode')!=0:raise RuntimeError('not completed '+str(q))
   s=q.stat();files.append({'path':str(q.relative_to(root)),'original_path':str(q),'bytes':s.st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'runner_status':dstatus})
 groups[key]={'remote_base':str(root),'directories':dirs,'files':files}
print(json.dumps({'groups':groups},sort_keys=True))
