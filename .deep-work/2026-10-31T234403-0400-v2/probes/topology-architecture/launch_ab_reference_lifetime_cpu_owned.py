"""Freeze or ROOT-execute one zero-SGD CPU audit of three OWN typed NNs."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-ab-reference-lifetime-cpu-v1'
REMOTE='/workspace/prometheus-v2-m0-architecture'
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
AUDIT='audit_ab_reference_lifetime_cpu.py'
TEMPLATE=next(ast.literal_eval(n.value)for n in ast.parse((ROOT/'launch_value_bridge_cpu_owned.py').read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='TEMPLATE'for t in n.targets))
TEMPLATE=TEMPLATE.replace('check_value_bridge_cpu.py',AUDIT).replace("out/'check'","out/'audit'").replace("out/'check/result.json'","out/'audit/result.json'").replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','COMPLETED_AB_OWN_REFERENCE_LIFETIME_CPU')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
 with path.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def prepare():
 version=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if version.exists()or protocol.exists():raise FileExistsError('preserve namespace')
 parent='parameter-typed-normalized-memory-ab-n8-seed0-fourstage-graph-v1';frozen=ROOT/'versions'/('code-'+parent)
 sources={leaf:(frozen/leaf).read_bytes()for leaf in('continuous_primitive_block.py','owned_fcba_core.py')};sources[AUDIT]=(ROOT/AUDIT).read_bytes()
 expected={'continuous_primitive_block.py':'93bdfea778c2f079ec04a6696ab8df77dc8bcc52b7f3fabdc57c05cbcad5250f','owned_fcba_core.py':'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158',AUDIT:'567242795700b028b821bd0c2ad31a7804678c98bfab5c75427fdf15e6a172ce'}
 for leaf,raw in sources.items():
  ast.parse(raw)
  if hashlib.sha256(raw).hexdigest()!=expected[leaf]:raise ValueError('reviewed source differs')
 files={}
 def owned(leaf,relative,digest=None):
  path=ROOT/'runs'/relative;actual=sha(path)
  if path.is_symlink()or not path.is_file()or(digest is not None and digest!=actual):raise ValueError('exact OWNinput differs')
  files[leaf]=dict(remote_relative=relative,sha256=actual,bytes=path.stat().st_size)
 owned('valid.npz','parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz','7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b')
 owned('baseline-soft.npz','parameter-ab-write-direction-controls-cpu-v1/audit/reference-raw-before-targets.npz')
 owned('baseline-native.npz','parameter-ab-fractional-store-cpu-v1/audit/native-NN-events-before-truth.npz')
 result=json.loads((ROOT/'runs'/parent/'learn/result.json').read_text());cp=result['checkpoint']
 if result['architecture_arm']!='AB'or result['actual_new_updates_this_process']!=2400:raise ValueError('completed AB only')
 owned('ab.pt',parent+'/learn/'+cp['file'],cp['sha256'])
 manifest=dict(schema='AB-own-reference-lifetime-CPU-v1',source_sha256=expected,checkpoint_sha256=cp['sha256'],parent_source_sha256=result['source_sha256'],baselines={'soft':dict(leaf='baseline-soft.npz',key='soft_logp',sha256=files['baseline-soft.npz']['sha256']),'native':dict(leaf='baseline-native.npz',key='after_logp',event7=True,sha256=files['baseline-native.npz']['sha256'])})
 config=dict(name=NAME,remote=REMOTE,runner_SHA=CPU,source_sha256=expected,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest);config=json.loads(json.dumps(config,sort_keys=True));source=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(source);version.mkdir()
 for leaf,raw in sources.items():
  with(version/leaf).open('xb')as stream:stream.write(raw)
  (version/leaf).chmod(0o444)
 rp=version/'remote-launch.source';rp.write_text(source);rp.chmod(0o444);put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',source_sha256=expected,input_files=files,NNs=1,checkpoint_sha256=cp['sha256'],modes=['held_soft','held_native'],SGD_updates=0,expected_active_transitions=34304,P=10351,carry_scalars=328,lifetime_rule='snapshotNNOWNupdatedaddressaftereachpublicKEYthenoverrideONLYupdatedaddress throughfollowingvalueconstructionandthatWRITE; WRITEusesheldincomingaddress; querysuffixuntouched',other_fields='h/V/z/c/actionupdatesandallfuturecallsnaturalclosedloop, nonchosenfractionalwritesretainedinSoft; noprovidedslot/value/address',privileged_information='publictwoKEYtoWRITEintervaltimingonly; diagnosticnotlearner/deployablecapability',outputs='fullstepaddressbefore/proposed/applied, readattention/w/action/h/read/z/logp, ownkeysnapshots/eventpositions/finalV+predictions; allrawbeforetargets',original_baselines='SHA-boundprevioussoft/nativeoutputsafterraw; norerun',outside_seconds=150,science_seconds=120,CPU_only=True,AS_GiB=64,file_limit_MiB=100,runner_sha256=CPU,remote_launch_sha256=sha(rp),no_new_permissions=True,full_FLOPs=None,provider_effective_compute=None,cost_scope='bothCPUfull512passes/NNload/publicintervalsnapshots/source/baselineinputcopies/allrawIO/CPU/RSS, nooptimizer/training; oldbaselinecostsretainedseparately')
 put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=expected,remote_launch_sha256=sha(rp),protocol_sha256=sha(protocol)),sort_keys=True))
def execute():
 version=ROOT/'versions'/('code-'+NAME);config=json.loads((version/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(version/'remote-launch.source').read_text()
 if sha(version/'remote-launch.source')!=protocol['remote_launch_sha256']:raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():
  if sha(version/leaf)!=digest:raise ValueError('frozen source differs')
  ast.parse((version/leaf).read_bytes())
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));frozen=ast.literal_eval(node.value)
 if frozen!=config or source!=TEMPLATE.replace('CONFIG',repr(frozen),1):raise ValueError('frozen config/template differs')
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actual receipt, no retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as error:value=dict(returncode=None,stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspect bounded actual state, no retry')
 value.update(name=NAME,remote_launch_sha256=sha(version/'remote-launch.source'),source_sha256=config['source_sha256']);put(receipt,value);print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');args=parser.parse_args()
 if args.prepare:prepare()
 else:execute()
