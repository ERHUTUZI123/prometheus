"""Fixed CPU admission: prepare the frozen TRAIN rows, then numerical F/B only."""
import argparse,hashlib,json,runpy,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 start=time.perf_counter();cpu=time.process_time();here=Path(__file__).resolve().parent
 cases=a.out/'fixed-cases.npz'
 sys.argv=[str(here/'prepare_factorized_graph_fb_cases.py'),'--inputs',str(a.inputs),'--out',str(cases)]
 runpy.run_path(sys.argv[0],run_name='__main__')
 digest=hashlib.sha256(cases.read_bytes()).hexdigest()
 with(a.out/'fixed-cases-receipt.json').open('x')as f:json.dump(dict(sha256=digest,bytes=cases.stat().st_size,preparation_wall_seconds=time.perf_counter()-start,preparation_CPU_seconds=time.process_time()-cpu),f,indent=2,sort_keys=True);f.write('\n')
 sys.argv=[str(here/'audit_factorized_graph_fb_owned.py'),'--core',str(here/'continuous_primitive_block.py'),'--checkpoint',str(a.inputs/'checkpoint.pt'),'--cases',str(cases),'--cases-sha256',digest,'--out',str(a.out/'contract'),'--device','cpu','--max-seconds','90']
 runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
