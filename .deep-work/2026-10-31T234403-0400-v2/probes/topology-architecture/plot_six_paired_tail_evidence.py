"""One bounded three-panel scientific figure; frozen evidence only, no NN."""
import argparse,csv,hashlib,json,math,os,resource,shutil,sys,time
from pathlib import Path
MODELS=('gru32','transformer32','typed');SAMPLERS=('paired','same_pool_shuffle')
NAMES={'gru32':'GRU32','transformer32':'Transformer32','typed':'Typed Block'}
COLORS={'gru32':'#2166AC','transformer32':'#B35806','typed':'#1B7837'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(inputs):
 manifest=json.loads((inputs/'manifest.json').read_text());assert manifest['schema']=='six-paired-tail-figure-inputs-v1'and sha(Path(__file__))==manifest['source_sha256'];values={}
 for leaf,digest in manifest['data_sha256'].items():
  p=inputs/leaf;assert p.is_file()and sha(p)==digest
  if leaf.endswith('.json'):values[leaf]=json.loads(p.read_text())
  elif leaf.endswith('.jsonl'):values[leaf]=[json.loads(x)for x in p.read_text().splitlines()]
  elif leaf.endswith('.csv'):
   with p.open(newline='')as f:values[leaf]=list(csv.DictReader(f))
 proof=values['six-evidence.json'];assert proof['status']=='PASS_ALL_SIX_PAIRED_SAMEPOOL_SAVED_RAW';assert len(values['summary.csv'])==6 and len(values['curves.csv'])==168;data=[]
 for model in MODELS:
  for sampler in SAMPLERS:
   prefix=model+'-'+sampler;r=values[prefix+'-result.json'];rows=values[prefix+'-records.jsonl'];u=[x for x in rows if x['kind']=='actual_update'];assert[x['step']for x in u]==list(range(1,601));assert r['sampling']==sampler
   actual=proof['models'][model][sampler];assert sha(inputs/(prefix+'-result.json'))==actual['result_sha256']and sha(inputs/(prefix+'-records.jsonl'))==actual['records_sha256'];curves=r['curves'];assert[x['step']for x in curves]==list(range(0,601,100));logged=[x for x in rows if x['kind']=='actual_VALID_curve'];assert len(logged)==7
   for a,b in zip(curves,logged):assert a['step']==b['step']and a['valid']==b['valid']
   infer=r['inference']['soft'];assert(infer['batch_size'],infer['word_length_max'],infer['warmups'],infer['timed_forwards'])==(64,39,3,20);cost=actual['costs'];assert math.isfinite(infer['wall_latency_p50'])and infer['wall_latency_p50']>0
   data.append(dict(model=model,sampling=sampler,label=NAMES[model]+(' paired'if sampler=='paired'else' shuffle'),steps=[x['step']for x in curves],new_VALID=[x['valid']['3']['soft']['accuracy']for x in curves],old_VALID_mean=[sum(x['valid'][str(task)]['soft']['accuracy']for task in range(3))/3 for x in curves],cost=cost,inference_soft_eager_p50_ms=1000*infer['wall_latency_p50'],inference_scope='same64VALIDexamples, active2067/maxL39,3warmups+20timed; soft/eagerforward, notGPUbusy',result_sha256=actual['result_sha256']))
 for model in MODELS:
  pair=[x for x in data if x['model']==model];assert pair[0]['new_VALID'][0]==pair[1]['new_VALID'][0]and pair[0]['old_VALID_mean'][0]==pair[1]['old_VALID_mean'][0]
 for row in values['curves.csv']:
  x=next(z for z in data if z['model']==row['model']and z['sampling']==row['sampling']);i=x['steps'].index(int(row['update']));r=values[row['model']+'-'+row['sampling']+'-result.json'];assert float(row['accuracy'])==r['curves'][i]['valid'][row['task']]['soft']['accuracy']
 return manifest,proof,data

def render(data,proof,out):
 sys.path.insert(0,'/opt/prometheus-analysis-python');os.environ['MPLCONFIGDIR']=str(out/'.matplotlib-cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(14.8,5.7),gridspec_kw={'width_ratios':[1,1,1.23]});a,b,c=axes;handles=[]
 for x in data:
  style=dict(color=COLORS[x['model']],linestyle='-'if x['sampling']=='paired'else'--',marker='o'if x['sampling']=='paired'else's',markersize=3,linewidth=1.5)
  line,=a.plot(x['steps'],x['new_VALID'],label=x['label'],**style);handles.append(line);b.plot(x['steps'],x['old_VALID_mean'],**style)
 a.set(title='a  New-task VALID learning',xlabel='New optimizer updates',ylabel='Soft accuracy',ylim=(0,1.02),xticks=(0,200,400,600));b.set(title='b  Mean old-task VALID',xlabel='New optimizer updates',ylabel='Mean soft accuracy (three tasks)',ylim=(0,1.02),xticks=(0,200,400,600))
 for ax in(a,b):ax.grid(axis='y',color='.9',lw=.6);ax.set_axisbelow(True)
 c.set_title('c  Parameters and measured cost scopes');c.axis('off');cells=[]
 for x in data:
  z=x['cost'];cells.append([('GRU32'if x['model']=='gru32'else'TF32'if x['model']=='transformer32'else'Typed')+('/pair'if x['sampling']=='paired'else'/shuffle'),f"{z['P']/1000:.3f}",f"{z['training_wall_seconds']:.2f}",f"{z['science_wall_seconds']:.2f}",f"{z['local_cold_seconds']:.2f}",f"{x['inference_soft_eager_p50_ms']:.2f}"])
 table=c.table(cellText=cells,colLabels=['Model / batch','P\n(k)','Train\n(s)','Science\n(s)','Cold\n(s)','Eager\n(ms)'],cellLoc='center',colWidths=[.31,.12,.13,.14,.13,.17],bbox=[-.015,.35,1.06,.61]);table.auto_set_font_size(False);table.set_fontsize(8)
 for (row,col),cell in table.get_celld().items():
  cell.set_edgecolor('#D5D5D5');cell.set_linewidth(.5)
  if row==0:cell.set_facecolor('#EFEFEF');cell.set_text_props(weight='bold')
  elif col==0:cell.set_text_props(color=COLORS[data[row-1]['model']])
 c.text(.0,.29,'Eager: B64, max L39; soft wall p50, 20 timed + 3 warmups.\nTrain / science / cold are nested scopes: do not sum.\nNN / Adam: typed 44.0 / 88.0 kB; GRU 56.1 / 112.2 kB;\nTF 107.0 / 214.3 kB. Replay 55.3 kB per tail.\nTyped training uses one Graph; eager inference does not.',transform=c.transAxes,va='top',fontsize=8,linespacing=1.45)
 fig.suptitle('Paired minibatches versus the same globally shuffled TRAIN pool',fontsize=14,y=.985);fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False,handlelength=2.8,columnspacing=2.4)
 totals=proof['totals'];foot=('Development seed0 only; no Primary or architecture-advantage claim. Each tail restores its own model’s same stage2 NN / Adam / RNG / Replay.\n'
       'Current-row multiplicities and replay draws match; grouping and temporal order both change. Initial old-task competence differs across model families.\n'
       f"Six new tails: 3,600 updates, {totals['training_wall_seconds']:.2f}s training / {totals['science_wall_seconds']:.2f}s science / {totals['local_cold_seconds']:.2f}s cold. "
       f"One-time formation + CPU admissions bring cold total to {totals['all6_plus_once_shared_CPU_local_cold_seconds']:.2f}s.\n"
       'Earlier training, configurations and failed attempts remain charged separately. Full training FLOPs, GPU busy time, energy and provider-effective compute are unknown.')
 fig.text(.035,.025,foot,ha='left',va='bottom',fontsize=8,linespacing=1.35);fig.subplots_adjust(left=.055,right=.975,bottom=.285,top=.78,wspace=.28)
 for ext in('png','svg','pdf'):fig.savefig(out/('six-paired-tails-v1.'+ext),dpi=180)
 plt.close(fig)
 return dict(matplotlib_version=matplotlib.__version__,panels=3,legend_entries=6)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.perf_counter();cpu=time.process_time();a.out.mkdir(parents=True,exist_ok=False);manifest,proof,data=load(a.inputs);backend=render(data,proof,a.out);cache=a.out/'.matplotlib-cache';files=[x for x in cache.rglob('*')if x.is_file()]if cache.exists()else[];cachecost=dict(files=len(files),bytes=sum(x.stat().st_size for x in files),retained=False,scope='temporaryperrunMPLcacheformationcharged; removedafterrender tokeep3figures+receipt')
 if cache.exists():shutil.rmtree(cache)
 record=dict(status='RENDERED_SIX_PAIRED_TAIL_FIGURE',source_sha256=sha(Path(__file__)),input_sha256=manifest['data_sha256'],backend=backend,plotted_data=data,shared_costs=proof['shared_costs'],new_NN_queries=0,SGD_updates=0,render_wall_seconds=time.perf_counter()-start,render_CPU_seconds=time.process_time()-cpu,render_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),matplotlib_cache=cachecost,artifacts={x.name:dict(bytes=x.stat().st_size,sha256=sha(x))for x in sorted(a.out.iterdir())if x.is_file()},scope='savedresult/JSONL/verifiedCSV only; noNN/checkpointimport/compiler/network/dependencyinstallation; singledevelopmentseed, unknownFLOPs/provider; oldVALIDbehaviornotuniversalretentionclaim')
 with(a.out/'figure-result.json').open('x')as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps(dict(status=record['status'],out=str(a.out),new_NN_queries=0,SGD_updates=0)))
if __name__=='__main__':main()
