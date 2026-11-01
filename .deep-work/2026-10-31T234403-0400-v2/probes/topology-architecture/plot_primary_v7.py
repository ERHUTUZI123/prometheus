"""Plot frozen completed-training records, without model execution or training."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/opt/prometheus-v2-python')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=json.loads(a.input.read_text());a.out.mkdir(parents=True,exist_ok=False)
colors={'topo':'#2563eb','topo_off':'#64748b','transformer':'#ea580c','replay':'#16a34a','adapter':'#9333ea'}
labels={'topo':'Continuous topology ON','topo_off':'Step-guard OFF','transformer':'Transformer','replay':'Transformer + Replay','adapter':'Transformer + Adapter'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
for i,n in enumerate((8,16)):
 for task in range(3):
  ax=axes[i,task]
  for name in colors:
   rows=[r for r in s['curves'] if r['scale']==n and r['model']==name and r['task']==task and r['split']=='iid' and r['stage']>=task]
   rows.sort(key=lambda r:(r['stage'],r['step']))
   ax.plot([r['global_update'] for r in rows],[100*r['hard_acc'] for r in rows],color=colors[name],label=labels[name],lw=1.7,ls='--' if name=='topo_off' else '-')
  for boundary in (800,1600):ax.axvline(boundary,c='#cbd5e1',ls=':',lw=1)
  ax.set_title(f'N={n}, task {task+1}');ax.set_ylim(0,103);ax.set_xlim(0,2400);ax.grid(alpha=.15)
  if task==0:ax.set_ylabel('IID accuracy (%)')
  if i==1:ax.set_xlabel('Cumulative parameter-update budget')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.005),frameon=False)
fig.suptitle('Actual acquisition and retention | seed 0 | 800 updates per stage',y=.985)
fig.tight_layout(rect=(0,.09,1,.95))
for ext in ('png','svg'):fig.savefig(a.out/f'learning-and-retention.{ext}',bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,7))
for i,n in enumerate((8,16)):
 for name in colors:
  m=next(x for x in s['models'] if x['scale']==n and x['model']==name)
  train=[];wall=[];t=w=0
  for row in m['stages']:
   t+=row['train_seconds'];w+=row['wall_seconds'];train.append(t);wall.append(w)
  for j,values in enumerate((m['params'],train,wall)):
   axes[i,j].plot((1,2,3),values,marker='o',c=colors[name],label=labels[name],ms=4,ls='--' if name=='topo_off' else '-')
 for j,title in enumerate(('Total model parameters','Cumulative measured training (s)','Cumulative stage wall time (s)')):
  axes[i,j].set_title(f'N={n}: {title}');axes[i,j].set_xticks((1,2,3));axes[i,j].set_xlabel('Learning stage completed');axes[i,j].grid(alpha=.15)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.005),frameon=False)
fig.suptitle('Recorded growth and time | costs are not matched-accuracy efficiency',y=.99)
fig.tight_layout(rect=(0,.1,1,.95))
for ext in ('png','svg'):fig.savefig(a.out/f'growth-and-time.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps({'status':'PLOTTED_SAVED_RECORDS_ONLY','files':sorted(x.name for x in a.out.iterdir()),'curves':len(s['curves'])}))
