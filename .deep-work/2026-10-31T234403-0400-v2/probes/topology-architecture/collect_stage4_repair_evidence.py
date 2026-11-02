"""Rebuild actual matched fourth-phase outcome/P/full-resource evidence.

No model/data-label reads or new experiments. Preserve earlier ledgers.
CSV is a view; JSON retains all measured resource scopes and their overlaps.
"""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    runs={key:'parameter-memory-live-stage-n8-seed0-'+key+'-stage4-v1'for key in('dynamic','static','gru','transformer')}
    runs.update(delta='parameter-memory-live-delta-stage-n8-seed0-dynamic-stage4-v1',credit='parameter-memory-live-credit-stage-n8-seed0-dynamic-stage4-v1',group8='parameter-memory-live-group-carry-stage-n8-seed0-dynamic-stage4-v2')
    rows=[]
    for model,name in runs.items():
        p=ROOT/'runs'/name/'learn/result.json';v=json.loads(p.read_text());receipt=ROOT/(name+'-coordinator-receipt.json');r=json.loads(receipt.read_text())
        if r['returncode']!=0 or v['new_training_updates']!=600:raise ValueError('actual completed600 required')
        records=p.parent/'records.jsonl';updates=[]
        for line in records.open():
            row=json.loads(line)
            if row['kind']=='actual_update':updates.append(row['step'])
        if updates!=list(range(1,601)):raise ValueError('actual600 raw update sequence differs')
        c=v['costs'];tests=v['frozen_test_metrics'];valid=v['final_TRAIN_VALID']['metrics']
        compact=dict(model=model,P=v['resources']['parameters'],new_VALID=valid['3']['valid']['soft']['accuracy'],new_IID=tests['3']['test_iid']['soft']['accuracy'],new_LONG=tests['3']['test_long']['soft']['accuracy'],new_UNSEEN=tests['3']['test_unseen']['soft']['accuracy'],old0_IID=tests['0']['test_iid']['soft']['accuracy'],old1_IID=tests['1']['test_iid']['soft']['accuracy'],old2_IID=tests['2']['test_iid']['soft']['accuracy'],training_seconds=c['training_wall_seconds'],training_forward_GMAC=c['training_forward_MAC_proxy']/1e9,coordinator_seconds=r['wall_seconds'],optimizer_tensor_bytes=v['resources']['optimizer_tensor_bytes'],replay_bytes=v['resources']['replay_bytes'],checkpoint_bytes=v['checkpoint']['bytes'])
        rows.append(dict(summary=compact,run=name,result_sha256=sha(p),raw_updates_sha256=sha(records),actual_update_count=len(updates),costs=c,all_resources=v['resources'],growth=v['growth'],inference=v['inference'],inference_total_wall_seconds=v['inference_total_wall_seconds'],frozen_test_costs=v['frozen_test_measurement_costs'],curves=v['curves'],counterfactual_audit=v['counterfactual_prediction_audit']))
    prefix=ROOT/'stage4-repairs-parameter-complete-cost-evidence-v1'
    with prefix.with_suffix('.json').open('x')as f:json.dump(dict(scope='actual matched OWN same-parent fourth-stage repairs; four original arms retain own three-stage parents; single seed development',rows=rows,full_paid_baseline_lineage='memory-live-stage-total-paid-training-cost-evidence-v1.json',cost_overlap='shared original dynamic parent acquired ONCE then reused by three repairs; do not sum inherited costs once per branch as a physical campaign bill; training/eval/runner timings nested',parameter_efficiency='different quality/retention: no same-capability parameter advantage or finite deltaP/deltaCapability claimed',unknown_costs='full training/backward FLOPs, hardware energy and optimal deployment costs remain unknown, not zero'),f,indent=2,sort_keys=True)
    with prefix.with_suffix('.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]['summary']));w.writeheader();w.writerows(r['summary']for r in rows)
    print(json.dumps([r['summary']for r in rows],indent=2))
if __name__=='__main__':main()
