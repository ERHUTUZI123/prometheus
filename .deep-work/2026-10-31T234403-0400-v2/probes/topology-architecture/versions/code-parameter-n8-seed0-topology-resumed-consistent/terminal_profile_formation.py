"""Bounded passive state/operation formation from actual terminal examples.

No teacher transition, hidden mode, correct route, environment callback or
test label is accepted. All public one-symbol prefixes are examined symmetrically.
Observed continuation profiles provide the state-separation witness and closure.
Same procedure is available to topology, ordinary constraints and frozen modules.
"""
import copy
import hashlib
import time

import numpy as np
import torch

from topology_model import linear_sum_assignment, assignment


def observe(data, alphabet, seconds=15.):
    began=time.perf_counter();deadline=began+seconds
    alphabet=tuple(sorted(int(a) for a in alphabet))
    if len(alphabet)>6:raise ValueError('six-symbol process limit')
    tasks=np.unique(data['task_ids'])
    if len(tasks)!=1:raise ValueError('one fixed task per observation table')
    records={};source_rows={};digest=hashlib.sha256()
    for key in ('tokens','starts','task_ids','targets'):
        digest.update(key.encode());digest.update(np.asarray(data[key],dtype='<i8').tobytes())
    for row,(tokens,x,y) in enumerate(zip(data['tokens'],data['starts'],data['targets'])):
        if time.perf_counter()>deadline:raise TimeoutError('terminal observation indexing cap')
        word=tuple(int(a) for a in tokens if a)
        key=int(x),word
        if key in records and records[key]!=int(y):raise ValueError('inconsistent deterministic feedback')
        records[key]=int(y);source_rows.setdefault(key,row)
    starts=tuple(sorted(int(x) for x in np.unique(data['starts'])))
    prefixes=((),)+tuple((a,) for a in alphabet)
    suffixes=prefixes
    histories=[(x,p) for x in starts for p in prefixes]
    profiles={};groups={};members={}
    for h in histories:
        if time.perf_counter()>deadline:raise TimeoutError('profile construction cap')
        x,p=h
        try:profile=tuple(records[x,p+s] for s in suffixes)
        except KeyError as error:raise ValueError('incomplete public prefix/suffix terminal coverage') from error
        profiles[h]=profile
        group=groups.setdefault(profile,len(groups))
        members.setdefault(group,[]).append(h)
    if len(groups)>64:raise ValueError('observed state witness exceeds64-state bound')
    trans={a:{} for a in alphabet}
    for h in histories:
        x,p=h;src=groups[profiles[h]]
        for a in alphabet:
            profile=tuple(records[x,p+(a,)+s] for s in suffixes)
            if profile not in groups:raise ValueError('observation rows not continuation-closed')
            dst=groups[profile]
            if src in trans[a] and trans[a][src]!=dst:raise ValueError('merged profiles have conflicting observed successors')
            trans[a][src]=dst
    if any(len(set(t.values()))!=len(groups) for t in trans.values()):
        raise ValueError('observed action is not reversible under current representation')
    pairs=[]
    signatures=list(groups)
    for i,left in enumerate(signatures):
        for j in range(i+1,len(signatures)):
            k=next(k for k,(a,b) in enumerate(zip(left,signatures[j])) if a!=b)
            h0,h1=members[i][0],members[j][0]
            pairs.append(dict(left=i,right=j,suffix=list(suffixes[k]),
                terminal_labels=[left[k],signatures[j][k]],
                actual_source_rows=[source_rows[h0[0],h0[1]+suffixes[k]],source_rows[h1[0],h1[1]+suffixes[k]]]))
    return dict(task=int(tasks[0]),alphabet=alphabet,groups=groups,members=members,
        profiles=profiles,trans=trans,required_states=len(groups),starts=starts,
        audit=dict(training_only=True,fixed_task_id=int(tasks[0]),
            rule='passive terminal-response partition plus executable continuation closure',
            input_sha256=digest.hexdigest(),rows=len(data['starts']),unique_records=len(records),
            public_history_count=len(histories),observed_state_classes=len(groups),
            witness_pairs=pairs,prefixes=[list(p) for p in prefixes],suffixes=[list(s) for s in suffixes],
            teacher_transition_used=False,hidden_mode_used=False,correct_call_used=False,
            seconds=time.perf_counter()-began))


@torch.no_grad()
def candidate(model, observations, data, seconds=15.):
    began=time.perf_counter();deadline=began+seconds
    task,alphabet=observations['task'],observations['alphabet']
    m=observations['required_states']
    if m>model.d:raise ValueError('strict observed state witness requires workspace growth')
    graph=copy.deepcopy(model.graph(task,alphabet))
    enc,dec=np.asarray(graph['enc']),np.asarray(graph['dec'])
    allowed=np.zeros((m,model.d),dtype=bool)
    signatures=list(observations['groups'])
    for i,p in enumerate(signatures):allowed[i]=dec==p[0]
    for x in observations['starts']:
        group=observations['groups'][observations['profiles'][x,()]]
        target=int(enc[x])
        if dec[target]!=int(x):raise ValueError('current learned initial encoding/readout conflicts with actual empty-word feedback')
        allowed[group]=False;allowed[group,target]=True
    row,column=linear_sum_assignment(np.where(allowed,0.,1e9))
    if len(row)!=m or not np.all(allowed[row,column]):raise ValueError('current learned readout cannot realize separated histories')
    mu=np.empty(m,dtype=np.int64);mu[row]=column
    bank=[np.arange(model.d,dtype=np.int64)]+[assignment(p) for p in model.operations]
    joint=model.joint_graph(task,alphabet)
    reused=[];matching_calls=1
    for a in alphabet:
        if time.perf_counter()>deadline:raise TimeoutError('terminal closure matching cap')
        destinations=np.asarray([observations['trans'][a][q] for q in range(m)])
        forced=mu[destinations]
        compatible=[i for i,p in enumerate(bank) if np.array_equal(p[mu],forced)]
        if compatible:
            graph['trans'][a]=bank[compatible[0]].copy()
            reused.append(dict(symbol=a,expert=compatible[0],source='existing learned action compatible on all observed reachable states'))
        else:
            permit=np.ones((model.d,model.d),dtype=bool)
            permit[mu]=False;permit[mu,forced]=True
            # Learned native/joint maps score the as-yet-unobserved complement;
            # terminal constraints determine every observed reachable transition.
            score=np.zeros_like(permit,dtype=float)
            score[np.arange(model.d),joint['trans'][a]]=1.
            r,c=linear_sum_assignment(np.where(permit,-score,1e9));matching_calls+=1
            if not np.all(permit[r,c]):raise ValueError('continuation closure has no full reversible completion')
            p=np.empty(model.d,dtype=np.int64);p[r]=c;graph['trans'][a]=p
    q=enc[data['starts']].copy();uses=0
    for pos in range(data['tokens'].shape[1]):
        for a in alphabet:
            active=data['tokens'][:,pos]==a
            q[active]=graph['trans'][a][q[active]];uses+=int(np.count_nonzero(active))
    errors=int(np.count_nonzero(dec[q]!=data['targets']))
    audit=copy.deepcopy(observations['audit'])
    audit.update(history_to_learned_state=mu.tolist(),reused_action_candidates=reused,
        matching_calls=matching_calls,training_terminal_evaluations=len(q),
        training_transition_uses=uses,training_errors=errors,
        formation_seconds=time.perf_counter()-began,
        allocation_is_inferred_from_current_neural_encoding_readout=True)
    graph['formation_audit']=audit
    if errors:raise ValueError(f'observed-profile model leaves{errors} actual TRAIN errors; preserve failure, no teacher repair')
    return graph,audit
