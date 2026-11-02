"""Strong Transformer baselines with explicitly priced task-code growth.

Reuse the actual TinyTransformer computation, widening to64/96 and3layers.
The same class serves ordinary learning and Replay; the runner owns replay.
No experiments run at import. No correct operations or task answers are built
in. Dynamic growth adds only random task-code rows, preserving all old weights.

MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
from numbers import Integral

import torch
from torch import nn

from torch_baselines import TinyTransformer


MAX_TASKS = 64


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


class GrowingTaskTransformer(TinyTransformer):
    """Actual shared Transformer; each newly allocated task costs d_model P.

    ``vocab`` counts NON-PAD symbols (1..vocab); PAD=0 is added internally.
    The source TinyTransformer instead takes the total embedding-row count.
    ``set_stage(stage)`` grows when stage>=n_tasks. Rebuild the optimizer after
    such growth because the task embedding Parameter changes. No old task row,
    attention block, head, or other learned state is reinitialized.

    Full state_dict includes two integer bookkeeping buffers. Dynamic task
    shape is restored before strict loading; stored stage is restored after
    all weights load. No task-specific warmstart/reset occurs during loading.
    Modular/adapter policies remain separate competitors owned by the runner.
    """

    def __init__(self, n_states, vocab=6, n_tasks=6, max_len=18, d_model=64):
        if n_states not in (8,16,32) or vocab not in range(1,7) or max_len != 18:
            raise ValueError("benchmark supports N8/16/32,1..6 non-PAD symbols,max_len18")
        if d_model not in (64,96) or isinstance(n_tasks,bool) or not isinstance(n_tasks,Integral) or not 1<=n_tasks<=MAX_TASKS:
            raise ValueError("choose width64/96 and1..64 initial task codes")
        self.max_nonpad_symbol = int(vocab)
        self.width = int(d_model)
        self.layers_count = 3
        self.heads_count = 4
        super().__init__(n_states=n_states, vocab=vocab+1, max_len=max_len,
                         d_model=d_model, n_heads=4, n_layers=3,
                         n_tasks=int(n_tasks), adapter=False)
        self.register_buffer("_saved_task_stage", torch.tensor(0,dtype=torch.long))
        self.register_buffer("_task_growth_events", torch.tensor(0,dtype=torch.long))
        self.last_growth_report = dict(grew=False, optimizer_rebuild_required=False)

    def _resize_task_codes(self, count, loading=False):
        count=int(count)
        if not 1<=count<=MAX_TASKS:
            raise ValueError("task-code allocation exceeds finite64-task bound")
        old=self.tasks.weight
        previous=int(self.n_tasks)
        if count==previous:
            return
        if loading:
            # Shape restoration only; strict load supplies every saved value.
            values=old.new_zeros((count,self.width))
        else:
            if count<previous:
                raise ValueError("learning cannot silently remove old task codes")
            rows=old.new_empty((count-previous,self.width))
            nn.init.normal_(rows)  # same neutral initialization as Embedding
            values=torch.cat((old.detach(),rows),dim=0)
        self.tasks.weight=nn.Parameter(values,requires_grad=old.requires_grad)
        self.tasks.num_embeddings=count
        self.n_tasks=count
        if not loading and hasattr(self,"_task_growth_events"):
            self._task_growth_events.add_(1)
        self.last_growth_report=dict(
            grew=not loading and count>previous,
            loading_shape_restoration_only=bool(loading), old_task_codes=previous,
            new_task_codes=count, delta_parameter_scalars=(count-previous)*self.width,
            new_parameter_bytes=self.tasks.weight.numel()*self.tasks.weight.element_size(),
            old_embedding_scalars_copied=0 if loading else old.numel(),
            optimizer_rebuild_required=True,
            old_computational_weights_preserved=not loading)

    def set_stage(self, stage, new_token_ids=None):
        if isinstance(stage,bool) or not isinstance(stage,Integral) or not 0<=stage<MAX_TASKS:
            raise ValueError("stage must be an integer task ID in0..63")
        # TinyTransformer.__init__ dispatches here once at stage0. The saved
        # buffers do not yet exist then; no growth/reset is needed.
        if stage>=self.n_tasks:
            self._resize_task_codes(int(stage)+1)
        elif hasattr(self,"last_growth_report"):
            self.last_growth_report=dict(grew=False,optimizer_rebuild_required=False,
                                         delta_parameter_scalars=0,preallocated_task_code=True)
        super().set_stage(int(stage),new_token_ids=new_token_ids)
        if hasattr(self,"_saved_task_stage"):
            self._saved_task_stage.fill_(int(stage))
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "tasks.weight" not in state_dict or state_dict["tasks.weight"].ndim!=2:
            raise ValueError("complete saved task-code tensor required")
        count,width=state_dict["tasks.weight"].shape
        if int(width)!=self.width:
            raise ValueError("saved width differs; select its actual64/96 configuration")
        self._resize_task_codes(int(count),loading=True)
        result=super().load_state_dict(state_dict,strict=strict,assign=assign)
        stage=int(self._saved_task_stage.item())
        if not 0<=stage<self.n_tasks:
            raise ValueError("saved stage exceeds actual restored task-code allocation")
        # This is the PURE shared Transformer; base set_stage changes only
        # gradient flags/masks, never copies heads or changes learned values.
        super().set_stage(stage)
        return result

    def expected_parameter_count(self):
        d=self.width
        return (8*self.layers_count*d*d
                +(11*self.layers_count+self.vocab+2*self.n_states+self.n_tasks+self.max_len+5)*d
                +self.n_states)

    def resource_counts(self, optimizer=None):
        counts=self.parameter_counts()
        expected=self.expected_parameter_count()
        if counts["total"]!=expected:
            raise RuntimeError("actual architecture parameters differ from declared formula")
        return dict(**counts,
            model_parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
            registered_buffer_tensor_bytes=sum(b.numel()*b.element_size() for b in self.buffers()),
            allocated_task_codes=self.n_tasks, started_task_id=self.stage,
            task_embedding_scalars=self.tasks.weight.numel(),
            deltaP_per_newly_allocated_task_code=self.width,
            deltaP_when_using_preallocated_code=0,
            preallocated_codes_already_in_total_parameter_cost=True,
            source_architecture="torch_baselines.TinyTransformer, shared attention/FFN/head",
            architecture=dict(width=self.width,layers=3,heads=4,ffn_width=2*self.width,
                              dropout=0,max_len=18,nonpad_vocab=self.max_nonpad_symbol),
            optimizer_tensor_bytes=None if optimizer is None else _tensor_bytes(optimizer.state_dict()),
            replay_storage_external_to_model=True,
            masks_and_snapshots_tensor_bytes=_tensor_bytes(dict(
                masks=self._row_masks,snapshots=self._row_snapshots)),
            last_growth_report=dict(self.last_growth_report))


def make_transformer(n_states, width=64, n_tasks=6, vocab=6):
    return GrowingTaskTransformer(n_states,vocab=vocab,n_tasks=n_tasks,d_model=width)
