"""Compact native models for parameter-growth experiments; no runs at import.

Each learned expert stores D destination coordinates, not D² score parameters:
W[k,i,j]=-8*(j-xi[k,i])². Rearrangement gives its hard matching by sorting xi;
xi_i=permutation(i) represents every permutation. Parent graph/materialization
still use their actual checked matching and old-relation transactions.

The SAME representation is available to the ordinary constrained control.
Coordinate compression is not claimed as a uniquely topological mechanism.
Root owns frozen/module wrappers, optimizer rebuilding, tests and execution.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from semantic_promotion_model import SemanticPromotionModel
from continuous_native_model import ContinuousNativeModel
from topology_model import assignment
from train_monodromy_extension import kinetic_moments


MAX_COMPACT_BANK = 48
SCORE_SCALE = 8.


class _CompactOperations:
    def __getattr__(self, name):
        if name == "operations":
            parameters = self.__dict__.get("_parameters", {})
            if "operation_coordinates" in parameters:
                coordinates = parameters["operation_coordinates"]
                destinations = torch.arange(coordinates.shape[1],device=coordinates.device,dtype=coordinates.dtype)
                return -SCORE_SCALE*(destinations[None,None,:]-coordinates[:,:,None]).square()
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name == "operations" and self.__dict__.get("_compact_enabled",False):
            if not torch.is_tensor(value) or value.ndim != 3 or value.shape[1] != value.shape[2]:
                raise ValueError("parent operation transaction must supply square dense scores")
            count,dimension,_=value.shape
            if not 1<=count<=self.max_bank or not 1<=dimension<=64:
                raise ValueError("compact score transaction exceeds declared bank/state bound")
            maps=np.stack([assignment(matrix) for matrix in value])
            coordinates=torch.as_tensor(maps,device=value.device,dtype=value.dtype).clone()
            preserve=self.__dict__.get("_compact_growth_source")
            if preserve is not None and dimension==preserve["new_dim"]:
                # Preserve fractional old coordinates, not just their ranks.
                old=preserve["parameter"]
                coordinates[:old.shape[0],:old.shape[1]]=old.detach()
            nn.Module.__setattr__(self,"operation_coordinates",nn.Parameter(
                coordinates,requires_grad=bool(value.requires_grad)))
            return
        super().__setattr__(name,value)

    def _finish_compact(self, requested_bank, alphabets):
        if not 1<=int(requested_bank)<=MAX_COMPACT_BANK:
            raise ValueError("finite compact bank cap must lie in1..48")
        dense=self._parameters.pop("operations")
        maps=np.stack([assignment(matrix) for matrix in dense])
        self.register_parameter("operation_coordinates",nn.Parameter(
            torch.as_tensor(maps,device=dense.device,dtype=dense.dtype).clone()))
        self.max_bank=int(requested_bank)
        self._compact_enabled=True
        self.task_alphabets={int(t):tuple(int(a) for a in tokens) for t,tokens in
            (alphabets.items() if alphabets is not None else
             ((t,range(1,self.vocab+1)) for t in range(self.n_tasks)))}
        if (set(self.task_alphabets)!=set(range(self.n_tasks))
                or any(not tokens or len(set(tokens))!=len(tokens)
                       or any(not 1<=a<=self.vocab for a in tokens)
                       for tokens in self.task_alphabets.values())):
            raise ValueError("one observed input alphabet per allocated task required")
        if callable(getattr(self,"_invalidate_derived_caches",None)):
            self._invalidate_derived_caches()

    def set_stage(self, stage, new_token_ids=None):
        stage=int(stage)
        if not 0<=stage<self.n_tasks:
            raise ValueError("stage outside actually allocated task embeddings")
        self.stage=stage
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self

    @torch.no_grad()
    def materialize(self, *args, **kwargs):
        coordinates=self.operation_coordinates
        original_count=self.n_ops
        original_call=self.call
        flags={name:p.requires_grad for name,p in self.named_parameters()}
        try:
            report=super().materialize(*args,**kwargs)
        except BaseException:
            self.operation_coordinates=coordinates
            self.call,self.n_ops=original_call,original_count
            for name,p in self.named_parameters():p.requires_grad_(flags[name])
            raise
        if not report.get("success"):
            # Parent's dense rollback is a generated Tensor. Restore the REAL
            # original coordinate Parameter so existing optimizer references
            # and all fractional coordinate values survive a failed attempt.
            self.operation_coordinates=coordinates
            self.call,self.n_ops=original_call,original_count
            for name,p in self.named_parameters():p.requires_grad_(flags[name])
        costs=report["costs"]
        costs["parent_generated_score_tensor_bytes"]=costs.get("expert_parameter_bytes")
        costs["expert_parameter_bytes"]=self.operation_coordinates.numel()*self.operation_coordinates.element_size()
        costs["temporary_dense_score_bytes"]=self.n_ops*self.d*self.d*self.operation_coordinates.element_size()
        report.update(operation_representation="D destination coordinates per expert",
                      ordinary_constraint_control_has_identical_compression=True,
                      coordinate_parameter_identity_restored_on_failure=not report.get("success"))
        if self.materialization_history:
            # Parent appends before this wrapper replaces its generated-score
            # byte count with the actual persistent coordinate parameter cost.
            self.materialization_history[-1]=copy.deepcopy(report)
        return report

    @torch.no_grad()
    def grow_safe(self, new_dim):
        """Use the existing checked growth transaction, preserving real xi.

        Dense temporary discovery scores are actually generated and charged by
        the old routine; only their resulting compact coordinates persist.
        Old root logits remain exactly represented in the old top-left block.
        """
        from workspace_refinement import grow_workspace
        original=dict(coordinates=self.operation_coordinates,encoder=self.encoder,
                      decoder=self.decoder,call=self.call,d=self.d,n_ops=self.n_ops,eye=self._eye)
        flags={name:p.requires_grad for name,p in self.named_parameters()}
        self._compact_growth_source=dict(parameter=self.operation_coordinates,new_dim=int(new_dim))
        try:
            report=grow_workspace(self,int(new_dim))
        except BaseException:
            self.operation_coordinates=original["coordinates"]
            self.encoder,self.decoder,self.call=original["encoder"],original["decoder"],original["call"]
            self.d,self.n_ops,self._eye=original["d"],original["n_ops"],original["eye"]
            for name,p in self.named_parameters():p.requires_grad_(flags[name])
            raise
        finally:
            self._compact_growth_source=None
        if not report.get("success"):
            self.operation_coordinates=original["coordinates"]
            self.encoder,self.decoder,self.call=original["encoder"],original["decoder"],original["call"]
            self.d,self.n_ops,self._eye=original["d"],original["n_ops"],original["eye"]
            for name,p in self.named_parameters():p.requires_grad_(flags[name])
        if callable(getattr(self,"_invalidate_derived_caches",None)):
            self._invalidate_derived_caches()
        report["compact_coordinate_growth"]=True
        report["persistent_coordinate_scalars"]=self.operation_coordinates.numel()
        report["temporary_dense_initialization_is_not_free"]=True
        return report

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if "operations" in state_dict or "operation_coordinates" not in state_dict:
            raise ValueError("compact state must store real coordinates and no dense operations Parameter")
        state=state_dict["operation_coordinates"]
        if (state.ndim!=2 or not torch.isfinite(state).all()
                or not 1<=state.shape[0]<=self.max_bank or not 1<=state.shape[1]<=64):
            raise ValueError("invalid saved compact coordinates")
        count,dimension=state.shape
        if (tuple(state_dict["encoder"].shape)!=(self.n,dimension)
                or tuple(state_dict["decoder"].shape)!=(dimension,self.n)
                or tuple(state_dict["token.weight"].shape)!=(self.vocab+1,self.call.in_features)
                or tuple(state_dict["task.weight"].shape)!=(self.n_tasks,self.call.in_features)):
            raise ValueError("use actual shape/configuration factory for saved model")
        device,dtype=self.encoder.device,self.encoder.dtype
        for name in ("encoder","decoder"):
            if tuple(getattr(self,name).shape)!=tuple(state_dict[name].shape):
                setattr(self,name,nn.Parameter(torch.empty_like(state_dict[name],device=device,dtype=dtype)))
        if tuple(self.operation_coordinates.shape)!=tuple(state.shape):
            self.operation_coordinates=nn.Parameter(torch.empty_like(state,device=device,dtype=dtype))
        width=self.slots*(int(count)+1)
        if tuple(state_dict["call.weight"].shape)!=(width,self.call.in_features):
            raise ValueError("saved call head differs from coordinate bank")
        if self.call.out_features!=width:
            self.call=nn.Linear(self.call.in_features,width,device=device,dtype=dtype)
        self.d,self.n_ops=int(dimension),int(count)
        self._eye=None
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def extra_state_dict(self):
        extra=super().extra_state_dict()
        extra["compact_operations"]=dict(kind="destination-coordinate-squared-distance-v1",
            coordinate_state_key="operation_coordinates",score_scale=SCORE_SCALE,
            n_tasks=self.n_tasks,vocab=self.vocab,max_bank=self.max_bank,
            task_alphabets={t:list(a) for t,a in self.task_alphabets.items()},
            actual_dense_operation_parameters=False,
            checkpoint_loader="parameter_native_models.compact_model_from_checkpoint",
            model_kind="continuous" if hasattr(self,"path_omega") else "ordinary_constraint")
        if hasattr(self,"factored_monodromy"):
            extra["compact_operations"]["factored_monodromy"]=self.factored_monodromy
        if "continuous_monodromy" in extra:
            extra["continuous_monodromy"]["checkpoint_loader"]="parameter_native_models.compact_model_from_checkpoint"
        return extra

    def load_extra_state_dict(self,extra):
        meta=extra.get("compact_operations")
        if (not meta or meta.get("kind")!="destination-coordinate-squared-distance-v1"
                or float(meta.get("score_scale",-1))!=SCORE_SCALE
                or int(meta.get("n_tasks",-1))!=self.n_tasks or int(meta.get("vocab",-1))!=self.vocab):
            raise ValueError("saved compact operation configuration differs")
        self.task_alphabets={int(t):tuple(a) for t,a in meta["task_alphabets"].items()}
        super().load_extra_state_dict(extra)

    def compact_resource_counts(self):
        parameters=list(self.parameters())
        return dict(total=sum(p.numel() for p in parameters),
            trainable=sum(p.numel() for p in parameters if p.requires_grad),
            frozen=sum(p.numel() for p in parameters if not p.requires_grad),
            parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
            operation_coordinate_scalars=self.operation_coordinates.numel(),
            generated_score_tensor_scalars=self.n_ops*self.d*self.d,
            generated_scores_are_execution_work_not_persistent_parameters=True,
            reference_array_bytes=self.representation_state_bytes(),
            n_ops=self.n_ops,D=self.d,n_tasks=self.n_tasks,emb_dim=self.call.in_features,
            topology_extra_trainable_scalars=4 if hasattr(self,"path_omega") else 0)


class CompactSemanticPromotionModel(_CompactOperations,SemanticPromotionModel):
    def __init__(self,n_states,vocab=6,n_tasks=6,emb_dim=48,n_ops=1,max_bank=48,
                 latent_factor=1,task_alphabets=None,**kwargs):
        self._compact_enabled=False
        # Old default bank bound is left intact; the explicit local model cap
        # is installed after a legal small initial parent construction.
        if not 1<=n_ops<=24:raise ValueError("construct a small initial bank; factory resizes saved banks")
        kwargs.setdefault("constraint",True)
        super().__init__(n_states,vocab=vocab,n_tasks=n_tasks,emb_dim=emb_dim,n_ops=n_ops,
                         max_bank=min(int(max_bank),24),latent_factor=latent_factor,**kwargs)
        self._finish_compact(max_bank,task_alphabets)


class CompactContinuousNativeModel(_CompactOperations,ContinuousNativeModel):
    def __init__(self,n_states,vocab=6,n_tasks=6,emb_dim=48,n_ops=1,max_bank=48,
                 latent_factor=1,task_alphabets=None,homotopy_lift_cache_capacity=128,
                 factored_monodromy=False,**kwargs):
        self._compact_enabled=False
        self.factored_monodromy=bool(factored_monodromy)
        if self.factored_monodromy and (not homotopy_lift_cache_capacity or kwargs.get('geometry_cache_capacity',0)):
            raise ValueError('factored execution requires an actual first-lift cache and its own uncached quadrature')
        if not 1<=n_ops<=24:raise ValueError("construct a small initial bank; factory resizes saved banks")
        super().__init__(n_states,vocab=vocab,n_tasks=n_tasks,emb_dim=emb_dim,n_ops=n_ops,
            max_bank=min(int(max_bank),24),latent_factor=latent_factor,
            homotopy_lift_cache_capacity=homotopy_lift_cache_capacity,**kwargs)
        self._finish_compact(max_bank,task_alphabets)

    def set_homotopy_lift_cache(self,capacity,shared_cache=None):
        if self.factored_monodromy and capacity:
            from factored_monodromy_cache import FactoredMonodromyCache
            if shared_cache is None:
                shared_cache=FactoredMonodromyCache(capacity=int(capacity))
            elif not isinstance(shared_cache,FactoredMonodromyCache):
                raise ValueError('factored execution cannot share an unfactored proof namespace')
        return super().set_homotopy_lift_cache(capacity,shared_cache=shared_cache)

    def geometry_regularizer(self):
        """Existing action rule with actual configurable six-task alphabets."""
        self._geometry_costs["geometry_regularizer_calls"]+=1
        if not torch.isfinite(self.path_omega).all() or self.path_omega.abs().max().item()>8:
            raise ValueError("shared path weights exceed existing finite envelope")
        start=time.perf_counter()
        graph=self.graph(self.stage,self.task_alphabets[self.stage])
        active=sorted(set(int(k) for k in graph["routes"].ravel()))
        classes=tuple(sorted(set(tuple(int(q) for q in graph["permutations"][k]) for k in active)))
        signature=int(self.d),classes
        self._geometry_costs["geometry_class_graph_extractions"]+=1
        self._geometry_costs["geometry_class_bank_assignments"]+=self.n_ops
        self._geometry_costs["geometry_class_check_seconds"]+=time.perf_counter()-start
        if signature!=self._moment_signature:
            start=time.perf_counter()
            if self._path_moment_cache is None:
                if self.factored_monodromy:
                    from factored_monodromy_cache import factored_kinetic_moments
                    self._moments=factored_kinetic_moments(classes)
                else:
                    self._moments=kinetic_moments(classes)
                actual_points=self._moments["quadrature_point_strands"]
            else:
                before=self._path_moment_cache.statistics()["point_strands_actually_computed_on_misses"]
                self._moments=self._path_moment_cache.get(classes)
                actual_points=self._path_moment_cache.statistics()["point_strands_actually_computed_on_misses"]-before
            self._moment_signature=signature
            self._moment_linear=self.path_omega.new_tensor(self._moments["linear"])
            self._moment_quadratic=self.path_omega.new_tensor(self._moments["quadratic"])
            self._geometry_costs["geometry_moment_rebuilds"]+=1
            self._geometry_costs["geometry_quadrature_point_strands"]+=actual_points
            self._geometry_costs["geometry_quadrature_reference_point_strands"]+=self._moments["quadrature_point_strands"]
            self._geometry_costs["geometry_moment_seconds"]+=time.perf_counter()-start
        elif self._moment_linear.device!=self.path_omega.device or self._moment_linear.dtype!=self.path_omega.dtype:
            self._moment_linear=self.path_omega.new_tensor(self._moments["linear"])
            self._moment_quadratic=self.path_omega.new_tensor(self._moments["quadratic"])
        start=time.perf_counter()
        omega=self.path_omega
        raw=self._moments["constant"]+2*torch.dot(self._moment_linear,omega)+omega@self._moment_quadratic@omega
        self._geometry_costs["geometry_torch_objective_seconds"]+=time.perf_counter()-start
        return raw/self._moments["normalizer"]

    @torch.no_grad()
    def _actual_bank(self):
        # Coordinate identity/version is stable; generated dense scores are
        # temporary tensors and must not force a false cache miss every call.
        parameter=self.operation_coordinates
        signature=(id(parameter),parameter.data_ptr(),int(parameter._version),
                   tuple(parameter.shape),str(parameter.device),str(parameter.dtype),self.d)
        if signature!=self._bank_parameter_signature:
            started=time.perf_counter()
            self._hard_bank=[np.arange(self.d,dtype=np.int64)]+[assignment(w) for w in self.operations]
            self._bank_parameter_signature=signature
            self._geometry_costs["hard_bank_assignment_batches"]+=1
            self._geometry_costs["hard_bank_hungarian_assignments"]+=self.n_ops
            self._geometry_costs["hard_bank_assignment_seconds"]+=time.perf_counter()-started
        return self._hard_bank


def compact_model_from_checkpoint(checkpoint,device="cpu"):
    """Restore OWN actual compact weights AND retention/geometry metadata.

    The caller performs trusted torch.load. This factory makes no file/network
    calls. Cache maps are not fabricated or restored warm; parent rules apply.
    """
    state=checkpoint["model"]
    extra=checkpoint.get("model_extra",{})
    meta=extra.get("compact_operations",{})
    if "operation_coordinates" not in state:raise ValueError("actual compact checkpoint required")
    n,dimension=state["encoder"].shape
    if dimension%n or not 1<=dimension//n<=2:raise ValueError("saved workspace exceeds declared N→2N scope")
    emb=state["token.weight"].shape[1]
    count=state["operation_coordinates"].shape[0]
    width=state["call.weight"].shape[0]
    if width%(count+1):raise ValueError("saved head/bank shape mismatch")
    options=dict(vocab=state["token.weight"].shape[0]-1,n_tasks=state["task.weight"].shape[0],
        emb_dim=emb,n_ops=1,max_bank=int(meta.get("max_bank",48)),latent_factor=dimension//n,
        slots=width//(count+1),task_alphabets=meta.get("task_alphabets"),
        **extra.get("promotion_settings",{}))
    if "path_omega" in state:
        geometry=extra.get("continuous_monodromy",{})
        policy=geometry.get("homotopy_lift_cache_policy")
        options.update(initial_omega=state["path_omega"].detach().cpu().tolist(),
            continuous_lift_seconds=geometry.get("continuous_lift_seconds",110.),
            geometry_cache_capacity=geometry.get("moment_cache_capacity",0),
            homotopy_lift_cache_capacity=0 if policy is None else int(policy["capacity"]),
            factored_monodromy=bool(meta.get('factored_monodromy',False)))
        cls=CompactContinuousNativeModel
    else:
        cls=CompactSemanticPromotionModel
    model=cls(n,**options).to(device)
    model.load_state_dict(state,strict=True)
    model.load_extra_state_dict(extra)
    return model
