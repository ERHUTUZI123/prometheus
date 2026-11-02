"""Execution-only v2: unchanged dual rule, FP64 private read and solve_ex.

θ/carry/caller remain FP32. All forward solver infos must be checked by the
trusted driver outside capture. This is not a new trained architecture arm.
"""
import torch
import owned_dual_v1_core as v1
CHECKPOINT_VERSION='continuous-primitive-block-typed-dual-memory-fp64-solve-ex-v2'
POLICY='typed-memory-actions/input-address+key-composition-dual/raw-value/group8/fp64-solve-ex-v2'
EXECUTION_POLICY='private-read-FP64/solve_ex-check_errors_false/all-info-checked-outside-capture/v2'
LOAD,STORE,ADDRESS=v1.LOAD,v1.STORE,v1.ADDRESS
class ContinuousPrimitiveBlock(v1.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    organization_policy=POLICY
    def __init__(self,*args,action_policy=POLICY,input_only_address=True,dual_epsilon=1e-4,dual_ridge=1e-4,execution_policy=EXECUTION_POLICY,internal_read_dtype='float64',returned_read_dtype='state-float32',**kwargs):
        if action_policy!=POLICY or execution_policy!=EXECUTION_POLICY or internal_read_dtype!='float64'or returned_read_dtype!='state-float32':raise ValueError('one explicit FP64 solve_ex execution policy')
        super().__init__(*args,action_policy=v1.POLICY,input_only_address=input_only_address,dual_epsilon=dual_epsilon,dual_ridge=dual_ridge,**kwargs)
        if self.encoder.weight.dtype!=torch.float32:raise ValueError('theta and episode state remain FP32')
        self.dual_identity=self.dual_identity.to(dtype=torch.float64)
        self._forward_solve_infos=[]
    def configuration(self):
        cfg=super().configuration();cfg.update(action_policy=POLICY,execution_policy=EXECUTION_POLICY,internal_read_dtype='float64',returned_read_dtype='state-float32');return cfg
    @classmethod
    def from_checkpoint(cls,payload,*,expected_action_policy=POLICY):
        cfg=dict(payload['configuration'])
        if expected_action_policy!=POLICY or payload.get('version')!=CHECKPOINT_VERSION or payload.get('organization_policy')!=POLICY or cfg.get('action_policy')!=POLICY or cfg.get('execution_policy')!=EXECUTION_POLICY or cfg.get('internal_read_dtype')!='float64'or cfg.get('returned_read_dtype')!='state-float32'or cfg.get('input_only_address')is not True or cfg.get('dual_epsilon')!=1e-4 or cfg.get('dual_ridge')!=1e-4 or 'coverage_normalization'in cfg:raise ValueError('strict execution-v2 checkpoint required; v1 is not v2')
        t=payload['state_dict']['encoder.weight'];model=cls(**cfg,device=t.device,dtype=t.dtype);model.load_state_dict(payload['state_dict'],strict=True);return model
    def clear_forward_solve_infos(self):self._forward_solve_infos=[]
    def forward_solve_info_tensors(self):return tuple(self._forward_solve_infos)
    def dual_read(self,values,key_composition,query):
        # Every read-normalization/Gram/RHS/ridge/solve/value-mix operation is
        # FP64; only the returned read/trace scalars cast back to state dtype.
        C=key_composition.to(torch.float64);V=values.to(torch.float64);q=query.to(torch.float64)
        z=C.sum(-1);denominator=z.clamp_min(self.dual_epsilon);Cbar=C/denominator[:,:,None];Vbar=V/denominator[:,:,None]
        gram=torch.bmm(Cbar,Cbar.transpose(1,2))+self.dual_ridge*self.dual_identity
        rhs=torch.bmm(Cbar,q[:,:,None]);solution,info=torch.linalg.solve_ex(gram,rhs,check_errors=False)
        self._forward_solve_infos.append(info)
        b=solution.squeeze(-1);read=torch.bmm(b[:,None],Vbar).squeeze(1)
        return read.to(values.dtype),b.to(values.dtype),z.to(values.dtype),Cbar,Vbar
    def resource_report(self,*args,**kwargs):
        result=super().resource_report(*args,**kwargs);n=result['active_steps'];result['registered_buffer_bytes']=512
        result['dual_memory'].update(shared_nonpersistent_identity_bytes=512,execution_policy=EXECUTION_POLICY,internal_read_dtype='float64',theta_and_carry_dtype='float32',returned_read_dtype='float32',forward_info_tensor_dtype='int32',forward_info_scalars_per_read=1,forward_info_check_scope='driver checks every forward info outside capture; backward solver internals not exposed',explicit_FP64_intermediate_elements_per_example_read=776,explicit_FP64_intermediate_logical_bytes_per_example_read=6208,intermediate_scope='logical named temporaries not allocation peak; autograd/workspace/caching measured separately',normalization_divisions_dtype='float64',solver_API='torch.linalg.solve_ex(check_errors=False)')
        result['execution_policy']=EXECUTION_POLICY;return result
