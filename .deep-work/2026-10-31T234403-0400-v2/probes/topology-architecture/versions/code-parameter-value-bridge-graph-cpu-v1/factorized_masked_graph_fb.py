"""NO-RUN cost candidate: exact fcba all-row masked forward/backward.

Reuses the original _transition without a new parameter, compiler or optimizer.
Each transition is row-separable (feature LayerNorm, within-example memory and
CALL softmax), so selecting active rows equals computing all rows then masking
carry in real arithmetic. Floating GEMM/reduction order may differ; actual CPU
and CUDA output/gradient contracts are required. No validation bool/nonzero or
CPU tensor extraction occurs inside masked_forward_value.

Only fixed weights/gradients are exercised by the companion contract. A future
training caller must keep clipping and Adam eager, after graph.replay(). This
module NEVER calls optimizer.step; F/B speed is not full-training speed.
"""
import torch
from torch.nn import functional as F

CORE_SHA256='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
OUTPUT_ATOL,OUTPUT_RTOL=2e-5,2e-5
GRAD_ATOL,GRAD_RTOL=2e-5,2e-4
BUCKETS=(8,16,48)
BATCH=64


def masked_forward_value(model,initial,tokens,lengths):
    """Pure device operations only; terminal output at each row's true length."""
    carry=model.initial_carry(initial)
    embedded=model.token_embedding(tokens)
    for step in range(tokens.shape[1]):
        updated,_=model._transition(carry,embedded[:,step],'soft')
        active=step<lengths
        carry={key:torch.where(active.reshape((tokens.shape[0],)+(1,)*(value.ndim-1)),
                               updated[key],value)
               for key,value in carry.items()}
    return F.log_softmax(model.readout(carry['h']),-1),carry


def objective(model,forward,initial,tokens,lengths,targets,anchor):
    # Matches the actual training driver's computation, including anchor0's
    # charged target lookup and MSE followed by multiplication by zero.
    output,carry=forward(model,initial,tokens,lengths)
    nll=F.nll_loss(output,targets)
    target=model.encode_state(targets).detach()
    binding=(carry['h']-target).square().mean()
    return nll+anchor*binding,output,nll,binding


def original_forward_value(model,initial,tokens,lengths):
    return model.forward_value(initial,tokens,lengths,execution='soft')


def original_step(model,initial,tokens,lengths,targets,anchor):
    # Exact existing training zeroing policy, but NO clipping/Adam/update.
    model.zero_grad(set_to_none=True)
    loss,output,nll,binding=objective(model,original_forward_value,initial,tokens,lengths,targets,anchor)
    loss.backward()
    return output.detach(),loss.detach(),nll.detach(),binding.detach()


def allocate_gradient_buffers(model):
    # Run outside graph capture, charge allocations. Primary actual batches must
    # have the same all-present parameter usage pattern as the original loop.
    for parameter in model.parameters():
        parameter.grad=torch.zeros_like(parameter,memory_format=torch.preserve_format)


def masked_step(model,initial,tokens,lengths,targets,anchor):
    # These zero_ operations MUST be captured; replay must never accumulate the
    # previous draw's gradients. Buffers were allocated before warmup/capture.
    for parameter in model.parameters():
        parameter.grad.zero_()
    loss,output,nll,binding=objective(model,masked_forward_value,initial,tokens,lengths,targets,anchor)
    loss.backward()
    return output.detach(),loss.detach(),nll.detach(),binding.detach()
