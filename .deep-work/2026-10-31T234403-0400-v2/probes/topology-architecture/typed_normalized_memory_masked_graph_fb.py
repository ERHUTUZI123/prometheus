"""Pure masked typed A/B normalized-memory F/B; cached logits are the ONLY emission.

Fresh candidate, NLL only; no teacher/binding/argmax/state label injected.
Computes allrows for a fixedbucket, masks ALL carry fields including logp.
Uses exact newmodel _transition; no bool/nonzero/item/CPU inside pure path.
Gradstorage is allocated once outsidecapture and zeroed in everycapturedF/B.
"""
import torch
from torch.nn import functional as F
OUTPUT_ATOL,OUTPUT_RTOL=2e-5,2e-5
GRAD_ATOL,GRAD_RTOL=2e-5,2e-4
BATCH=64
BUCKETS=(16,48)

def masked_forward_value(model,initial,tokens,lengths):
 carry=model.initial_carry(initial);embedded=model.token_embedding(tokens)
 for step in range(tokens.shape[1]):
  updated,_=model._transition(carry,embedded[:,step],'soft');active=step<lengths
  carry={key:torch.where(active.reshape((tokens.shape[0],)+(1,)*(value.ndim-1)),updated[key],value)for key,value in carry.items()}
 return carry['logp'],carry

def original_forward_value(model,initial,tokens,lengths):return model.forward_value(initial,tokens,lengths,execution='soft')
def objective(model,forward,initial,tokens,lengths,targets):
 output,_=forward(model,initial,tokens,lengths)
 return F.nll_loss(output,targets),output

def allocate_gradient_buffers(model):
 for parameter in model.parameters():parameter.grad=torch.zeros_like(parameter,memory_format=torch.preserve_format)

def masked_step(model,initial,tokens,lengths,targets):
 for parameter in model.parameters():parameter.grad.zero_()
 loss,output=objective(model,masked_forward_value,initial,tokens,lengths,targets);loss.backward()
 return output.detach(),loss.detach()
