"""Learned ordinary primitives with context-dependent calls; no import-time run.

This is a reusable architecture component, not a training or benchmark result.
Raw observed initial states/tokens are the only inputs. No operation, route,
intermediate state, or hidden generator state is supplied to this module.
The K learned D-by-D row-logit operators may be nonbijective. Identity is the
fixed call0; learned primitives are calls1..K. Routing and recurrent mixtures
are ordinary neural mechanisms; this file makes no independent novelty claim.

Soft execution uses probabilistic E/calls/operators/R. Straight-through uses
hard-forward, soft-backward calls/operator rows, with SOFT E/R for usable NLL
gradients. Native uses actual argmax E/calls/operator rows/R under no_grad.
The same weights are used in every mode. Native memory is still the learned
continuous GRU state. Memory resets to zero at each forward, and R reads h only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContextualPrimitiveConfig:
    n_states: int
    n_tokens: int
    internal_states: int
    n_primitives: int
    token_dim: int = 16
    state_dim: int = 16
    memory_dim: int = 16
    caller_hidden: int = 32
    n_slots: int = 4
    routing: str = "dynamic"
    initialization_std: float = .02


class ContextualPrimitiveBlock(nn.Module):
    """Finite latent register, ordinary primitive bank, contextual caller/GRU.

    forward(initial_state[B], tokens[B,L], lengths[B], execution=...) returns
    probabilities[B,N] by default. output='logits' returns finite log-probs.
    return_traces=True returns (output, trace_dict). n_tokens is the TOTAL
    number of embedding rows: active IDs are0..n_tokens-1, with no extra PAD.
    Inactive padding IDs are ignored and may be outside that range.

    Each token executes n_slots shared calls with a learned slot embedding added
    to its token embedding. Calls read CURRENT h and fixed pre-token m. One GRU
    update at token end reads pre-token h/raw token/m, preserving S=1 semantics.
    Dynamic calls read [h @ state_embedding, token_embedding+slot_embedding, m]. Static calls
    zero ONLY the first/last inputs to the SAME caller MLP. Both modes execute
    the same GRU and retain its memory; static R/calls cannot read that memory.
    Neither mode provides a hidden memory-to-output shortcut.
    """

    checkpoint_version = "contextual-primitive-block-v1"

    def __init__(self, n_states: int, n_tokens: int, internal_states: int,
                 n_primitives: int, token_dim: int = 16, state_dim: int = 16,
                 memory_dim: int = 16, caller_hidden: int = 32,
                 routing: str = "dynamic", initialization_std: float = .02, n_slots: int = 4,
                 *, device: Any = None, dtype: torch.dtype | None = None):
        super().__init__()
        dimensions = {"n_states": n_states, "n_tokens": n_tokens,
                      "internal_states": internal_states, "token_dim": token_dim,
                      "state_dim": state_dim, "memory_dim": memory_dim,
                      "caller_hidden": caller_hidden, "n_slots": n_slots}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in dimensions.values()):
            raise ValueError("state/vocabulary/embedding/GRU/caller dimensions must be positive integers")
        if not isinstance(n_primitives, int) or isinstance(n_primitives, bool) or n_primitives < 0:
            raise ValueError("n_primitives must be a nonnegative integer")
        if routing not in ("dynamic", "static"):
            raise ValueError("routing must be dynamic or static")
        if not 0 < initialization_std < 1:
            raise ValueError("finite positive initialization_std below1 required")
        self.config = ContextualPrimitiveConfig(n_states=n_states, n_tokens=n_tokens,
            internal_states=internal_states, n_primitives=n_primitives, token_dim=token_dim,
            state_dim=state_dim, memory_dim=memory_dim, caller_hidden=caller_hidden,
            n_slots=n_slots, routing=routing, initialization_std=initialization_std)
        self.n_states, self.n_tokens = n_states, n_tokens
        self.internal_states, self.n_primitives = internal_states, n_primitives
        self.token_dim, self.state_dim, self.memory_dim = token_dim, state_dim, memory_dim
        self.caller_hidden, self.routing = caller_hidden, routing
        self.n_slots = n_slots
        factory = {"device": device, "dtype": dtype}
        self.encoder_logits = nn.Parameter(torch.empty(n_states, internal_states, **factory))
        self.primitive_logits = nn.Parameter(torch.empty(n_primitives, internal_states, internal_states, **factory))
        self.readout_logits = nn.Parameter(torch.empty(internal_states, n_states, **factory))
        self.state_embedding = nn.Parameter(torch.empty(internal_states, state_dim, **factory))
        self.token_embedding = nn.Embedding(n_tokens, token_dim, **factory)
        self.slot_embedding = nn.Embedding(n_slots, token_dim, **factory)
        self.caller = nn.Sequential(
            nn.Linear(state_dim + token_dim + memory_dim, caller_hidden, **factory),
            nn.Tanh(), nn.Linear(caller_hidden, n_primitives + 1, **factory))
        self.memory_cell = nn.GRUCell(state_dim + token_dim, memory_dim, **factory)
        self.register_buffer("identity_operator", torch.eye(internal_states, **factory))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Data-independent initialization; no binding/routing/map is seeded."""
        for parameter in (self.encoder_logits, self.primitive_logits,
                          self.readout_logits, self.state_embedding):
            nn.init.normal_(parameter, std=self.config.initialization_std)
        self.token_embedding.reset_parameters()
        self.slot_embedding.reset_parameters()
        for layer in self.caller:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.memory_cell.reset_parameters()

    def configuration(self) -> dict[str, Any]:
        result = asdict(self.config)
        result["routing"] = self.routing
        return result

    def set_routing(self, routing: str) -> None:
        if routing not in ("dynamic", "static"):
            raise ValueError("routing must be dynamic or static")
        self.routing = routing

    @staticmethod
    def _hard_or_st(probabilities: torch.Tensor, execution: str) -> torch.Tensor:
        hard = F.one_hot(probabilities.argmax(-1), probabilities.shape[-1]).to(probabilities)
        if execution == "native":
            return hard
        # This ordering gives exactly hard forward values, avoiding subtract/add
        # roundoff while retaining the derivative of the soft probabilities.
        return hard + (probabilities - probabilities.detach())

    def _validate(self, initial_state: torch.Tensor, tokens: torch.Tensor,
                  lengths: torch.Tensor) -> None:
        if any(not isinstance(value, torch.Tensor) for value in (initial_state, tokens, lengths)):
            raise TypeError("initial_state, tokens and lengths must be tensors")
        if any(value.dtype != torch.long for value in (initial_state, tokens, lengths)):
            raise TypeError("initial_state, tokens and lengths must have torch.long dtype")
        if initial_state.ndim != 1 or tokens.ndim != 2 or lengths.ndim != 1:
            raise ValueError("expected initial_state[B], tokens[B,L], lengths[B]")
        if initial_state.shape[0] != tokens.shape[0] or lengths.shape != initial_state.shape:
            raise ValueError("input batch dimensions differ")
        device = self.encoder_logits.device
        if any(value.device != device for value in (initial_state, tokens, lengths)):
            raise ValueError("inputs and model must be on the same device")
        if bool(((initial_state < 0) | (initial_state >= self.n_states)).any()):
            raise ValueError("observed initial_state outside0..n_states-1")
        if bool(((lengths < 0) | (lengths > tokens.shape[1])).any()):
            raise ValueError("length outside0..sequence_length")
        active = torch.arange(tokens.shape[1], device=device)[None, :] < lengths[:, None]
        if bool((active & ((tokens < 0) | (tokens >= self.n_tokens))).any()):
            raise ValueError("active token outside0..n_tokens-1")

    def forward(self, initial_state: torch.Tensor, tokens: torch.Tensor,
                lengths: torch.Tensor, execution: str = "soft", *,
                return_traces: bool = False, output: str = "probabilities") -> Any:
        if execution not in ("soft", "straight_through", "native"):
            raise ValueError("execution must be soft, straight_through or native")
        if output not in ("probabilities", "logits"):
            raise ValueError("output must be probabilities or logits")
        self._validate(initial_state, tokens, lengths)
        if execution == "native":
            with torch.no_grad():
                return self._execute(initial_state, tokens, lengths, execution, return_traces, output)
        return self._execute(initial_state, tokens, lengths, execution, return_traces, output)

    def _execute(self, initial_state: torch.Tensor, tokens: torch.Tensor,
                 lengths: torch.Tensor, execution: str, return_traces: bool,
                 output: str) -> Any:
        batch, horizon = tokens.shape
        width, choices = self.internal_states, self.n_primitives + 1
        native = execution == "native"
        if native:
            initial_index = self.encoder_logits[initial_state].argmax(-1)
            h = F.one_hot(initial_index, width).to(self.encoder_logits)
            primitive_indices = self.primitive_logits.argmax(-1)
            native_bank = torch.cat((torch.arange(width, device=h.device)[None, :], primitive_indices), 0)
            readout_indices = self.readout_logits.argmax(-1)
            operator_bank = None
        else:
            h = F.softmax(self.encoder_logits[initial_state], -1)
            primitive_probabilities = F.softmax(self.primitive_logits, -1)
            executed_primitives = (self._hard_or_st(primitive_probabilities, execution)
                                   if execution == "straight_through" else primitive_probabilities)
            operator_bank = torch.cat((self.identity_operator[None, :, :], executed_primitives), 0)
            readout = F.softmax(self.readout_logits, -1)
        m = h.new_zeros((batch, self.memory_dim))
        states, memories = ([h], [m]) if return_traces else ([], [])
        call_probabilities, call_weights, call_logits, call_indices = [], [], [], []
        slot_state_history = []
        active_masks = []
        active_steps = 0
        for step in range(horizon):
            active = step < lengths
            indices = active.nonzero(as_tuple=False).flatten()
            active_steps += indices.numel()
            if return_traces:
                slot_prob, slot_weight, slot_logits, slot_index = [], [], [], []
                slot_states = [h]
            if indices.numel():
                active_h = h.index_select(0, indices)
                active_m = m.index_select(0, indices)
                token = tokens[:, step].index_select(0, indices)
                x = self.token_embedding(token)
                # Preserve S=1 memory semantics: one GRU update from PRE-token
                # h and raw x, after all calls. m is fixed throughout the token.
                memory_feature = (self.state_embedding[active_h.argmax(-1)] if native
                                  else active_h @ self.state_embedding)
            for slot in range(self.n_slots):
                if return_traces:
                    trace_prob = h.new_zeros((batch, choices))
                    trace_weight = h.new_zeros((batch, choices))
                    trace_logits = h.new_zeros((batch, choices))
                    trace_index = torch.full((batch,), -1, device=h.device, dtype=torch.long)
                if indices.numel():
                    state_feature = (memory_feature if slot == 0 else
                        (self.state_embedding[active_h.argmax(-1)] if native
                         else active_h @ self.state_embedding))
                    call_state = state_feature if self.routing == "dynamic" else torch.zeros_like(state_feature)
                    call_memory = active_m if self.routing == "dynamic" else torch.zeros_like(active_m)
                    call_token = x + self.slot_embedding.weight[slot]
                    logits = self.caller(torch.cat((call_state, call_token, call_memory), -1))
                    if native:
                        selected = logits.argmax(-1)
                        next_index = native_bank[selected, active_h.argmax(-1)]
                        next_h = F.one_hot(next_index, width).to(h)
                        if return_traces:
                            probability = F.softmax(logits, -1)
                            weight = F.one_hot(selected, choices).to(h)
                    else:
                        probability = F.softmax(logits, -1)
                        weight = (self._hard_or_st(probability, execution)
                                  if execution == "straight_through" else probability)
                        selected = weight.argmax(-1)
                        # All primitives see the same CURRENT slot-state. The
                        # next slot can condition its shared caller on next_h.
                        candidates = torch.einsum("bi,kij->bkj", active_h, operator_bank)
                        next_h = (weight[:, :, None] * candidates).sum(1)
                    active_h = next_h
                    h = h.index_copy(0, indices, next_h)
                    if return_traces:
                        trace_prob = trace_prob.index_copy(0, indices, probability)
                        trace_weight = trace_weight.index_copy(0, indices, weight)
                        trace_logits = trace_logits.index_copy(0, indices, logits)
                        trace_index = trace_index.index_copy(0, indices, selected)
                if return_traces:
                    slot_prob.append(trace_prob)
                    slot_weight.append(trace_weight)
                    slot_logits.append(trace_logits)
                    slot_index.append(trace_index)
                    slot_states.append(h)
            if indices.numel():
                next_m = self.memory_cell(torch.cat((memory_feature, x), -1), active_m)
                m = m.index_copy(0, indices, next_m)
            if return_traces:
                states.append(h)
                memories.append(m)
                call_probabilities.append(torch.stack(slot_prob, 1))
                call_weights.append(torch.stack(slot_weight, 1))
                call_logits.append(torch.stack(slot_logits, 1))
                call_indices.append(torch.stack(slot_index, 1))
                slot_state_history.append(torch.stack(slot_states, 1))
                active_masks.append(active)
        if native:
            probabilities = F.one_hot(readout_indices[h.argmax(-1)], self.n_states).to(h)
        else:
            probabilities = h @ readout
        # Correct only floating-point mass drift from repeated stochastic rows.
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny)
        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        answer = probabilities if output == "probabilities" else log_probabilities
        if not return_traces:
            return answer
        empty_calls = lambda: h.new_empty((batch, 0, self.n_slots, choices))
        traces = {"states": torch.stack(states, 1), "memories": torch.stack(memories, 1),
                  "call_probabilities": torch.stack(call_probabilities, 1) if horizon else empty_calls(),
                  "call_weights": torch.stack(call_weights, 1) if horizon else empty_calls(),
                  "call_logits": torch.stack(call_logits, 1) if horizon else empty_calls(),
                  "call_indices": torch.stack(call_indices, 1) if horizon else tokens.new_empty((batch, 0, self.n_slots)),
                  "slot_states": torch.stack(slot_state_history, 1) if horizon else h.new_empty((batch, 0, self.n_slots + 1, width)),
                  "active_mask": torch.stack(active_masks, 1) if horizon else tokens.new_empty((batch, 0), dtype=torch.bool),
                  "final_state": h, "final_memory": m, "lengths": lengths,
                  "execution": execution, "routing": self.routing,
                  "active_steps": active_steps,
                  "active_calls": active_steps * self.n_slots,
                  "memory_update_timing": "once at token end from PRE-token h, raw x, old m; m fixed across slots",
                  "straight_through_bindings": "soft E/R; hard-forward/soft-backward calls/operator rows",
                  "readout_reads_memory": False,
                  "resources": self.resource_report(batch_size=batch, sequence_length=horizon,
                       active_steps=active_steps, execution=execution, return_traces=True)}
        return answer, traces

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def resource_report(self, batch_size: int = 1, sequence_length: int = 0, *,
                        active_steps: int | None = None, execution: str = "soft",
                        return_traces: bool = False) -> dict[str, Any]:
        """All resident P/buffers/carry; explicit forward-work proxy, not timing.

        active_steps=sum(lengths) is exact for a particular batch. Omitting it
        reports the all-full-length upper bound. Backend autograd/allocator RAM
        must be measured by the driver; logical payloads here are not peak RSS.
        """
        if execution not in ("soft", "straight_through", "native"):
            raise ValueError("unknown execution mode")
        if batch_size < 0 or sequence_length < 0:
            raise ValueError("nonnegative batch/horizon required")
        steps = batch_size * sequence_length if active_steps is None else active_steps
        if not 0 <= steps <= batch_size * sequence_length:
            raise ValueError("active_steps outside batch/horizon")
        d, k, n = self.internal_states, self.n_primitives, self.n_states
        s, t, m, hidden = self.state_dim, self.token_dim, self.memory_dim, self.caller_hidden
        choices = k + 1
        calls = steps * self.n_slots
        native = execution == "native"
        components = {"state_embedding": 0 if native else calls * d * s,
                      "caller_mlp": calls * ((s + t + m) * hidden + hidden * choices),
                      "gru_linear": steps * 3 * m * (s + t + m),
                      "dense_primitive_application": 0 if native else calls * choices * d * d,
                      "call_weighted_sum": 0 if native else calls * choices * d,
                      "readout": 0 if native else batch_size * d * n}
        scalar_bytes = self.encoder_logits.element_size()
        float_trace_scalars = (batch_size * (sequence_length + 1) * (d + m)
                               + 3 * batch_size * sequence_length * self.n_slots * choices
                               + batch_size * sequence_length * (self.n_slots + 1) * d) if return_traces else 0
        return {"configuration": self.configuration(), "execution": execution,
                "total_parameters": self.parameter_count(), "learnable_parameters": self.parameter_count(True),
                "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad),
                "parameter_payload_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
                "parameter_counts": {name: p.numel() for name, p in self.named_parameters()},
                "registered_buffer_scalars": sum(b.numel() for b in self.buffers()),
                "buffer_payload_bytes": sum(b.numel() * b.element_size() for b in self.buffers()),
                "buffer_counts": {name: b.numel() for name, b in self.named_buffers()},
                "batch_size": batch_size, "sequence_length": sequence_length, "active_steps": steps,
                "slots_per_token": self.n_slots, "active_calls": calls,
                "persistent_example_state_scalars": batch_size * (d + m),
                "persistent_example_state_payload_bytes": batch_size * (d + m) * scalar_bytes,
                "fresh_memory_initialization": "zero each forward; no persistent cross-example memory",
                "operator_execution_workspace_scalars": choices * d if native else choices * d * d,
                "operator_execution_workspace_payload_bytes": choices * d * 8 if native else choices * d * d * scalar_bytes,
                "operator_workspace_dtype": "int64 indices" if native else str(self.encoder_logits.dtype),
                "operator_soft_probability_scalars": 0 if native else k * d * d,
                "operator_st_hard_scalars": k * d * d if execution == "straight_through" else 0,
                "readout_soft_probability_scalars": 0 if native else d * n,
                "trace_logical_payload_bytes": float_trace_scalars * scalar_bytes
                    + ((8 * self.n_slots + 1) * batch_size * sequence_length if return_traces else 0),
                "trace_scope": "token states/memories, slot states,3 slot-call tensors, slot-call int64 indices and token bool mask; final carry counted separately",
                "proxy_forward_macs": sum(components.values()), "proxy_forward_macs_by_component": components,
                "proxy_softmax_elements": ((batch_size * d + k * d * d + d * n + calls * choices)
                                           if not native else (calls * choices if return_traces else 0)),
                "native_argmax_elements": (2 * batch_size * d + k * d * d + d * n
                                           + calls * (choices + 2 * d)) if native else 0,
                "categorical_argmax_elements": ((k * d * d + 2 * calls * choices)
                    if execution == "straight_through" else (calls * choices if not native else 0)),
                "native_transition_lookups": calls if native else 0,
                "native_one_hot_output_scalars": batch_size * (d + n) + calls * d if native else 0,
                "slot_embedding_additions": calls * t,
                "output_probability_normalizations": batch_size * n,
                "proxy_scope": "explicit forward linear/dense MACs; excludes nonlinearities, scatter/gather, backward and backend workspace",
                "backend_autograd_allocator_memory": "not inferred; driver must measure separately",
                "same_mechanism_reference": "ordinary dynamic routing may use this exact module"}

    def checkpoint(self) -> dict[str, Any]:
        """Actual weights/buffers and shape metadata; no task/oracle information."""
        return {"version": self.checkpoint_version, "configuration": self.configuration(),
                "state_dict": {name: tensor.detach().cpu().clone()
                               for name, tensor in self.state_dict().items()},
                "dtype": str(self.encoder_logits.dtype),
                "memory_lifecycle": "zero at forward entry; transient m is not checkpointed",
                "resources": self.resource_report()}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "ContextualPrimitiveBlock":
        if payload.get("version") != cls.checkpoint_version:
            raise ValueError("unsupported contextual primitive checkpoint")
        state = payload["state_dict"]
        tensor = state["encoder_logits"]
        model = cls(**dict(payload["configuration"]), device=tensor.device, dtype=tensor.dtype)
        model.load_state_dict(state, strict=True)
        return model
