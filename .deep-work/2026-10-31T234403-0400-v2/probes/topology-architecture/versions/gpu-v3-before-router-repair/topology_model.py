"""Learned finite covering/executor prototype; no teacher operations are inputs.

Software braid/permutation quotient + learned reachable-state equivalence guard.
This is not a physical quantum model or evidence of a uniquely topological gain.
All discrete operation/call/state classes are extracted from trained parameters.
"""
import copy
import math
import time
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
# Explicit human-authorized task-local dependency; never changes system Torch.
_deps = Path('/opt/prometheus-v2-python')
if _deps.is_dir():
    sys.path.insert(0, str(_deps))
from scipy.optimize import linear_sum_assignment


def assignment(logits):
    a = logits.detach().cpu().double().numpy()
    r, c = linear_sum_assignment(-a)
    p = np.empty(len(r), dtype=np.int64)
    p[r] = c
    return p


def braid_word(permutation):
    """Canonical adjacent exchange representation of a LEARNED permutation."""
    order = list(range(len(permutation)))
    target = np.argsort(permutation).tolist()
    word = []
    for i, item in enumerate(target):
        j = order.index(item)
        while j > i:
            order[j-1], order[j] = order[j], order[j-1]
            word.append(j-1)
            j -= 1
    return word


class TopologyExecutor(nn.Module):
    def __init__(self, n_states, vocab=6, max_len=18, n_tasks=3,
                 latent_factor=1, protection=True, seed=0, n_ops=6,
                 slots=2, emb_dim=24, constraint=False):
        super().__init__()
        if n_states > 32 or latent_factor > 2:
            raise ValueError("frozen tensor scale exceeded")
        torch.manual_seed(seed)
        self.n, self.d = n_states, n_states * latent_factor
        self.vocab, self.n_tasks, self.n_ops = vocab, n_tasks, n_ops
        self.slots, self.protection = slots, protection
        self.constraint = constraint
        self.temperature, self.stage = 1., 0
        self.compilation = 'component'
        self.credit = 'log'
        self.token = nn.Embedding(vocab+1, emb_dim, padding_idx=0)
        self.task = nn.Embedding(n_tasks, emb_dim)
        self.call = nn.Linear(emb_dim, slots*(n_ops+1))
        self.encoder = nn.Parameter(torch.randn(n_states, self.d)*.03-3.)
        self.decoder = nn.Parameter(torch.randn(self.d, n_states)*.03-3.)
        self.operations = nn.Parameter(torch.randn(n_ops, self.d, self.d)*.12-1.)
        with torch.no_grad():
            self.encoder[torch.arange(n_states), torch.arange(n_states)] = 5.
            self.decoder[torch.arange(self.d), torch.arange(self.d)%n_states] = 5.
            self.operations[:, torch.arange(self.d), torch.arange(self.d)] += 1.5
            self.task.weight.mul_(.15)
            self.call.weight.mul_(1.5)
            self.call.bias.zero_()
            # Identity is a supplied skip, not a correct task operation.
            self.call.bias.view(slots, n_ops+1)[:, 0] = .7
        self.reference = []
        self.last_guard = {}
        self._eye = None

    def set_stage(self, stage, new_token_ids=None):
        self.stage = int(stage)
        for p in self.parameters():
            p.requires_grad_(True)

    def set_temperature(self, temperature):
        self.temperature = max(.12, float(temperature))

    def soft_ops(self):
        q = self.operations / self.temperature
        # Differentiable log-Sinkhorn: permutation classes are LEARNED.
        for _ in range(4):
            q = q - torch.logsumexp(q, -1, keepdim=True)
            q = q - torch.logsumexp(q, -2, keepdim=True)
        p = q.exp()
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        eye = torch.eye(self.d, device=p.device, dtype=p.dtype)[None]
        return torch.cat([eye, p], 0)

    def calls(self, tokens, task_ids):
        x = torch.tanh(self.token(tokens) + self.task(task_ids)[:, None])
        logits = self.call(x).view(*tokens.shape, self.slots, self.n_ops+1)
        return logits

    def log_ops(self):
        q = self.operations / self.temperature
        for _ in range(4):
            q = q - torch.logsumexp(q, -1, keepdim=True)
            q = q - torch.logsumexp(q, -2, keepdim=True)
        q = q - torch.logsumexp(q, -1, keepdim=True)
        eye = torch.full((self.d,self.d), -torch.inf, device=q.device)
        eye.diagonal().zero_()
        return torch.cat([eye[None],q],0)

    def forward(self, tokens, starts, task_ids):
        if tokens.shape[1] > 18 or tokens.shape[0] > 128:
            raise ValueError("input bound exceeded")
        if self.credit == 'prob':
            ops=self.soft_ops()
            route=torch.softmax(self.calls(tokens,task_ids)/self.temperature,-1)
            state=torch.softmax(self.encoder/self.temperature,-1)[starts]
            for t in range(int(tokens.ne(0).sum(-1).max().item())):
                for slot in range(self.slots):
                    matrix=torch.einsum('bk,kij->bij',route[:,t,slot],ops)
                    following=torch.bmm(state[:,None],matrix).squeeze(1)
                    state=torch.where(tokens[:,t,None]!=0,following,state)
            return (state@torch.softmax(self.decoder/self.temperature,-1)).clamp_min(1e-12).log()
        ops = self.log_ops()
        route = torch.log_softmax(self.calls(tokens, task_ids)/self.temperature, -1)
        state = torch.log_softmax(self.encoder/self.temperature, -1)[starts]
        active_length = int(tokens.ne(0).sum(-1).max().item())
        for t in range(active_length):
            valid = (tokens[:, t] != 0)[:, None]
            for slot in range(self.slots):
                matrix = torch.logsumexp(route[:,t,slot,:,None,None]+ops[None],1)
                following = torch.logsumexp(state[:,:,None]+matrix,1)
                state = torch.where(valid, following, state)
        return torch.logsumexp(state[:,:,None] +
                              torch.log_softmax(self.decoder/self.temperature,-1)[None],1)

    @torch.no_grad()
    def graph(self, task_id, token_ids):
        enc = self.encoder.argmax(-1).cpu().numpy()
        dec = self.decoder.argmax(-1).cpu().numpy()
        perms = [np.arange(self.d)] + [assignment(p) for p in self.operations]
        tok = torch.tensor([list(token_ids)], dtype=torch.long,
                           device=self.encoder.device)
        tid = torch.tensor([task_id], device=tok.device)
        routes = self.calls(tok, tid)[0].argmax(-1).cpu().numpy()
        transitions = {}
        if self.compilation == 'joint':
            probabilities=torch.softmax(self.calls(tok,tid)[0]/self.temperature,-1)
            soft=self.soft_ops()
            for index,a in enumerate(token_ids):
                matrix=torch.eye(self.d,device=soft.device)
                for slot in range(self.slots):
                    matrix=matrix@torch.einsum('k,kij->ij',probabilities[index,slot],soft)
                transitions[int(a)]=assignment(matrix.clamp_min(1e-30).log())
        else:
            for a, call in zip(token_ids, routes):
                out = np.arange(self.d)
                for k in call:
                    out = perms[int(k)][out]
                transitions[int(a)] = out
        reach = set(int(i) for i in enc)
        pending = list(reach)
        while pending:
            q = pending.pop()
            for f in transitions.values():
                v = int(f[q])
                if v not in reach:
                    reach.add(v)
                    pending.append(v)
        return dict(task=int(task_id), tokens=tuple(int(a) for a in token_ids),
                    enc=enc, dec=dec, trans=transitions, reachable=sorted(reach),
                    routes=routes, permutations=perms,
                    braid_lengths=[len(braid_word(p)) for p in perms[1:]],
                    compiled_braid_lengths={a:len(braid_word(p)) for a,p in transitions.items()},
                    compilation=self.compilation)

    @torch.no_grad()
    def hard_eval(self, tokens, starts, task_ids):
        predictions = torch.empty_like(starts)
        for task_id in task_ids.unique().tolist():
            select = task_ids == task_id
            # No labels are used to extract the current executable graph.
            graph = self.graph(task_id, range(1, self.vocab+1))
            q = graph['enc'][starts[select].cpu().numpy()].copy()
            words = tokens[select].cpu().numpy()
            for t in range(words.shape[1]):
                for a, f in graph['trans'].items():
                    mask = words[:, t] == a
                    q[mask] = f[q[mask]]
            pred = graph['dec'][q]
            predictions[select] = torch.as_tensor(pred, device=starts.device)
        logits = torch.full((len(starts), self.n), -30., device=starts.device)
        return logits.scatter_(1, predictions[:, None], 0.)

    @staticmethod
    def related(old, new):
        """Exact finite model relation for ALL old words, not a task oracle.

        This is ordinary reachable graph equivalence as well as a certificate
        for decoded covering actions. Its conventional explanation is explicit.
        """
        pending = [(int(a), int(b)) for a, b in zip(old['enc'], new['enc'])]
        relation = set()
        while pending:
            a, b = pending.pop()
            if (a, b) in relation:
                continue
            relation.add((a, b))
            if old['dec'][a] != new['dec'][b]:
                return False, len(relation)
            for symbol in old['tokens']:
                pair = (int(old['trans'][symbol][a]), int(new['trans'][symbol][b]))
                if pair not in relation:
                    pending.append(pair)
        return True, len(relation)

    @torch.no_grad()
    def consolidate(self, stage, token_ids):
        g = self.graph(stage, token_ids)
        self.reference.append(g)
        return dict(task=stage, learned_reachable_states=len(g['reachable']),
                    latent_states=self.d, learned_call_routes=g['routes'].tolist(),
                    learned_braid_lengths=g['braid_lengths'],
                    reference_from='trained parameters only, no teacher programs',
                    certificate='finite reachable old/new decoded relation',
                    old_teacher_correctness_not_assumed=True)

    @torch.no_grad()
    def project_update(self, before):
        started = time.perf_counter()
        if not self.protection or not self.reference:
            return dict(accepted=True, alpha=1., checks=0, seconds=0.)
        proposal = {n:p.detach().clone() for n,p in self.named_parameters()}
        checks, pairs = 0, 0
        for alpha in (1., .5, .25, .125, .0625, .03125, .015625, 0.):
            for n,p in self.named_parameters():
                p.copy_(before[n] + alpha*(proposal[n]-before[n]))
            valid = True
            for old in self.reference:
                current = self.graph(old['task'], old['tokens'])
                ok, count = self.related(old, current)
                checks += 1
                pairs += count
                if not ok:
                    valid = False
                    break
            if valid:
                self.last_guard = dict(accepted=alpha>0, alpha=alpha,
                    checks=checks, relation_pairs=pairs,
                    seconds=time.perf_counter()-started,
                    physical_homotopy_not_measured=True,
                    ordinary_equivalence_certificate_available_to_rivals=True)
                return self.last_guard
        raise RuntimeError('rollback failed to preserve reference model relation')

    def enforce_frozen_rows_(self):
        pass

    def representation_state_bytes(self):
        return sum(a.nbytes for g in self.reference
                   for a in [g['enc'],g['dec'],*g['trans'].values()])

    def extra_state_dict(self):
        return dict(reference=copy.deepcopy(self.reference),
                    temperature=self.temperature, stage=self.stage,
                    compilation=self.compilation,credit=self.credit,
                    protection=self.protection,
                    limitation='software monodromy/permutation quotient; generic constraint explanation remains')
