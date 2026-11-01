"""Restricted, deterministic construction probe. No network, subprocess or eval.

This supplies a two-field fold and repeat-decode skeleton; only boundary formula,
decode count expression and task field programs are synthesized. It must not be
described as synthesizing an unconstrained representation or neural learning.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()

    @property
    def size(self):
        return 1 + sum(a.size for a in self.args)

    def show(self):
        if not self.args:
            return self.op
        return f"({self.op} {' '.join(a.show() for a in self.args)})"

    def run(self, vals, cost):
        cost['boolean_ast_nodes'] += 1
        if self.op == 'true': return True
        if self.op == 'false': return False
        if self.op in ('a0', 'a1', 'a2', 'a3'): return vals[int(self.op[1:])]
        if self.op == 'not': return not self.args[0].run(vals, cost)
        # No short circuit: the cost convention is explicit and deterministic.
        left = self.args[0].run(vals, cost)
        right = self.args[1].run(vals, cost)
        return left and right if self.op == 'and' else left or right


def truth_key(expr, cost):
    cost['truth_rows_evaluated'] += 16
    return tuple(expr.run(v, cost) for v in product([False, True], repeat=4))


def expressions(max_size=5, cost=None):
    """Exact enumeration up to size bound, modulo abstract Boolean equivalence."""
    if cost is None: cost = Counter()
    best = {}
    levels = {}
    for size in range(1, max_size + 1):
        candidates = []
        if size == 1:
            candidates = [Expr(x) for x in ['false', 'true', 'a0', 'a1', 'a2', 'a3']]
        else:
            candidates += [Expr('not', (x,)) for x in levels.get(size - 1, [])]
            for left_size in range(1, size - 1):
                right_size = size - 1 - left_size
                for a in levels.get(left_size, []):
                    for b in levels.get(right_size, []):
                        if a.show() <= b.show():
                            candidates += [Expr(op, (a, b)) for op in ['and', 'or']]
        levels[size] = []
        for expr in sorted(candidates, key=lambda x: x.show()):
            cost['formula_candidates_generated'] += 1
            key = truth_key(expr, cost)
            if key not in best:
                best[key] = expr
                levels[size].append(expr)
    return sorted(best.values(), key=lambda x: (x.size, x.show()))


def parse(s, predicate, cost):
    out = []
    for i, c in enumerate(s):
        cost['source_symbol_reads'] += 1
        if not out:
            split = True
        else:
            vals = (c == s[i-1], i % 2 == 0, out[-1][1] % 2 == 0, c == out[-1][0])
            cost['atom_evaluations'] += 4
            split = predicate.run(vals, cost)
        if split:
            out.append([c, 1])
            cost['record_scalar_writes'] += 2
        else:
            out[-1][1] += 1
            cost['record_scalar_reads'] += 1
            cost['record_scalar_writes'] += 1
    return tuple((c, n) for c, n in out)


WRITERS = ['n', 'n+1', 'n-1', '2*n', 'n//2', '(n+1)//2', '1']
WRITER_SIZE = dict(zip(WRITERS, [1, 3, 3, 3, 3, 5, 1]))


def decode(v, writer, cost):
    out = []
    for c, n in v:
        cost['record_scalar_reads'] += 2
        cost['integer_ast_nodes'] += WRITER_SIZE[writer]
        k = {'n': n, 'n+1': n+1, 'n-1': n-1, '2*n': 2*n,
             'n//2': n//2, '(n+1)//2': (n+1)//2, '1': 1}[writer]
        if k <= 0 or k > 100:
            raise ValueError('outside bounded positive count domain')
        for _ in range(k):
            out.append(c)
            cost['raw_output_writes'] += 1
    return ''.join(out)


def map_counts(v, coeff, cost):
    a, b = coeff
    out = []
    for c, n in v:
        cost['record_scalar_reads'] += 2
        cost['integer_ast_nodes'] += 5
        m = a*n+b
        if m <= 0 or m > 100:
            raise ValueError('outside bounded positive count domain')
        out.append((c, m))
        cost['record_scalar_writes'] += 2
    return tuple(out)


def task_run(s, predicate, writer, coeff, cost):
    return decode(map_counts(parse(s, predicate, cost), coeff, cost), writer, cost)


def strings(alphabet, max_len):
    for n in range(max_len + 1):
        for xs in product(alphabet, repeat=n):
            yield ''.join(xs)


def oracle_increment(s, cost):
    """Independent raw algorithm; no record representation or parser calls."""
    out = []
    for i, c in enumerate(s):
        cost['source_symbol_reads'] += 1
        out.append(c); cost['raw_output_writes'] += 1
        if i + 1 == len(s):
            last = True
        else:
            cost['source_symbol_reads'] += 1
            cost['scalar_comparisons'] += 1
            last = c != s[i+1]
        if last:
            out.append(c); cost['raw_output_writes'] += 1
    return ''.join(out)


def oracle_double(s, cost):
    out = []
    for c in s:
        cost['source_symbol_reads'] += 1
        for _ in range(2): out.append(c); cost['raw_output_writes'] += 1
    return ''.join(out)


def first_failure(predicate, writer, domain):
    for s in domain:
        try:
            v = parse(s, predicate, Counter())
            out = decode(v, writer, Counter())
            if out != s: return {'source': s, 'view': v, 'output': out}
        except ValueError as e:
            return {'source': s, 'error': str(e)}
    return None


def main():
    start = time.perf_counter()
    formula_generation_cost = Counter()
    es = expressions(PROTOCOL['search']['max_boolean_ast_nodes'], formula_generation_cost)
    domain = list(strings('abc', 3))
    construction_cost = Counter()
    accounting = Counter()
    winners = []
    for pred in es:
        for writer in WRITERS:
            accounting['representation_pairs_tried'] += 1
            ok = True
            for s in domain:
                accounting['roundtrip_cases_executed'] += 1
                try:
                    if decode(parse(s, pred, construction_cost), writer, construction_cost) != s:
                        ok = False; break
                except ValueError:
                    ok = False; break
            if not ok: continue
            accounting['representation_pairs_passing_raw_roundtrip'] += 1
            task_coeffs = {}
            for name, examples in PROTOCOL['training'].items():
                choices = []
                for coeff in product([0,1,2], [-1,0,1]):
                    accounting['task_programs_tried'] += 1
                    good = True
                    for s, expected in examples:
                        accounting['training_cases_executed'] += 1
                        try:
                            v = parse(s, pred, construction_cost)
                            edited = map_counts(v, coeff, construction_cost)
                            out = decode(edited, writer, construction_cost)
                            # Requested edits must retain their representation meaning.
                            reparsed = parse(out, pred, construction_cost)
                            if out != expected or reparsed != edited:
                                good = False; break
                        except ValueError:
                            good = False; break
                    if good: choices.append(coeff)
                if not choices: ok = False; break
                task_coeffs[name] = choices[0]
            if ok:
                score = pred.size + WRITER_SIZE[writer] + 2*5
                winners.append((score, pred, writer, task_coeffs))
    winners.sort(key=lambda x: (x[0], x[1].show(), x[2]))
    if not winners: raise AssertionError('No executable view found within stated scaffold')
    score, pred, writer, coeffs = winners[0]

    # No held-out task example or law test above influences this selection.
    heldout = list(strings('abc', 6)) + ['xyz', 'wwwwwxyyyyyyyy', 'mmmmmmmmnnppp', '']
    check_cost = Counter(); learned_execution_cost = Counter(); fused_cost = Counter()
    semantic_checks = Counter(); failures = []
    for s in heldout:
        v = parse(s, pred, check_cost)
        if decode(v, writer, check_cost) != s: failures.append(['GetPut', s])
        semantic_checks['GetPut'] += 1
        for name, oracle in [('increment', oracle_increment), ('double', oracle_double)]:
            actual = task_run(s, pred, writer, coeffs[name], learned_execution_cost)
            expect = oracle(s, fused_cost)
            if actual != expect: failures.append([name, s, actual, expect])
            semantic_checks['heldout_raw_task_outputs'] += 1
        for a,b in [(1,1),(2,0)]:
            edited = map_counts(v, (a,b), check_cost)
            out = decode(edited, writer, check_cost)
            if parse(out, pred, check_cost) != edited: failures.append(['PutGet',s,edited])
            semantic_checks['PutGet'] += 1
        e1=map_counts(v,(1,1),check_cost); e2=map_counts(v,(2,0),check_cost)
        # Decode ignores old source in this lossless subcase: implement both calls.
        s1=decode(e1,writer,check_cost)
        left=decode(e2,writer,check_cost); right=decode(e2,writer,check_cost)
        if left!=right: failures.append(['PutPut',s,s1])
        semantic_checks['PutPut_lossless_special_case'] += 1
    if failures: raise AssertionError(failures[:10])

    identity_merging=(('a',1),('a',1))
    merging_raw=decode(identity_merging,writer,Counter())
    merging_reparse=parse(merging_raw,pred,Counter())
    identity_edit_rejected=merging_reparse!=identity_merging
    assert identity_edit_rejected
    false_parser=Expr('false')
    allsplit=Expr('true')
    literal_get_only_failure=task_run('abbc',allsplit,'n',(1,1),Counter())
    bag_source='aba'; bag_view=(('a',2),('b',1))
    bag_output=decode(bag_view,'n',Counter())
    assert bag_output!=bag_source

    # Source reconstruction check, independent of program synthesis.
    magic=[[4,9,2],[3,5,7],[8,1,6]]
    lines=[tuple(row) for row in magic]
    lines += [tuple(magic[r][c] for r in range(3)) for c in range(3)]
    lines += [tuple(magic[i][i] for i in range(3)),tuple(magic[i][2-i] for i in range(3))]
    from itertools import combinations
    triples={tuple(sorted(x)) for x in combinations(range(1,10),3) if sum(x)==15}
    assert {tuple(sorted(x)) for x in lines}==triples

    # Correct the explicitly flagged schematic typo, without changing frozen v0.
    frozen_wrong='aaaabbbbbb'
    corrected=oracle_double('aabccc',Counter())
    assert corrected=='aaaabbcccccc' and corrected!=frozen_wrong
    out={
      'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
      'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'runtime_seconds':time.perf_counter()-start,
      'boolean_semantic_classes_enumerated':len(es),
      'formula_generation_execution_units':dict(formula_generation_cost),
      'selection_accounting':dict(accounting),
      'construction_execution_units':dict(construction_cost),
      'selected':{'boundary':pred.show(),'atom_dictionary':PROTOCOL['search']['boolean_atoms'],
                  'writer':writer,'task_field_programs':coeffs,'ast_score_given_scaffold':score},
      'all_winning_candidates':[{'boundary':x[1].show(),'writer':x[2],'ast_score':x[0]} for x in winners],
      'post_selection_checks':dict(semantic_checks),'failures':failures,
      'heldout_case_count':len(heldout),
      'semantic_check_execution_units':dict(check_cost),
      'learned_view_execution_units':dict(learned_execution_cost),
      'fused_raw_execution_units':dict(fused_cost),
      'adversarial_examples':{
        'whole_input_as_one_record':first_failure(false_parser,'n',domain),
        'wrong_decoder':first_failure(pred,'2*n',domain),
        'Get_only_identity_parser_increment':{'input':'abbc','output':literal_get_only_failure,'expected':'aabbbcc'},
        'bag_counts_lose_order':{'input':bag_source,'view':bag_view,'output':bag_output},
        'identity_merging_edit':{'requested_view':identity_merging,'raw':merging_raw,'reparsed':merging_reparse,'rejected':identity_edit_rejected}},
      'source_reconstruction':{'simon_magic_square_triples':len(triples),'verified_equal':True},
      'frozen_v0_typo':{'input':'aabccc','task':'double every count','frozen_output':frozen_wrong,'correct_output':corrected},
      'limits':['two-field fold/repeat-decode scaffold is supplied','full AST schema discovery unimplemented','neural proposal learning unimplemented','no task-discovery baseline was run','per-operation total execution is worse than fused direct raw algorithms in this accounting','PutPut special case is structurally guaranteed because decoder ignores the old source; not independent evidence for general lenses']}
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
