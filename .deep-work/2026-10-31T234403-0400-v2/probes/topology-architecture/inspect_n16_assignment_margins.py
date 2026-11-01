"""Read exact known checkpoint FP32 tensor bytes WITHOUT unpickling/model code.

Fixed N16 v7 topo stage1-step800 checkpoint only, guarded by complete file hash.
Storage mapping was checked in its pickle opcode listing, not executed. This
reconstructs assignment and routing diagnostics; it trains/runs no network.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zipfile


EXPECTED = "b3a065f1d5ae7ace453eaef4ee055d68a64869d909073fe9bd4fcd5d2272ff4d"


def assign(weights, forbidden=None):
    """Ordinary exact combinatorial Hungarian structure; float64 score math."""
    n = len(weights)
    cost = [[-x for x in row] for row in weights]
    if forbidden:
        cost[forbidden[0]][forbidden[1]] = 1.e9
    u, v, p, way = [0.]*(n+1), [0.]*(n+1), [0]*(n+1), [0]*(n+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        best, used = [math.inf]*(n+1), [False]*(n+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, n+1):
                if not used[j]:
                    candidate = cost[i0-1][j-1]-u[i0]-v[j]
                    if candidate < best[j]:
                        best[j], way[j] = candidate, j0
                    if best[j] < delta:
                        delta, j1 = best[j], j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    best[j] -= delta
            j0 = j1
            if not p[j0]:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    result = [0]*n
    for j in range(1, n+1):
        result[p[j]-1] = j-1
    return result, math.fsum(weights[i][j] for i, j in enumerate(result))


def matrix(values, n, m):
    if len(values) != n*m:
        raise ValueError("fixed storage shape mismatch")
    return [values[i*m:(i+1)*m] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("new output required")
    started = time.perf_counter()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("this static storage map applies only to the known checkpoint SHA")
    with zipfile.ZipFile(args.checkpoint) as archive:
        root = archive.namelist()[0].split("/", 1)[0]
        if archive.read(root+"/byteorder") != b"little":
            raise ValueError("expected exact saved little-endian checkpoint")
        def raw(index):
            blob = archive.read(root+f"/data/{index}")
            return list(struct.unpack("<"+"f"*(len(blob)//4), blob))
        encoder, decoder = matrix(raw(0),16,16), matrix(raw(1),16,16)
        operations = raw(2)
        if len(operations) != 6*16*16:
            raise ValueError("fixed six-bank N16 checkpoint required")
        tokens, tasks = matrix(raw(4),7,24), matrix(raw(5),3,24)
        # The fitted shared head was saved with shape(14,24), stride(1,14),
        # because it originated as solution.T. Respect its actual storage.
        call_storage = raw(6)
        if len(call_storage) != 14*24:
            raise ValueError("fixed strided call storage shape mismatch")
        call = [[call_storage[i+14*j] for j in range(24)] for i in range(14)]
        bias = raw(7)
    snapshots = [json.loads(line) for line in args.records.read_text().splitlines()]
    curve = next(r for r in snapshots if r.get("kind")=="curve" and r.get("model")=="topo"
                 and r.get("stage")==1 and r.get("step")==800)
    saved = next(g for g in curve["learned_structure"]["graphs"] if int(g["task"])==1)
    permutations, op_reports = [list(range(16))], []
    for k in range(6):
        if time.perf_counter()-started > 10:
            raise TimeoutError("10s static numerical diagnostic cap")
        w = matrix(operations[k*256:(k+1)*256],16,16)
        chosen, score = assign(w)
        alternatives = [assign(w, (i, j)) for i, j in enumerate(chosen)]
        runner_up, second_score = max(alternatives, key=lambda x:x[1])
        permutations.append(chosen)
        op_reports.append(dict(expert=k+1, permutation=chosen, optimal_score=score,
            highest_different_assignment_score=second_score, global_assignment_margin=score-second_score,
            runner_up_permutation=runner_up, changed_rows=[i for i in range(16) if chosen[i]!=runner_up[i]],
            assignments_computed=17))
    route_reports, same_routes, same_transitions = [], True, True
    for index, a in enumerate((1,2,3,4,5)):
        hidden = [math.tanh(tokens[a][j]+tasks[1][j]) for j in range(24)]
        logits = [math.fsum(x*y for x,y in zip(row, hidden))+offset for row,offset in zip(call,bias)]
        slots, gaps = [], []
        for s in range(2):
            values=logits[s*7:(s+1)*7]
            ranks=sorted(range(7),key=lambda k:values[k],reverse=True)
            slots.append(ranks[0]);gaps.append(values[ranks[0]]-values[ranks[1]])
        native = list(range(16))
        for k in slots:
            native = [permutations[k][q] for q in native]
        route_match=slots==saved["routes"][index]
        target=saved["learned_transitions"].get(str(a),saved["learned_transitions"].get(a))
        trans_match=native==target
        same_routes &= route_match;same_transitions &= trans_match
        route_reports.append(dict(token=a, slots=slots, raw_call_margin=gaps,
            margin_after_temperature_division=[g/curve["actual_temperature"] for g in gaps],
            saved_routes=saved["routes"][index], routes_match_saved=route_match,
            reconstructed_native_transition=native, transition_matches_saved=trans_match))
    result=dict(schema="fixed-checkpoint-static-assignment-diagnostic-v1",status="RECONSTRUCTION",
        checkpoint=str(args.checkpoint),checkpoint_sha256=EXPECTED, checkpoint_unpickled=False,
        torch_or_model_code_imported=False,new_training_or_task_execution=False,
        records=str(args.records),records_sha256=hashlib.sha256(args.records.read_bytes()).hexdigest(),
        model="topo",stage=1,step=800,operation_assignments=op_reports,routes=route_reports,
        native_routes_match_saved=same_routes,native_transitions_match_saved=same_transitions,
        call_weight_saved_shape=[14,24],call_weight_saved_stride=[1,14],
        parameter_count_from_exact_shapes=2642, wall_seconds=time.perf_counter()-started,
        scope=["Operation weights are exact saved float32 values read as float64; Hungarian objective uses those exact numeric values.",
               "Route logits are reconstructed with Python double arithmetic; GPU float32 tanh/add/matmul rounding is not reproduced.",
               "Agreement with saved native routes/transitions checks the reconstructed existing execution; it supplies no task answers.",
               "A small assignment gap indicates sensitivity, not proof it caused ON/OFF divergence or the learning failure."])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(out=str(args.out),native_routes_match=same_routes,native_transitions_match=same_transitions,
                         margins=[r["global_assignment_margin"] for r in op_reports],seconds=result["wall_seconds"])))


if __name__ == "__main__":main()
