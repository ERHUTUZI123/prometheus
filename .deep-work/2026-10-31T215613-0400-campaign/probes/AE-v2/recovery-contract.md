# Serialization-only recovery, before evaluation

The first acquisition process saved reference.npz and all six final policy files, then raised TypeError while json.dumps encountered numpy.bool_. Its log is retained. All six 2,000-episode training loops had finished before serialization, as shown by the saved final files and program order. This does not preserve their per-episode records.

Authorized parent decision: evaluate the saved frozen weights; do not retrain or retune. Recover by loading the already inspected definitions from acquire.py, the saved reference table and saved weights. Repeat only the specified numerical/score checks; evaluate final policies on a newly fixed replay stream (seed+10000), 64 samples/context. Initial random-policy evaluations use the original seed; reproduce the ordinary baseline's deterministic candidate stream as static replay. No policy weight update is executed. No favorable seed or checkpoint selection is allowed.

The original learning curves, training material totals, weighted training chemical-work totals and unique reward-query counts were held only in memory and are UNAVAILABLE. Report static logical query budgets, not invented measured totals. Original final evaluation draws are also unavailable; recovered evaluations are new frozen-policy observations, explicitly labelled.

One recovery execution: timeout 120s, CPU 110/115s, same network-denied runner and sanitized environment. Criterion and training configuration remain unchanged. Serialization uses numpy scalar conversion. Stop after recording recovered results, including numerical failures.
