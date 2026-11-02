# Frozen parameter/boundary-growth evidence

Built only from actual local JSON/raw and ZIP sizes; no model was imported, unpickled, trained or evaluated. This is a frozen evidence artifact serving the six paper TODOs, not another current Method or plan.

N8/seed0 has nine policies/versions below. Topology V1 failures remain V1 failures. Boundary maintenance and frozen-readout repair are generic rules available to the controls. Transformer failure is an acquisition problem, not evidence of a strong successful baseline being beaten. Shared stage0 sources are reused, not independent replications; do not sum versions as independent training lineages.

|Series|Stage|P|ΔP|Qstable|Current pass|Final neural min|Final deployed min|Old intermediate neural min|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|N8_Constraint_compact_boundary|0|1592|1592|1|1|1|1|NULL|
|N8_Constraint_compact_boundary|1|2228|636|2|2|1|1|0.347656|
|N8_Constraint_compact_boundary|2|2864|636|3|3|1|1|0.308594|
|N8_Constraint_compact_boundary|3|3942|1078|4|4|1|1|0.251953|
|N8_Constraint_compact_boundary|4|4626|684|5|5|1|1|0.242188|
|N8_Constraint_compact_boundary|5|5310|684|6|6|1|1|0.177734|
|N8_Freeze_readout_boundary|0|1536|1536|1|1|1|1|NULL|
|N8_Freeze_readout_boundary|1|2480|944|2|2|1|1|1|
|N8_Freeze_readout_boundary|2|3424|944|3|3|1|1|1|
|N8_Freeze_readout_boundary|3|4496|1072|4|4|1|1|1|
|N8_Freeze_readout_boundary|4|5440|944|5|5|1|1|1|
|N8_Freeze_readout_boundary|5|6384|944|6|6|1|1|1|
|N8_Modular_boundary|0|1352|1352|1|1|1|1|NULL|
|N8_Modular_boundary|1|2704|1352|2|2|1|1|1|
|N8_Modular_boundary|2|4056|1352|3|3|1|1|1|
|N8_Modular_boundary|3|5592|1536|4|4|1|1|1|
|N8_Modular_boundary|4|7128|1536|5|5|1|1|1|
|N8_Modular_boundary|5|8664|1536|6|6|1|1|1|
|N8_Topology_V1|0|1596|1596|1|1|1|1|NULL|
|N8_Topology_V1|1|2232|636|1|1|0.347656|1|0.347656|
|N8_Topology_V1|2|2868|636|1|2|0.308594|1|0.308594|
|N8_Topology_V1|3|3946|1078|1|1|0.251953|1|0.251953|
|N8_Topology_V1|4|4630|684|1|4|0.242188|1|0.242188|
|N8_Topology_V1|5|5314|684|1|3|0.177734|1|0.177734|
|N8_Topology_boundary|0|1596|1596|1|1|1|1|NULL|
|N8_Topology_boundary|1|2232|636|2|2|1|1|0.347656|
|N8_Topology_boundary|2|2868|636|3|3|1|1|0.308594|
|N8_Topology_boundary|3|3946|1078|4|4|1|1|0.251953|
|N8_Topology_boundary|4|4630|684|5|5|1|1|0.242188|
|N8_Topology_boundary|5|5314|684|6|6|1|1|0.177734|
|N8_replay64|0|103752|103752|0|0|0.123047|0.123047|NULL|
|N8_replay64|1|103752|0|0|0|0.103516|0.103516|0.115234|
|N8_replay64|2|103752|0|0|0|0.121094|0.121094|0.103516|
|N8_replay64|3|103752|0|0|0|0.09375|0.09375|0.101562|
|N8_replay64|4|103752|0|0|0|0.103516|0.103516|0.09375|
|N8_replay64|5|103752|0|0|0|0.101562|0.101562|0.103516|
|N8_replay96|0|229352|229352|0|0|0.103516|0.103516|NULL|
|N8_replay96|1|229352|0|0|0|0.115234|0.115234|0.103516|
|N8_replay96|2|229352|0|0|0|0.115234|0.115234|0.113281|
|N8_replay96|3|229352|0|0|0|0.107422|0.107422|0.0898438|
|N8_replay96|4|229352|0|0|0|0.109375|0.109375|0.101562|
|N8_replay96|5|229352|0|0|0|0.101562|0.101562|0.09375|
|N8_transformer64|0|103752|103752|0|0|0.123047|0.123047|NULL|
|N8_transformer64|1|103752|0|0|0|0.115234|0.115234|0.103516|
|N8_transformer64|2|103752|0|0|0|0.105469|0.105469|0.105469|
|N8_transformer64|3|103752|0|0|0|0.0859375|0.0859375|0.105469|
|N8_transformer64|4|103752|0|0|0|0.109375|0.109375|0.0859375|
|N8_transformer64|5|103752|0|0|0|0.0996094|0.0996094|0.0820312|
|N8_transformer96|0|229352|229352|0|0|0.103516|0.103516|NULL|
|N8_transformer96|1|229352|0|0|0|0.111328|0.111328|0.103516|
|N8_transformer96|2|229352|0|0|0|0.109375|0.109375|0.111328|
|N8_transformer96|3|229352|0|0|0|0.0976562|0.0976562|0.0957031|
|N8_transformer96|4|229352|0|0|0|0.0898438|0.0898438|0.0976562|
|N8_transformer96|5|229352|0|0|0|0.103516|0.103516|0.0898438|

Scores use task-end IID≥.95, OOD/L16≥.90 for both neural/deployed; VALID≥.95 controls admission. ‘Old intermediate’ is the observed minimum at0/100/300, not every update. Initial P is fully charged. Ratio is NULL when an old stable task is lost or no positive retained increment occurs.

The C stage0 checkpoint exists after600 updates, but its old reload helper failed before the completed stage report. Its P/metrics/scoring are explicitly derived from real curve/checkpoint events; missing train/eval/reload/phase-wall measurements remain NULL, and successful reload is not backfilled.

All times/storage columns are in the CSV and unabridged source resources in JSON. Phase wall encloses train/guard/refresh/eval/reload components and must not be added to them. Reference bytes overlap model-extra JSON; cache map/certificate bytes are subsets and current views may alias; checkpoint size includes optimizer/RNG/history. No single summed ‘total memory’ or concurrent throughput ranking is fabricated.

Formation precedes Adam for the native learners; its observation, constraint-solving and NN-installation costs are retained where actually measured. Equal600 gradient updates does not remove these extra data passes or formation work. Current/old sample allocation for Replay is reported separately.

N16 rows, if any, are only the explicitly synced completed chunks listed in JSON. Absent stages/scales/seeds are not silently filled. No new N32, complex-program, general Block or real-data success follows from this evidence snapshot.

Files: `parameter-boundary-growth-evidence.json`, `parameter-boundary-growth-evidence.csv`, `parameter-boundary-learning-curves.csv`. Every used source and actual600-step checkpoint has an integrity hash in JSON; new sync markers are verified where present. The builder is `build_parameter_boundary_growth_evidence.py`.
