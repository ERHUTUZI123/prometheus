# Exploratory learning diagnostic — witness-local credit for mode formation

Same selected hypothesis, not a new branch/benchmark or confirmatory study. Current mechanism-v1 is explicit but unfinished. The goal is to learn which part of the proposed write/hold learning rule works or fails, to guide its next construction; novelty and superiority are not entry conditions. All chosen conditions below precede this process.

## Question and actual information

Does a legitimate matched-continuation witness provide useful early write/carry updates where endpoint-only credit attenuates, and does the current mean local-carry objective actually establish useful long-hold separation? We do not assume it will. Disentangle prefix-only from prefix+carry, and include an already-persistent-initialization endpoint control. A failure narrows this version/condition, not the hypothesis; a successful small case is not a new principle or general acquisition advantage.

Training data: two retained raw episodes, first cue +/-1 followed by exactly64 zero inputs, with endpoint labels +/-1. Incumbent h=x exactly merges at the first zero; the miner must actually check full suffix equality and locate that merger, paying trace/pair processing. No latent memory label, correct intermediate state or future word is passed to recurrent execution. Seed orientation is designed from the already observed pair's endpoint labels. All histories and labels are shared with controls; labels are reused training data, not free new labels each update.

Provisional state r starts0 and follows r'=tanh(A*r+beta*x), answer v*r. This is the first ungated subfamily of mechanism-v1; clear/reset-gate acquisition is not claimed. Old h and its readout stay unchanged one-way. Four arms: endpoint-only; endpoint+prefix-seed; endpoint+prefix-seed+local-carry, all A.5/beta.01/v1; strong ordinary endpoint control starts A1.2/beta.01/v1. Different initialization is a control for the importance of initial persistence, not a matched-initial causal estimate. Main loss comparison uses the identical A.5 checkpoint.

Fixed settings: margin.4,512 gradient iterations per arm, learning rate.05, global gradient norm cap10 for every arm; project A to[0,2.5], beta[-4,4], v[-3,3]. No adaptation of these numbers after outcomes. True endpoint mean half-squared loss and seed mechanismEq2; carry is the MEAN over actual observed common-suffix one-step **quarter-normalized Eq3**, with previous actual candidate states detached, gradients through that one transition only. Recompute trajectories after each update. Coefficients1 for each included loss. The carry mean may dilute informative steps; if so preserve that failure and diagnose the sampling/update organization rather than silently change it mid-run.

## Observation and resources

Log initial/every16th/final parameters, losses, gradient components/norm, mode states/separation and all forward/derivative/local-pair/mining counts. Finite-difference checks before training compare endpoint/prefix analytic derivatives, and carry derivatives with predecessor states FROZEN; never compare a semigradient to a full-trajectory finite difference. No differentiating future labels or hard pair selection.

Evaluate no-update final models on cue holds32/128/512 and on changed-write sequences. New-write labels are reported under BOTH first-cue and last-write interpretations: training single-cue data do not identify which rule is intended. Their evaluation cannot be claimed learned overwrite or a decisive generalization win. No grid search, seed sweep or hyperparameter tuning. This deterministic small process is exploratory; later confirmation requires a new freeze and independent conditions.

One inspected process, external wall30s/CPU30s via existing OS-isolated runner, network/secrets denied, writes only this new probe dir, one thread, no install/model/permission change. At most4*512 training updates,2 training episodes length65 each replay; finite-difference checks and <=20 declared evaluation episodes per arm are additional charged work; <2MiB raw output. RawJSONL incremental and serialization smoke before launch; final summary/receipt with script+contract SHA256. If it times out or gradients fail, retain partial rows/failure and diagnose without inventing completion. No old experiment or old result is an input.

This is diagnosis within the CONTINUE interval, not a request for a new approval layer, a claimed externally validated new method or a benchmark victory prerequisite. It should inform the next actual update/control rule in this same lineage.
