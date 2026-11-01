# Raw-stream local construction of lexical computation — v1

**AGENT CONSTRUCTION / HYPOTHESIS, not an acquisition result.** Same after014 continuation. Root selected raw-stream local operational organization as MAIN focus and proposed removing the global encoder. This worker constructs the explicit witness protocol below. Root also proposed the optional nominal-atom interface; it is incorporated with its limitations. V0 is preserved unchanged. No experiment, generated implementation or training has been run. Provider effective mode remains unknown; file isolation is instruction_only.

**Judgment:** a single pair-local graph emitter can, by actual bounded-size rewrites, consume only a raw token chain, construct syntax and lexical-environment graphs, duplicate a captured higher-order function and evaluate it correctly under shadowing. The following witness is fully specified on well-formed simply typed terms with atom outputs. Its rules are manually constructed here; learning them from terminal feedback remains OPEN. This establishes an inspectable mechanism target, not that neural learning has acquired it or that the evaluator is new.

## 1. Input, resources and one emitter

The initial graph is **only** a fixed unary chain of raw tokens, an End node and Start. It contains no parse tree, stack, lambda/application graph, variable-occurrence aliases, scopes, environment, code package, parser, evaluator or learned global E_phi. A fixed tokenizer preserves token identity and order. Source notation is prefix:

`e ::= C atom | V name | L name e | A e e`.

These are raw literal tokens C,V,L,A followed, when applicable, by an opaque atom. Their arities and meanings belong to the witness's *learned target behavior*, not a host parser. There are no parentheses or host recursion: the local protocol must discover where each subexpression ends. Dataset syntax is an information scaffold, as it is for a Neural GPU receiving a raw program string. Names and ground literals may be fresh opaque atoms; source tags and structural marker codes remain ordinary finite values.

Each graph agent has one principal port, at most two auxiliary ports, a principal polarity + or -, a b-bit structural code and at most one opaque atom reference. Two opposite-polarity principal ports form an active pair. One and the same neural kernel emits a replacement of at most four agents plus a perfect matching of all boundary/new ports. Principal polarity fixes the input order; node addresses and event times are not visible to the kernel. Closed wire components are retained and metered as in v0.

The host supplies pointer wiring, fresh-node allocation, finite arithmetic, bounded local emission, an active-pair queue, external budgets and a terminal atom observer. It does **not** implement any rule below. The witness uses exact copied structural codes and atom references; an actual neural categorical emitter would have to acquire the appropriate emissions. All control/constructor names below are explanatory labels for code values, not seed operations installed during learning.

**Root-seeded nominal interface.** The kernel may see whether its two carried atoms are equal, but not their numerical IDs, string order, spelling or embeddings. Emissions choose an atom reference from either input or request a bounded fresh atom. The witness never needs a fresh lexical name: it creates fresh *scope nodes*. Copy/equality of raw identities is a supplied resource, distinct from binding. This makes global name-renaming covariance structural, up to paired fresh renaming if fresh atoms are ever used. It does not imply alpha-conversion or correct scope resolution. All atom/reference storage and comparisons count. The allocator's internal IDs are unobservable to K and are not a deterministic equivariant fresh-name oracle.

## 2. Port notation and the witness rule law

`p=X[u,v]` creates one X agent, attaches its principal port to wire p and its auxiliary ports to wires u,v. Each new internal wire occurs exactly twice; each old external boundary wire occurs once in a replacement. A displayed bare equality joins two boundary wires. Thus equations below specify actual reconnections, not host-language function calls. `X[]` is nullary. Subscripts x or c denote the agent's carried atom reference.

The table gives replacements for an active principal pair. All data/continuation constructors are +; all controllers, D (duplication) and Erase are -. The r-ary data constructor copy rules below cover r=0,1,2. No witness replacement exceeds four agents or arity two.

### 2.1 Raw stream to generated syntax

Raw `Tok_t[next]` has arity one and + principal. End has arity zero. Start has one output auxiliary port. P is a parser controller with one continuation port; PReturn carries a generated expression and the unconsumed raw tail. The parser's stack is the graph of Await nodes it actually allocates.

| Active pair and old boundary | Replacement equations | New agents |
|---|---|---:|
| `Start[out]` / `Tok_t[next]` | `h=Tok_t[next]; h=P[q]; q=PFinish[out]` | 3 |
| `P[k]` / `Tok_A[next]` | `next=P[q]; q=AwaitFun[k]` | 2 |
| `P[k]` / `Tok_L[next]` | `next=ReadL[k]` | 1 |
| `P[k]` / `Tok_V[next]` | `next=ReadV[k]` | 1 |
| `P[k]` / `Tok_C[next]` | `next=ReadC[k]` | 1 |
| `ReadL[k]` / `Tok_x[next]`, x an atom token | `next=P[q]; q=AwaitLam_x[k]` | 2 |
| `ReadV[k]` / `Tok_x[next]` | `k=PReturn[v,next]; v=Var_x[]` | 2 |
| `ReadC[k]` / `Tok_c[next]` | `k=PReturn[v,next]; v=Atom_c[]` | 2 |
| `PReturn[body,rest]` / `AwaitLam_x[k]` | `k=PReturn[l,rest]; l=Lam_x[body]` | 2 |
| `PReturn[f,rest]` / `AwaitFun[k]` | `rest=P[q]; q=AwaitArg[f,k]` | 2 |
| `PReturn[a,rest]` / `AwaitArg[f,k]` | `k=PReturn[v,rest]; v=App[f,a]` | 2 |
| `PReturn[e,rest]` / `PFinish[out]` | `rest=PCheckEnd[e,out]` | 1 |
| `PCheckEnd[e,out]` / `End[]` | `e=Eval[env,k]; env=Empty[]; k=Done[out]` | 3 |

The raw token atom is distinguished from the four reserved source-tag literals by its input-token descriptor, not by inspecting the atom's spelling. This is tokenization, not identification of which occurrences bind. ReadL passes the atom into an Await frame and later into the Lam constructor by copying its reference. No occurrence is linked to that binder while parsing. The meaning of the tag L and the entire stack discipline are rules of the witness kernel.

For n raw tokens encoding m syntax nodes, this parser uses exactly `n+m+2` interaction events, including Start and the End check: each token is consumed once by P/Read, each syntax node is returned once to its enclosing frame, plus Start and End. The first token is recreated by Start, so Start is the extra token interaction counted separately. Every replacement allocates at most three agents. Work/allocation and peak graph size are O(n), with at most O(n) Await frames. These are derived counts for the witness, not measurements of neural acquisition. The input grammar is not executed by a host parser.

### 2.2 Generated lexical environments and call-by-value execution

An environment is a generated Empty/Frame tree. `Frame_x[value,parent]` introduces the nearest binding of x. `Closure_x[body,captured_env]` is passive data, not an active computation. Ret returns a value to a generated continuation; continuation payloads are represented by Pair because no node may have three auxiliary ports.

| Active pair | Replacement equations | New agents |
|---|---|---:|
| `Eval[e,k]` / `App[f,a]` | `f=Eval[e1,q]; e=D[e1,e2]; q=KArg[p,k]; p=Pair[a,e2]` | 4 |
| `Eval[e,k]` / `Lam_x[body]` | `k=Ret[v]; v=Closure_x[body,e]` | 2 |
| `Eval[e,k]` / `Atom_c[]` | `k=Ret[v]; v=Atom_c[]; e=Erase[]` | 3 |
| `Eval[e,k]` / `Var_x[]` | `e=Lookup_x[k]` | 1 |
| `Ret[v]` / `KArg[p,k]` | `p=UnpackArg[v,k]` | 1 |
| `UnpackArg[v,k]` / `Pair[a,e]` | `a=Eval[e,q]; q=KApply[v,k]` | 2 |
| `Ret[arg]` / `KApply[fun,k]` | `fun=Apply[arg,k]` | 1 |
| `Apply[arg,k]` / `Closure_x[body,e]` | `body=Eval[eNew,k]; eNew=Frame_x[arg,e]` | 2 |
| `Lookup_x[k]` / `Frame_y[v,e]`, x=y | `k=Ret[v]; e=Erase[]` | 2 |
| `Lookup_x[k]` / `Frame_y[v,e]`, x!=y | `e=Lookup_x[k]; v=Erase[]` | 2 |
| `Ret[v]` / `Done[out]` | `v=out` | 0 |

For every passive data constructor `T_a[c1,...,cr]`, including syntax, closures, environment frames, Empty and Atom, the following local rules apply. a is either no atom or the carried atom, copied exactly through the allowed reference interface:

`D[u,v] / T_a[c1,...,cr]` emits

`u=T_a[u1,...,ur]; v=T_a[v1,...,vr]; cj=D[uj,vj] (j=1,...,r)`.

This allocates `2+r<=4` agents and makes two physically distinct data trees. It does not copy arbitrary graph fragments or arbitrary running agents. `Erase[] / T_a[c1,...,cr]` emits `cj=Erase[]` for each child. It allocates at most two agents. These rules *must themselves be acquired by K*; the host has no deep-copy or recursive-free operation.

This table specifies every collision needed for a well-formed, closed simply typed source term with an atom result. Types are a domain restriction, not supplied input annotations or a host type checker. An unexpected collision can become a locally blocked error pair preserving its auxiliary connections; malformed/unreadable output or fuel exhaustion is scored as failure. The intended typed domain does not use this error route. Untyped divergent source is operationally bounded by the external resource cap; no termination claim extends to it.

The raw stream has produced a parse stack, syntax, closures, environments, lookup traversal, copy processes, application continuations and execution sites. Every one is an inspectable graph creation/reconnection by K. There is no second neural encoder or symbolic evaluator running behind the table.

## 3. Complete shadowing/capture and duplicated-function example

Use distinct raw atoms a,b as ground values and x,y,f as lexical names. The term is

`(lambda x. (lambda f. (lambda x. f (f x)) b) (lambda y. x)) a`.

The only initial graph is Start followed by this exact raw chain:

`A L x A L f A L x A V f A V f V x C b L y V x C a End`.

There are no binder aliases or function links in that initial graph. P/A creates an AwaitFun frame; P/L then ReadL stores x in AwaitLam_x and continues along the raw stream. At the substream `A V f A V f V x`, the parser constructs the inner App from the returned Var_f and Var_x nodes before it returns the outer App. AwaitLam_x encloses this result; AwaitArg combines it with the Atom_b made from `C b`. The enclosing AwaitLam_f, the separate `L y V x`, and the two outer AwaitArg frames complete the tree. PCheckEnd sees End, allocates Empty and Done, and creates the first Eval pair. Each step is one of the wired rows above; no recursive host parse occurs.

The stream has n=25 raw tokens and produces m=15 syntax nodes, so the stated parser count is 42 local interactions. This is a hand-derived count, not a recorded execution. The resulting syntax, written compactly for the reader, is

`App(Lam_x(App(Lam_f(App(Lam_x(App(Var_f,App(Var_f,Var_x))),Atom_b)),Lam_y(Var_x))),Atom_a)`.

### 3.1 What actual binding graph forms

Below E1,E2,E3 denote environment *tree shapes*. Superscripted occurrences in actual execution are physically distinct copies made by D, not uncharged shared pointers.

1. The outer App rule copies Empty, evaluates Lam_x to a closure capturing one Empty, evaluates Atom_a in the other branch, then the Apply/Closure rule **allocates** `E1=Frame_x[Atom_a,Empty]`. The body starts evaluating with this newly wired scope.
2. Its App copies E1. Lam_f becomes a closure with one copy; Lam_y becomes `F=Closure_y[Var_x,E1']` with the other. F's body names x but stores no precomputed answer or direct occurrence-to-value alias. Applying Lam_f to F allocates `E2=Frame_f[F,E1'']`.
3. Inside that body, the Lam_x used on b captures a copy of E2. Applying it to Atom_b allocates `E3=Frame_x[Atom_b,E2']`. The current nearest x is now b, while F's captured environment still has x=a.
4. Evaluation of `App(Var_f,App(Var_f,Var_x))` copies E3 using D. Each recursive D/constructor collision makes distinct corresponding Frame and Closure nodes, preserving carried name identities and copying F's **passive** body/environment trees. The left f lookup and the right inner application receive distinct copies of E3. The inner application copies its own E3 again for f and x.

In graph form the caller binding chain is

`E3 -> Frame_x(b, Frame_f(F, Frame_x(a, Empty)))`,

where each F has its own attached tree

`Closure_y(Var_x, Frame_x(a,Empty))`.

The real graph has explicit Frame/Closure child wires, copied constructor by constructor. These schematic equal trees do not hide shared host environments.

### 3.2 Both uses of the captured function execute

- Left `Lookup_f` on its E3 copy compares f with x, skips the inner x=b frame and erases that copied value, then matches Frame_f. It returns `F_left`, erasing the unused parent environment.
- The inner application's `Lookup_f` similarly returns `F_right`. Its `Lookup_x` matches its E3's nearest Frame_x and returns b.
- `Apply[b,k] / F_right` reconnects the function's body to a new Eval and allocates `Frame_y[b,E1_right]`. **Its parent is the closure's captured E1_right, not the caller's E3.** The next Lookup_x skips y, erases the ignored b, matches captured x=a, and returns a.
- `Apply[a,k] / F_left` independently allocates `Frame_y[a,E1_left]`. Lookup_x again skips y and returns its captured a. Ret/Done joins that atom's principal wire to the external output.

The result is a. A dynamic-scope policy using the caller's inner x would return b. The two calls used independently constructed copies of the same captured higher-order function; no cached answer, host function call or arbitrary running-graph copier was used.

The existence of this trace is not an assertion that training finds it. It is also not a claim of a new lexical evaluator. Its role is to expose exactly what local computational organization must form and what output tests can distinguish it.

## 4. Structural invariants and the copying obstacle

**Scope invariant.** Each live closure carries a passive body and the environment present when that closure was formed. Apply places its argument in a newly allocated Frame whose parent wire comes from that closure. Lookup traverses the generated Frame chain from its head, stopping at the first atom-equal name. Induction over the execution rows gives ordinary lexical call-by-value binding on the specified domain. The argument requires these rows to be correct; nominal covariance alone does not supply them.

Root supplied a useful counterexample: `((lambda x. ((lambda x.x) b)) a)` and `((lambda x. ((lambda y.y) b)) a)` are alpha-equivalent and should both return b. A wrong earliest-name-binding strategy returns a on the first and b on the second while still respecting every global name permutation. This protocol uses the nearest generated Frame and passes the handwritten reasoning for both. Actual acquisition must be tested on such distinctions, not only global fresh-name renaming.

**Freshness/ownership invariant.** Every allocated scope, continuation and data clone is a fresh graph node. Copying a name reference preserves atom identity; copying a Frame/Closure does not reuse its structural node identity. No destructive variable substitution occurs and no fresh lexical name is needed to avoid capture. Consequently a closure's copied captured x frame cannot accidentally become the caller's distinct inner x frame through node aliasing. This relies on actual pointer discipline, which must be checked in implementation.

**Copy-domain invariant.** At every Eval/App, D receives an environment consisting only of passive Frame/Closure/syntax/Atom/Empty constructors. New closures combine two owned passive trees. Application consumes its owned closure/argument and creates a fresh Frame; branches receive D-generated copies. Thus D never receives a running Eval, Ret, continuation or arbitrary active net. The protocol copies closed *data descriptions* of a function and its captured values, then activates them through Eval. This explicitly avoids Lafont's arbitrary-executable-package duplication obstruction; it does not solve optimal or arbitrary cyclic graph duplication.

Intermediate copying may stream: an evaluator can wait at a D auxiliary output until a constructor copy becomes available. A valid explanatory schedule can finish each finite copy before using either output. The uncensored local diamond from v0 then removes dependence of a completed normalizing result on independent event order. Resource caps, space peaks and malformed candidates remain separate.

**Resource account.** Parsing is O(n) as above. Copying a passive constructor tree with N nodes requires exactly N D/constructor interactions and creates `3N-1` agents after the initial D, including internal D actors and the two output trees. Erasing an N-node tree requires N interactions and N-1 generated Erase actors after the initial one. Name comparisons and each neural kernel evaluation are extra metered work. Every application copies its caller environment; nested captured closures can make that tree and total work grow very rapidly, including exponential duplication. No efficient-sharing or optimal-reduction claim is made. The fixed four-agent local emission limit constrains one event, not total computation.

## 5. Precise acquisition interface

Only the generic port emitter, raw tokenizer/chain initializer, optional nominal reference interface, resource limits and terminal atom feedback are installed. None of the named witness codes, case rows, parser stack, copy domain, continuation grammar or correct intermediate graphs is supplied as a rule library or training label. A future implementation may encode the table solely as a separate reference/checker; it must not initialize the learner with it and then call the acquired evaluator emergent.

For raw input x, freeze the same initial chain construction and sample every local descriptor/code/reference/wiring choice from the one shared `p_theta`. The terminal loss compares the actual output atom with the requested atom, or assigns a fixed failure loss for unreadable/stuck/exhausted execution. The selection objective includes all training and execution work, allocation, maximum live graph size and retained parameters. Within an episode theta is fixed. Between episodes use the ordinary score-function update

`theta <- theta - eta * (loss + lambda*cost - baseline(x)) * sum_events grad_theta log p_theta(emission_e | local_pair_e)`.

The log probability includes structural code bits, atom-reference selector choices and port matchings. No teacher rewrite trace or gradient through a correct parser/evaluator enters. Own successful traces may be replayed for ordinary self-imitation, but they are generated evidence, not supplied traces; replay and failed searches count. Causal-key draws from v0 can couple schedules, without affecting nominal atom identity.

An extensional curriculum can begin with `C a -> a`, progress to identity/constant functions, then to capture/shadowing and duplicated closures. It still supplies only raw I/O and finite failure/cost feedback. Such a curriculum is a substantial designed bias and must be given equally to competitors. Short tasks alone admit atom-selection shortcuts. The decisive held-out cases must vary syntax depth, lexical shadowing, ignored arguments, repeated higher-order use and atom identities jointly. We have not frozen or run that study.

The constructive acquisition bias is now specific: the emitter can only change a bounded principal pair and can carry exact selected boundary wires into newly generated nodes. It can turn learned body/environment/continuation roles into persistent physical interfaces that work at any depth, instead of relearning an arbitrary depth-indexed action. Once a closure captures a particular environment wire, Apply can use that wire locally; access to the caller environment would require the learner to have explicitly retained and transported another route. This is a causal representation constraint, not a semantic oracle. Whether terminal training finds this useful organization is the one central OPEN bridge.

## 6. Prior-work and rival boundary

The witness is a local implementation of a familiar closure/environment evaluator with explicit tree copying. Its parsing stack, lexical scope semantics and tree-copy recursion are not new algorithms. The substantial architectural target is **learning their operational organization from the initial raw stream using one local graph-generating law**. Existing neural algorithm learners already make raw-input/final-output acquisition possible without a supplied symbolic evaluator; “learn execution instead of a given interpreter” cannot by itself be the novelty claim.

After constructing the witness, this worker entered Sinot's original **Call-by-Name and Call-by-Value as Token-Passing Interaction Nets** (2005), §§3.4–3.7 and §4. It already supplies explicit eval/return tokens, copying/erasure and call-by-value control. Its inspected translation starts from lambda syntax and places binder wires explicitly. This is concrete prior coverage of the evaluation organization; it is not evidence that terminal-feedback learning of the raw-stream protocol was already achieved. The source reading and limits are appended to `reading.md`. Our environment-tree implementation is a reconstruction in the same established implementation tradition, with a different worked representation, not a newly invented evaluator.

The supporting source worker's actual Neural GPU comparator is especially serious: raw token embedding grid, one shared learned local recurrent update and final-output supervision, with no symbolic parser/binder/copy protocol installed. A matched comparison must give it the same raw data/curriculum, nominal/equality information if used, state precision, parameter/search/training/runtime budgets and input/output interface. NPS, NTM, NPI and differentiable Forth have different supplied resources and must be described from their actual originals. The separate `sources/after014-higher-order/conventional-rival.md` is the supporting worker's source record, not our invented historical coverage proof.

Our concrete difference from a fixed-grid local neural learner is that the learned transition can allocate, consume and reconnect exact persistent communication links, and can form active scope/copy/evaluation processes on those links. A Neural GPU may emulate the same behavior through its state; that fact is not an equal-learning-cost proof. Conversely, merely showing successful shadowing with this witness or a trained graph machine would not establish an advantage over it. Generic graph-program induction is also an appropriate paradigm-level rival. Calling the same hypothetical kernel learner “old” would not constitute original-source evidence.

## 7. Present status and next single decision

**CONSTRUCTED, UNEXECUTED:** one emitter's complete successful-path local rule protocol from a raw chain through generated parse and lexical graphs, with an explicit duplicated captured-function trace and finite local resource bounds. The protocol avoids a host parser/stack/binder and avoids copying arbitrary active code. The hand witness uses a finite set of learned target roles; it does not demonstrate invention of roles or a new evaluator.

**OPEN:** acquisition from terminal feedback under honest total costs, particularly discovering the joint code/copy/environment discipline instead of an output shortcut. The architecture can express this discipline and exposes its causal constraints; no trained evidence currently shows that this bias improves its accessibility. Original token-passing evaluation has now been entered and covers the broad execution organization; learning/new-combination coverage remains incomplete. No novelty, universal neural-learning or paradigm conclusion is drawn.

Next single decision: use this fully specified witness to identify the most informative finite acquisition comparison against the actual raw-input Neural GPU/local recurrent comparator, while preserving capture, shadowing and repeated higher-order use. This is a scientific design decision; it does not authorize an experiment in this worker. MAIN remains an unfinished high-risk architecture, not a completed campaign.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

- Raw-chain parsing, environment formation and execution can use only arity<=2, <=4-new-agent local collisions. | CONSTRUCTION, manually checked | explicit port equations and event counts above | no executed port validator | global encoder/preencoded binder removed | independent wiring audit.
- Captured f used twice returns a under an inner x=b. | RECONSTRUCTION | complete generated-environment/lookup/apply trace | no execution; assumes witness rows | dynamic scope/earliest-name policies distinguished | preserve as causal target, not acquisition evidence.
- Copying the protocol's passive function descriptions is legitimate local tree copying. | RECONSTRUCTION with known substrate | copy rule and ownership invariant | potentially explosive total cost; arbitrary active/cyclic copy unsolved | unrestricted COPY removed | account all copies in any future study.
- Terminal feedback can teach this one kernel the organization efficiently. | HYPOTHESIS | explicit acquisition interface | no learning result or acquisition-cost advantage | nominal symmetry alone insufficient; finite rule identification not promoted | discriminating comparison design before execution.
