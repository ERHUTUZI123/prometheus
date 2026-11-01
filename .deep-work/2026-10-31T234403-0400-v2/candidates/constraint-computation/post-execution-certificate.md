# Post-execution analytic clarification: deformed contours and an exact unsafe control

RECONSTRUCTION by root after the frozen central check. This is a new derivation, not a change to the pre-execution contract/code or a retroactively claimed predeclared certificate. No new run, parameter choice or numerical tuning. The original49/64 lemma covers the exact unit loop. The actual noisy loops stay in a slightly wider annulus, so their guarantee needs an additional argument.

## Uniform certificate for all declared old-loop perturbations

The code's perturbation has norm <=.025 sqrt(2)<.05 of the unit radius. Thus every old contour and its straight deformation from the unit circle lies in .95<=|u|<=1.05.

For P=z²-u the old roots have magnitudes between sqrt(.95) and sqrt(1.05). Coarse rational bounds .97<sqrt(.95) and sqrt(1.05)<1.03 suffice. Radius1/4 disks around the two old roots are disjoint. On each boundary,

    |other-root distance| >= 2*.97-.25 = 1.69,
    |z-3| >= 3-1.03-.25 = 1.72,
    |(z-3)(z²-u)| >= .25*1.69*1.72 = 7267/10000.

On the radius1/4 disk around3, |z|>=11/4, so

    |(z-3)(z²-u)| >= (1/4)[(11/4)²-21/20] = 521/320.

All declared safe epsilon values are <=1/5<7267/10000. Rouché therefore identifies each old root and the new root through the entire coefficient homotopy AND the declared path deformations. Old noisy paths have the same winding class as the unit loop. This supplies an analytic preservation argument for the actual tested noisy family, rather than extrapolating from sampled tracking success. It remains a holomorphic polynomial result.

## Exact discriminant argument for the unsafe epsilon4 case

For Q=z³-3z²-uz+3u-epsilon, direct cubic-discriminant expansion gives

    Delta(u)=4u(u-9)²+108 epsilon(u-1)-27 epsilon².

At epsilon4,

    Delta(u)=4[u³-18u²+189u-216].

On |u|1, |u³-18u²+189u|<=208<216. Another application of Rouché shows that this discriminant has NO zeros in the unit disk (the dominating constant has none). With fixed nonzero leading coefficient, Q consequently defines an unbranched finite cover there; the disk is simply connected, so the old unit loop has identity monodromy on every sheet. This predicts the observed loss of the old transposition without treating violation of a sufficient bound as automatic failure.

The coefficient migration to4 uses the explicitly predeclared complex detour to avoid a resting-control collision. Its branch labels are honestly migrated; independent root reseeding was not used. The unit-disk argument alone does not cover noisy paths. To extend it explicitly, let f(u)=u³-18u²+189u-216. Its real derivative is3[(u-6)²+27]>0; f(6/5)<0 and f(7/5)>0. Its only real root alpha therefore lies in(1.2,1.4). The other roots are a conjugate pair with modulus sqrt(216/alpha)>12, by the product of roots. Hence ALL discriminant roots have modulus>1.2. The entire noisy old-path family lies inside1.05, so the unsafe identity conclusion also covers those paths. This additional algebra is post-execution reasoning, not a claim that the initial sufficient bound's violation alone predicted failure.

## What the executed check means

Safe cases e=.005,.05,.2: A=[1,0,2], B=[0,2,1], actual AB=[2,0,1], BA=[1,2,0], AA=identity for all three fixed perturbation seeds. Coefficient migration physically moves anchors; it does not preserve numeric output coordinates. Both old and new loops agree with their finer discretization. The new B path is regenerated per epsilon from one PAID critical-point solve; B's retention across epsilon was NOT tested.

Unsafe4: A=identity and AB=BA in this example. All data in probes/constraint-growth/raw.jsonl and summary.json. State-root oracles verify only; critical-point solves design control paths and are separately counted. The process ran2.182999seconds. Root comparisons to an analytic continuation/state-machine constructor, and useful neural acquisition, remain OPEN; no performance/novelty/physical-energy claim follows.
