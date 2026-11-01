# RECONSTRUCTION: bounds for candidate phase-relation computations

This is a mathematical audit note, not a source result or evidence of a learned architecture. It is not sent to independent branches before v0 freezes.

1. Gauge/reference accounting. If a unit-circle relation is z_c=z_a z_b, a common phase rotation z_i→exp(iα)z_i changes its left and right sides by different powers. Encoding operands relative to a reference z_r repairs it as z_c z_r=z_a z_b, i.e. θ_a+θ_b−θ_c−θ_r=0 (mod2π). The coefficient vector sums to zero, so common phase rotations cancel. A reference or equivalent gauge-fixing cost is designed infrastructure. This computes phase addition/unit-circle multiplication, not unrestricted real-number multiplication.

2. For E(θ)=κ[1−cos(nᵀθ−β)], θdot=−∇E=−κ sin(nᵀθ−β)n. Near a satisfying relation, Hessian H=κnnᵀ. It stabilizes the constrained direction, with neutral directions tangent to the relation; this need not erase operand values if they are clamped or remain in the intended neutral subspace.

3. For several simultaneously active relations at a satisfying point, H=Nᵀ diag(κ)N. On the free-variable subspace, positive eigenvalues give local transverse contraction. This is a local statement; it does not guarantee a globally reachable unique solution, a valid discrete Euler implementation, or compatibility of learned relations.

4. Minimal composition conflict: equal-strength energies κ[1−cos ψ] and κ[1−cos(ψ−π)] sum to constant 2κ. Each alone enforces a valid relation, while their joint force vanishes. Thus composition cannot mean unconditionally activating all learned relations. A valid mechanism must specify context/gates, conflict handling and its discovery cost. A controller selecting the right relations may itself do the difficult computation.

These calculations can discriminate a genuine relational-computation mechanism from generic synchronization or a precompiled constraint system. They do not establish novelty, physical efficiency, acquisition from experience, or practical superiority.
