# Finite-time retention for an entire learned operation family

ROOT RECONSTRUCTION, same human-locked topology V2, M0 NOT PASSED / M3 goal. This is conditional mathematics, not numerical or hardware data and not a new-physics/novelty claim. It strengthens the execution interface of root-uniform-chart rather than changing the milestone or reviving a parked branch.

## Reference the actual geometry, not an incorrectly perfect nominal shortcut

For each actual plant lambda=(c,kappa) in the paid uncertainty set U, let n_c(t) be its actual normalized Y direction along the requested ordered contour. Define an ANALYTICAL reference Hamiltonian

    H_ref,c = H0,c + Hcd,c,
    Hcd,c = i/2 sum_(i<j)(n_j dot n_i - n_i dot n_j) gamma_i gamma_j.

The controller does not know or implement this reference. Its actual correction is f_kappa(q_hat), where q_hat is computed by nominal inverse compensation of Hcd,chat. The difference from the analytical reference is entirely a correction residual

    E_lambda = f_kappa(q_hat) - h_cd,c.

This comparison is valid even though nominal Hcd is wrong. Each reference's final geometric operation is identical, because the chart cancels BOTH geometric moments for every c in the prior. Gap g=1 contributes the same idle-code global phase. Uncontrolled calls are the stated interface. Thus Duhamel bounds actual finite execution error by integral ||E_lambda||, without incorrectly treating an integral H0 geometry error as a necessary error growing with duration. Endpoint/smoothness and available couplers remain actual premises.

Assume a declared command range on which nominal inverse compensation has uniform relative coefficient error <=e, as supplied conditionally by calibration-p Eq13 (or the exact finite-prior maximum). For the three outer couplings in each active Y, the Hermitian bilinears anticommute and square to I. Their vector coefficient norm therefore gives their exact operator norm. Componentwise relative error bounded by e gives a correction-response contribution <=e*integral||h_cd,chat||. This uses actual independent component inversion; a cubic response is not a vector gain.

Let delta1=max_U|c1-chat1|, delta2=max_U|c2-chat2|,

    F=(pi/2)(delta1/4 + sqrt(3)*delta2/18).

Then |phi_c-phi_chat|<=F because max P1=1/4, max |P2|=sqrt(3)/18. On a vertical sweep, h_cd has norm |dot vartheta|/2 and its axis rotates by that phi difference. Consequently its integrated geometry-mismatch contribution is <=(pi/4)*F per north/equator sweep. On an equator connector,

    h_cd,12= -dot phi/2,
    |delta h_cd,12| <= (pi/4)|dot x|(delta1+delta2),

using max|P1'|=max|P2'|=1. Exact integrals can sharpen this deliberately conservative bound.

## Echo and old operations are fully charged

A half-loop uses two disjoint rectangles of total width a. Its four meridian sweeps have correction area pi, and its two equator connectors have total correction area pi*a/4, because the nominal transformed widths still sum to a. Thus

    A_half = pi+pi*a/4.

The echo uses a forward and a reversed half-loop plus TWO physical logical X calls. Each X is TWO routed exchanges, and each three-edge exchange has correction area3pi/4. Therefore

    A_echo = 2 A_half + 4*(3pi/4) = 5pi+pi*a/2.

The eight half-loop meridian sweeps contribute <=2pi*F. The two half-loops' equator connectors contribute <=pi*a*(delta1+delta2)/2. A routed edge exchange has no vertical phi mismatch at its pinned endpoints, but its equator timing is still warped; bound its remaining mismatch by pi*(delta1+delta2)/4. Four such exchanges in the echo contribute <=pi*(delta1+delta2). The complete conditional bound is therefore

    eps_echo <= (5pi+pi*a/2)*e
                +2pi*F +pi*(1+a/2)*(delta1+delta2)
                +r*T + eps_implementation + rad(B)/2.       (1)

The gate is Rz(beta_hat) with a=beta_hat/pi, after the specified echo/global-phase convention. Static diagonal parasitic phases cancel only under the stated repeatability/echo assumptions; otherwise they are included in a separately justified residual r, not assumed away. Preparation/readout/idle or word requirements add their actual terms. No unknown coherent state is used for destructive calibration.

For a single old routed exchange the corresponding bound is

    eps_exchange <= (3pi/4)*e +pi*(delta1+delta2)/4
                    +r*T +eps_implementation.               (2)

These are sufficient scores, NOT lower bounds on true error. Finite-time error cancellations can make them very conservative. Numerical model propagation is an independent sharper diagnostic; a sampled small error does not prove these scores pessimistic uniformly.

## Actual uniform protected-learning freedom and its cost

Take max a over the fixed task range, bound T/command amplitude/slew and implementation error uniformly across the chart, and evaluate e,delta1,delta2 from the SAME actual calibration event. If (1) fits every retained task/word allocation, then ANY later shared rho_theta(a) update inside the chart/range preserves these executable error contracts without individually repropagating every old contour. The library size does not enter this sufficient uniform certificate. This is a conditional, explicit improvement of the proposed maintenance INTERFACE, not an observed cost improvement over competitors. New task accuracy or duration demands can require fresh information or tighter bounds. Task target intervals/program scalars still consume growing memory; failed updates and all control/observation work count.

Crucially, a legal direct idle pulse implements the same Rz(beta) with area A_direct=beta/2=pi*a/2 for the FULL echoed gate, requiring only the common kappa response, not c calibration or a geometric echo. Its uniform response-only score is A_direct*e. It too has a uniform all-target certificate and can use the same acquired target information. Under the current deliberately memoryless/no-other-noise model, the topological construction's global-certification property alone cannot establish an important retention advantage: the direct competitor has the same property with less calibration burden and smaller correction area. Gap-idle action, ramps/full Hamiltonian norm, hardware prices and actual cancellations must still be compared honestly; this is not a blanket physical dominance theorem.

The useful scientific outcome is now sharper: paid uncertainty determines whether genuine shared geometric freedom is executable; when U collapses to a true finite hypothesis, all the reference residuals vanish ideally for both topology and direct control. A positive topology benefit needs an ACTUAL additional mechanism/noise/acquisition tradeoff beyond this common-inverse model, not merely more tasks, stronger neural generators or per-old-context penalties assigned to the rival. The bounded current diagnostic can locate the finite-model safe set and real overhead; it cannot establish that missing general advantage by itself.

STATUS: by-hand conditional derivation, pending independent examination and any numerical diagnostic. Claim/version/hypothesis remain separate. MAIN is not changed or parked by the comparison.
