"""Bounded acquired-frontier pilot; root must inspect/freeze before executing.

NumPy only. Generated environment fields are instruction-isolated, not a security
sandbox. Learners read typed caches; reference/scout state is paid and released.
The adaptation-aware port baseline is MAML-Rep-style, not exact OML Algorithm 2.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
N, D = 6, 2
P = N * N + N * D + N
SEEDS = (5501, 5502, 5503)
NS, NQ = 8, 8
K, LR = 8, 0.25
SCOUT_LENGTH = 32
PRODUCTION_LENGTHS = (64, 128)
DELAYS = (8, 64)
LAST_M = 16
TOL = 0.01
FD_EPS = 1e-3
META_STEPS, META_LR = 6, 0.20
TIME_LIMIT = 110.0
RAW = None
START = None


def emit(kind, **fields):
    row = dict(kind=kind, **fields)
    data = json.dumps(row, allow_nan=False)
    assert json.loads(data)["kind"] == kind
    RAW.write(data + "\n")
    RAW.flush()


def guard():
    if time.monotonic() - START > TIME_LIMIT:
        raise RuntimeError("internal 110-second guard; preserve partial evidence")


class Work:
    """Operation counts are explicit proxies, not FLOPs/RSS/energy."""
    def __init__(self):
        self.v = dict(rnn_steps=0, bptt_steps=0, dense_multiply_proxy=0,
                      native_beta_steps=0, meta_objectives=0,
                      labels_used=0, covariance_updates=0,
                      ridge_fit_calls=0, decoder_evaluations=0)

    def add(self, name, count=1):
        self.v[name] += int(count)

    def merge(self, other):
        for name, count in other.v.items():
            self.add(name, count)


def numeric_bytes(*objects):
    """Unique NumPy allocation bytes, including view bases; Python overhead excluded."""
    seen = set()

    def visit(obj):
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            key = id(base)
            if key in seen:
                return 0
            seen.add(key)
            return int(base.nbytes)
        if isinstance(obj, dict):
            return sum(visit(x) for x in obj.values())
        if isinstance(obj, (tuple, list)):
            return sum(visit(x) for x in obj)
        return 0

    return sum(visit(x) for x in objects)


def unpack(theta):
    return (theta[:N*N].reshape(N, N),
            theta[N*N:N*N+N*D].reshape(N, D), theta[-N:])


def rnn_forward(theta, x, work, keep_tape=True):
    """x=(batch,length,D); last state and, optionally, an actual working tape."""
    w, u, bias = unpack(theta)
    batch, length, _ = x.shape
    h = np.zeros((batch, N))
    tape = np.empty((batch, length + 1, N)) if keep_tape else None
    if tape is not None:
        tape[:, 0] = h
    for t in range(length):
        h = np.tanh(h @ w.T + x[:, t] @ u.T + bias)
        if tape is not None:
            tape[:, t + 1] = h
    work.add("rnn_steps", batch * length)
    work.add("dense_multiply_proxy", batch * length * (N*N + N*D))
    assert np.all(np.isfinite(h))
    return h, tape


def rnn_vjp(theta, x, tape, cotangent, work):
    """Exact per-instance BPTT VJP for the fixed producer parameter version."""
    w, _, _ = unpack(theta)
    batch, length, _ = x.shape
    adj = np.asarray(cotangent, dtype=float).copy()
    gw = np.zeros((batch, N, N))
    gu = np.zeros((batch, N, D))
    gb = np.zeros((batch, N))
    for t in range(length - 1, -1, -1):
        delta = adj * (1.0 - tape[:, t + 1] ** 2)
        gw += delta[:, :, None] * tape[:, t, None, :]
        gu += delta[:, :, None] * x[:, t, None, :]
        gb += delta
        adj = delta @ w
    work.add("bptt_steps", batch * length)
    work.add("dense_multiply_proxy", batch * length * (2*N*N + N*D))
    out = np.concatenate((gw.reshape(batch, -1), gu.reshape(batch, -1), gb), axis=1)
    assert out.shape == (batch, P) and np.all(np.isfinite(out))
    return out


def rnn_jacobian(theta, x, tape, work):
    batch = len(x)
    ans = np.empty((batch, N, P))
    for j in range(N):
        seed = np.zeros((batch, N))
        seed[:, j] = 1.0
        ans[:, j] = rnn_vjp(theta, x, tape, seed, work)
    return ans


def unit(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        ans = np.zeros(N)
        ans[0] = 1.0
        return ans
    ans = np.asarray(vec, dtype=float) / norm
    if ans[int(np.argmax(np.abs(ans)))] < 0:
        ans = -ans
    return ans.copy()


def principal(cov):
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    return unit(vectors[:, int(np.argmax(values))])


def compose(q1, q2):
    value = np.tanh(q1 + .4*q2 + .2*q1*q2)
    factor = 1.0 - value*value
    return value, factor*(1.0 + .2*q2), factor*(.4 + .2*q1)


def q_value(beta, u, feature):
    w = beta[0]
    return u + w + w*w*(feature @ beta[1:])


def q_parameter_grad(beta, feature, loss_cotangent):
    w = beta[0]
    c = beta[1:]
    grad = np.empty_like(beta)
    grad[0] = np.sum(loss_cotangent * (1.0 + 2*w*(feature @ c)))
    grad[1:] = w*w * (feature.T @ loss_cotangent)
    return grad


def adapt_single(u, feature, labels, work):
    beta = np.zeros(1 + feature.shape[1])
    for _ in range(K):
        value = q_value(beta, u, feature)
        grad = q_parameter_grad(beta, feature, (value-labels)/len(labels))
        beta -= LR*grad
        work.add("native_beta_steps")
        work.add("dense_multiply_proxy", len(labels)*(4 + 4*feature.shape[1]))
        assert np.all(np.isfinite(beta))
    return beta


def adapt_pair(us, features, labels, work):
    betas = [np.zeros(1 + f.shape[1]) for f in features]
    for _ in range(K):
        qs = [q_value(b, u, f) for b, u, f in zip(betas, us, features)]
        value, c1, c2 = compose(*qs)
        cot = (value-labels)/len(labels)
        grads = [q_parameter_grad(betas[0], features[0], cot*c1),
                 q_parameter_grad(betas[1], features[1], cot*c2)]
        # Simultaneous native SGD from the same pre-update state.
        betas = [b-LR*g for b, g in zip(betas, grads)]
        work.add("native_beta_steps", 2)
        work.add("dense_multiply_proxy", len(labels)*(16 + 4*sum(f.shape[1] for f in features)))
        assert all(np.all(np.isfinite(b)) for b in betas)
    return betas


class Environment:
    """Its teacher/readout fields are never arguments to formation algorithms."""
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.theta = []
        for _ in range(2):
            w = .35*np.eye(N) + rng.normal(0, .06/np.sqrt(N), (N, N))
            u = rng.normal(0, .35/np.sqrt(D), (N, D))
            bias = rng.normal(0, .03, N)
            self.theta.append(np.r_[w.ravel(), u.ravel(), bias])
        self.a = unit(rng.normal(size=N))
        self._teacher = [x.copy() for x in self.theta]
        self._hidden_b = [np.zeros(N), np.zeros(N),
                          1.5*unit(rng.normal(size=N)), 1.5*unit(rng.normal(size=N))]
        self.teacher_work = Work()

    def new_episode(self, seed, context, length, count=NS+NQ):
        rng = np.random.default_rng(seed)
        return dict(context=int(context), length=int(length),
                    xa=rng.normal(size=(count, length, D)),
                    xb=rng.normal(size=(count, length, D)),
                    _mu=float(rng.uniform(.6, 1.2)))

    def reveal(self, episode, indices):
        # Environment-only computation; h_teacher is not returned to a learner.
        ha, _ = rnn_forward(self._teacher[0], episode["xa"][indices],
                            self.teacher_work, False)
        hb, _ = rnn_forward(self._teacher[1], episode["xb"][indices],
                            self.teacher_work, False)
        mu = episode["_mu"]
        y = ha @ self.a + mu + mu*mu*(hb @ self._hidden_b[episode["context"]])
        return y

    def fresh_query(self, original, seed):
        # New sequences for the SAME external procedure/task; mu is not learner input.
        ans = self.new_episode(seed, original["context"], original["length"], NQ)
        ans["_mu"] = original["_mu"]
        return ans


class Cache:
    """Only this object is available to a deployed consumer/partial-SGD routine."""
    def __init__(self, kind, u, ju, payload=None, derivative=None):
        self.kind = kind
        self.u = np.asarray(u).copy()
        self.ju = np.asarray(ju).copy()
        self.payload = None if payload is None else np.asarray(payload).copy()
        self.derivative = None if derivative is None else np.asarray(derivative).copy()

    def arrays(self):
        return (self.u, self.ju, self.payload, self.derivative)

    def nbytes(self):
        return numeric_bytes(self.arrays())


def spec(kind, p=None, decoder=None):
    return dict(kind=kind, p=None if p is None else unit(p), decoder=decoder)


def feature_dim(kind):
    return {"u": 0, "port": 1, "cache": N, "full_projected": 1,
            "full": N, "raw": N, "suffix_raw": N}[kind]


def decode_features(z, jac, decoder, work):
    """Reads only actual (z,Jz), not hB; J columns are STOP_GRAD for producer SGD."""
    raw = np.concatenate((z.reshape(-1, 1), jac), axis=1)
    standardized = (raw-decoder["mean"])/decoder["std"]
    design = np.concatenate((standardized, np.ones((len(raw), 1))), axis=1)
    work.add("decoder_evaluations", len(raw))
    work.add("dense_multiply_proxy", len(raw)*design.shape[1]*N)
    return design @ decoder["coef"]


def produce_cache(theta, a, episode, choice, work, tag="unspecified"):
    """Emit the current A capability before any future B work or role labels."""
    xa, xb = episode["xa"], episode["xb"]
    ha, ta = rnn_forward(theta[0], xa, work, True)
    ju = rnn_vjp(theta[0], xa, ta, np.broadcast_to(a, ha.shape), work)
    current_u = ha @ a
    emit("current_capability_emitted", tag=tag, context=episode["context"],
         count=len(current_u), current_u=current_u.tolist(), before_B_work=True,
         support_query_roles_available=False)
    # Actual arrays required during A return; no hA/tape survives this function.
    working_peak = numeric_bytes(xa, ha, ta, ju, current_u)
    del ha, ta
    base_bytes = numeric_bytes(current_u, ju)
    kind = choice["kind"]
    if kind == "u":
        cache = Cache(kind, current_u, ju)
    elif kind in ("raw", "suffix_raw"):
        # A real copied suffix buffer, not a view retaining the full input allocation.
        retained_x = xb if kind == "raw" else xb[:, -LAST_M:].copy()
        cache = Cache(kind, current_u, ju, retained_x)
        working_peak = max(working_peak, base_bytes + numeric_bytes(retained_x))
    else:
        hb, tb = rnn_forward(theta[1], xb, work, True)
        if kind in ("port", "cache"):
            p = choice["p"]
            z = hb @ p
            jz = rnn_vjp(theta[1], xb, tb, np.broadcast_to(p, hb.shape), work)
            cache = Cache(kind, current_u, ju, z[:, None], jz)
            working_peak = max(working_peak, base_bytes + numeric_bytes(xb, hb, tb, z, jz))
        elif kind in ("full", "full_projected"):
            jac = rnn_jacobian(theta[1], xb, tb, work)
            cache = Cache(kind, current_u, ju, hb, jac)
            working_peak = max(working_peak, base_bytes + numeric_bytes(xb, hb, tb, jac))
        else:
            raise ValueError(kind)
    return cache, int(working_peak)


def exact_scout(theta, a, episode, env, work):
    """One capped full-state probation episode; caller releases h/J/tapes after use."""
    cache, peak = produce_cache(theta, a, episode, spec("full"), work, "scout")
    support = np.arange(NS)
    query = np.arange(NS, NS+NQ)
    ys = env.reveal(episode, support)
    work.add("labels_used", NS)
    beta = adapt_single(cache.u[support], cache.payload[support], ys, work)
    prediction = q_value(beta, cache.u[query], cache.payload[query])
    yq = env.reveal(episode, query)  # prediction already frozen
    work.add("labels_used", NQ)
    loss = float(np.mean(.5*(prediction-yq)**2))
    g = beta[0]**2 * beta[1:]
    # Query producer partial gradients: beta is held fixed, not a hypergradient.
    lam = (prediction-yq)/NQ
    ga = np.sum(lam[:, None] * cache.ju[query], axis=0)
    gb = np.einsum("i,j,ijp->p", lam, g, cache.derivative[query])
    native = [theta[0]-.01*ga, theta[1]-.01*gb]
    assert all(np.all(np.isfinite(x)) for x in native)
    return dict(cache=cache, beta=beta, ys=ys, yq=yq, prediction=prediction,
                loss=loss, g=g, peak=peak,
                native_gradient_norm=[float(np.linalg.norm(ga)), float(np.linalg.norm(gb))])


def cached_choice(cache, choice):
    """Probation-only candidate from the paid full reference, with actual arrays."""
    kind = choice["kind"]
    if kind == "u":
        return Cache("u", cache.u, cache.ju)
    if kind in ("port", "cache"):
        p = choice["p"]
        z = cache.payload @ p
        jz = np.einsum("j,ijp->ip", p, cache.derivative)
        return Cache(kind, cache.u, cache.ju, z[:, None], jz)
    if kind == "full":
        return Cache("full", cache.u, cache.ju, cache.payload, cache.derivative)
    raise ValueError("raw is compared through the full reference after replay")


def score_candidate(cache, choice, ys, yq, work):
    candidate = cached_choice(cache, choice)
    dim = feature_dim(candidate.kind)
    if dim == 0:
        f = np.zeros((NS+NQ, 0))
    elif candidate.kind == "cache":
        f = decode_features(candidate.payload, candidate.derivative, choice["decoder"], work)
    else:
        f = candidate.payload
    beta = adapt_single(candidate.u[:NS], f[:NS], ys, work)
    pred = q_value(beta, candidate.u[NS:], f[NS:])
    return float(np.mean(.5*(pred-yq)**2)), candidate.nbytes(), beta.nbytes


def raw_archive(episode, scout):
    """Paid post-feedback raw replay archive, not persistent h/input-J shadows."""
    return dict(xb=episode["xb"].copy(), u=scout["cache"].u.copy(),
                ys=scout["ys"].copy(), yq=scout["yq"].copy())


def meta_objective(p, archives, theta_b, work):
    work.add("meta_objectives")
    loss = 0.0
    # At most one recomputed episode's full states is live at once.
    for data in archives:
        hb, _ = rnn_forward(theta_b, data["xb"], work, False)
        f = (hb @ unit(p))[:, None]
        beta = adapt_single(data["u"][:NS], f[:NS], data["ys"], work)
        pred = q_value(beta, data["u"][NS:], f[NS:])
        loss += float(np.mean(.5*(pred-data["yq"])**2))/len(archives)
    return loss


def fit_meta_port(initial, archives, theta_b, work):
    """Six fixed finite-difference outer steps; full support batch is reused."""
    p = unit(initial)
    for _ in range(META_STEPS):
        guard()
        grad = np.zeros(N)
        for j in range(N):
            plus, minus = p.copy(), p.copy()
            plus[j] += FD_EPS
            minus[j] -= FD_EPS
            grad[j] = (meta_objective(unit(plus), archives, theta_b, work)
                       - meta_objective(unit(minus), archives, theta_b, work))/(2*FD_EPS)
        p = unit(p-META_LR*grad)
    return p


def fit_cache_decoder(p, archives, theta_b, work):
    """Ridge on paid own-producer replay; accumulate stats, not full h histories."""
    width = 1+P
    count = 0
    sf, sy = np.zeros(width), np.zeros(N)
    sff, sfy = np.zeros((width, width)), np.zeros((width, N))
    working_peak = 0
    for data in archives:
        hb, tb = rnn_forward(theta_b, data["xb"], work, True)
        jp = rnn_vjp(theta_b, data["xb"], tb, np.broadcast_to(p, hb.shape), work)
        z = hb @ p
        f = np.concatenate((z[:, None], jp), axis=1)
        count += len(f)
        sf += f.sum(0)
        sy += hb.sum(0)
        sff += f.T @ f
        sfy += f.T @ hb
        work.add("dense_multiply_proxy", len(f)*(width*width+width*N))
        working_peak = max(working_peak, numeric_bytes(archives, hb, tb, jp, z, f, sf, sy, sff, sfy))
    mean = sf/count
    std = np.sqrt(np.maximum(np.diag(sff)/count-mean*mean, 0.0))+.03
    gram = np.zeros((width+1, width+1))
    gram[:-1, :-1] = (sff-count*np.outer(mean, mean))/(std[:, None]*std[None, :])
    gram[-1, -1] = count
    rhs = np.empty((width+1, N))
    rhs[:-1] = (sfy-mean[:, None]*sy[None, :])/std[:, None]
    rhs[-1] = sy
    penalty = np.eye(width+1)*(1e-3*count)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(gram+penalty, rhs)
    work.add("ridge_fit_calls")
    work.add("dense_multiply_proxy", (width+1)**3)
    decoder = dict(mean=mean, std=std, coef=coef)
    assert all(np.all(np.isfinite(v)) for v in decoder.values())
    working_peak = max(working_peak, numeric_bytes(archives, sf, sy, sff, sfy, gram, rhs, penalty, decoder))
    return decoder, int(working_peak)


def select_cut(validation, proposal):
    """Earliest legal cut first; otherwise a learned scalar port or paid raw replay."""
    means = {name: float(np.mean([v[name] for v in validation]))
             for name in ("full_loss", "u_loss", "port_loss", "u_bytes", "port_bytes", "raw_bytes")}
    if means["u_loss"] <= means["full_loss"]+TOL and means["u_bytes"] < means["raw_bytes"]:
        chosen = spec("u")
    elif (means["port_loss"] <= means["full_loss"]+TOL
          and means["port_bytes"] < means["raw_bytes"]):
        chosen = proposal
    else:
        chosen = spec("raw")
    return chosen, means


def form_all(env, seed):
    work = Work()
    extra = {name: Work() for name in ("gradient_port", "pca_port", "adaptation_port", "cache_ridge")}
    chosen = {name: [] for name in extra}
    decisions = []
    max_probation_bytes = 0
    max_archive_bytes = 0
    for context in range(4):
        guard()
        cg = np.zeros((N, N))
        cx = np.zeros((N, N))
        archives = []
        for episode_id in range(4):
            ep = env.new_episode(seed*10000+context*100+episode_id, context, SCOUT_LENGTH)
            scout = exact_scout(env.theta, env.a, ep, env, work)
            cg += np.outer(scout["g"], scout["g"])
            cx += scout["cache"].payload.T @ scout["cache"].payload/(NS+NQ)
            work.add("covariance_updates", 2)
            archives.append(raw_archive(ep, scout))
            held = numeric_bytes(scout["cache"].arrays(), scout["beta"], scout["ys"],
                                 scout["yq"], cg, cx, archives)
            max_probation_bytes = max(max_probation_bytes, scout["peak"]+held)
            max_archive_bytes = max(max_archive_bytes, numeric_bytes(archives))
            emit("scout_training", seed=seed, context=context, episode=episode_id,
                 query_loss=scout["loss"], post_learning_cotangent=scout["g"].tolist(),
                 full_cache_bytes=scout["cache"].nbytes(), retained_archive_bytes=numeric_bytes(archives),
                 declared_probation_peak_bytes=scout["peak"]+held,
                 producer_partial_gradient_norm=scout["native_gradient_norm"])
            # Full states/Jacobians do not remain in the formation archive.
            del scout, ep
        gp, pp = principal(cg), principal(cx)
        mp = fit_meta_port(gp, archives, env.theta[1], extra["adaptation_port"])
        decoder, ridge_peak = fit_cache_decoder(gp, archives, env.theta[1], extra["cache_ridge"])
        max_probation_bytes = max(max_probation_bytes, ridge_peak)
        emit("cache_decoder_fit", seed=seed, context=context,
             extra_labels=0, decoder_bytes=numeric_bytes(decoder),
             mean=decoder["mean"].tolist(), std=decoder["std"].tolist(),
             coef=decoder["coef"].tolist(),
             producer_gradient_scope="STOP_GRAD_J metadata features; z path differentiated")
        emit("port_meta_fit", seed=seed, context=context,
             method="MAML-Rep-style finite-difference port learner; post-learning gradient-covariance initializer",
             outer_steps=META_STEPS, finite_difference=FD_EPS,
             retained_raw_archive_bytes=numeric_bytes(archives), unit_port=mp.tolist())
        del archives  # Training data are dropped once this context's formation ends.
        ports = {"gradient_port": spec("port", gp), "pca_port": spec("port", pp),
                 "adaptation_port": spec("port", mp), "cache_ridge": spec("cache", gp, decoder)}
        validation = {name: [] for name in ports}
        for episode_id in range(4, 6):
            ep = env.new_episode(seed*10000+context*100+episode_id, context, SCOUT_LENGTH)
            scout = exact_scout(env.theta, env.a, ep, env, work)
            common = numeric_bytes(env.theta, env.a)
            u_loss, ubytes, ubeta = score_candidate(scout["cache"], spec("u"), scout["ys"], scout["yq"], work)
            # Raw B stores only the input stream plus the shared source version.
            rawbytes = numeric_bytes(scout["cache"].u, scout["cache"].ju, ep["xb"])
            raw_complete = common + rawbytes + (1+N)*8
            for name, proposal in ports.items():
                ploss, pbytes, pbeta = score_candidate(scout["cache"], proposal,
                                                       scout["ys"], scout["yq"], extra[name])
                row = dict(full_loss=scout["loss"], u_loss=u_loss, port_loss=ploss,
                           u_bytes=common+ubytes+ubeta,
                           port_bytes=common+pbytes+pbeta+numeric_bytes(proposal["p"], proposal["decoder"]),
                           raw_bytes=raw_complete)
                validation[name].append(row)
                emit("scout_validation", seed=seed, context=context, episode=episode_id,
                     method=name, **row)
            held = numeric_bytes(scout["cache"].arrays(), scout["beta"], scout["ys"], scout["yq"])
            max_probation_bytes = max(max_probation_bytes, scout["peak"]+held)
            del scout, ep
        for name, proposal in ports.items():
            choice, means = select_cut(validation[name], proposal)
            ordinary_admission = choice["kind"]
            diagnostic_override = False
            if name == "cache_ridge":
                # Predeclared COSTED information diagnostic, not an admitted low-byte method.
                # Exactly the gradient method's formed cuts; raw fallbacks remain raw.
                reference_choice = chosen["gradient_port"][-1]
                if reference_choice["kind"] == "port":
                    choice = spec("cache", reference_choice["p"], decoder)
                    diagnostic_override = True
                else:
                    choice = spec(reference_choice["kind"])
            chosen[name].append(choice)
            decision = dict(seed=seed, context=context, method=name,
                            cut=choice["kind"], unit_port=None if choice["p"] is None else choice["p"].tolist(),
                            decoder_bytes=numeric_bytes(choice["decoder"]),
                            ordinary_byte_admission=ordinary_admission,
                            costed_diagnostic_admission_override=diagnostic_override,
                            validation=means)
            decisions.append(decision)
            emit("formed_code", **decision)
    total = Work()
    total.merge(work)
    for item in extra.values():
        total.merge(item)
    return chosen, decisions, dict(work=total.v, shared_acquisition_work=work.v,
                                   method_specific_work={name: item.v for name, item in extra.items()},
                                   max_declared_probation_bytes=max_probation_bytes,
                                   max_retained_raw_meta_archive_bytes=max_archive_bytes,
                                   permanent_full_shadow=False)


def prepare_feedback(cache, theta_b, choice, work):
    """Reveal only what the typed cache physically permits."""
    if cache.kind == "u":
        return np.zeros((len(cache.u), 0)), None, 0
    if cache.kind in ("full", "port"):
        return cache.payload, None, 0
    if cache.kind == "full_projected":
        feature = (cache.payload @ choice["p"])[:, None]
        return feature, None, numeric_bytes(feature)
    if cache.kind == "cache":
        feature = decode_features(cache.payload, cache.derivative, choice["decoder"], work)
        return feature, None, numeric_bytes(feature)
    # Raw/hybrid replay: support B once; query B once plus one VJP after labels.
    hs, _ = rnn_forward(theta_b, cache.payload[:NS], work, False)
    hq, tq = rnn_forward(theta_b, cache.payload[NS:], work, True)
    feature = np.concatenate((hs, hq), axis=0)
    working = numeric_bytes(hs, hq, tq, feature)
    return feature, tq, working


def backward_cache(cache, theta_b, choice, q_feature_cotangent, query_tape, work):
    if cache.kind == "u":
        return np.zeros(P)
    if cache.kind == "port":
        return np.sum(q_feature_cotangent * cache.derivative[NS:], axis=0)
    if cache.kind == "full":
        return np.einsum("ij,ijp->p", q_feature_cotangent, cache.derivative[NS:])
    if cache.kind == "full_projected":
        # Same scalar native learner as the narrow port, full physical h/J storage.
        full_cot = q_feature_cotangent*choice["p"][None, :]
        return np.einsum("ij,ijp->p", full_cot, cache.derivative[NS:])
    if cache.kind == "cache":
        decoder = choice["decoder"]
        # Surrogate: metadata J columns are observed constants; no Hessian claim.
        derivative_z = decoder["coef"][0]/decoder["std"][0]
        scalar = q_feature_cotangent @ derivative_z
        return np.sum(scalar[:, None]*cache.derivative[NS:], axis=0)
    per_instance = rnn_vjp(theta_b, cache.payload[NS:], query_tape,
                           q_feature_cotangent, work)
    return np.sum(per_instance, axis=0)


def shared_model_arrays(theta, a, choices, betas):
    # Explicit arrays for model, immutable version, structural choice and native state.
    ids = np.array([{"u": 0, "port": 1, "full": 2, "raw": 3, "cache": 4,
                     "suffix_raw": 5, "full_projected": 6}[s["kind"]] for s in choices], dtype=np.int8)
    ports = [s["p"] for s in choices if s["p"] is not None]
    decoders = [s["decoder"] for s in choices if s["decoder"] is not None]
    return dict(theta=theta, a=a, ports=ports, decoders=decoders, structural_record=ids,
                native_beta=betas)


def post_update_query(env, updated, betas, choices, originals, seed, trial, contexts, length, work, name):
    """Actual next predictions after the one producer update, on new observed inputs."""
    qs, episodes = [], []
    working_peak = 0
    for i, (context, choice, beta) in enumerate(zip(contexts, choices, betas)):
        ep = env.fresh_query(originals[i], seed*100000+trial*10+i+500000000)
        episodes.append(ep)
        ha, _ = rnn_forward(updated[0], ep["xa"], work, False)
        u = ha @ env.a
        emit("current_capability_emitted", tag=name+"_post_update", context=context,
             count=NQ, current_u=u.tolist(), before_B_work=True,
             support_query_roles_available=False)
        if choice["kind"] == "u":
            feature = np.zeros((NQ, 0))
            hb = None
        else:
            need_tape = choice["kind"] == "cache"
            bx = ep["xb"][:, -LAST_M:] if choice["kind"] == "suffix_raw" else ep["xb"]
            hb, tb = rnn_forward(updated[1], bx, work, need_tape)
            if choice["kind"] == "cache":
                jp = rnn_vjp(updated[1], ep["xb"], tb, np.broadcast_to(choice["p"], hb.shape), work)
                z = hb @ choice["p"]
                feature = decode_features(z, jp, choice["decoder"], work)
                working_peak = max(working_peak, numeric_bytes(ep["xa"], ep["xb"], ha, u, hb, tb, jp, z, feature, qs))
            else:
                feature = ((hb @ choice["p"])[:, None]
                           if choice["kind"] in ("port", "full_projected") else hb)
        qs.append(q_value(beta, u, feature))
        working_peak = max(working_peak, numeric_bytes(ep["xa"], ep["xb"], ha, u, hb, feature, qs))
    prediction = compose(*qs)[0]
    emit("post_update_prediction_frozen", seed=seed, trial=trial, method=name,
         contexts=list(contexts), length=length, prediction=prediction.tolist())
    labels_child = [env.reveal(ep, np.arange(NQ)) for ep in episodes]
    labels = compose(*labels_child)[0]
    work.add("labels_used", NQ)
    return float(np.mean(.5*(prediction-labels)**2)), int(working_peak)


def production_pair(env, seed, trial, contexts, length, choices, name, deployed_choices):
    guard()
    work = Work()
    theta = [v.copy() for v in env.theta]  # Common checkpoint for every arm/trial.
    episodes = [env.new_episode(seed*100000+trial*10+i, c, length) for i, c in enumerate(contexts)]
    caches, working_peaks = [], []
    for ep, choice in zip(episodes, choices):
        cache, peak = produce_cache(theta, env.a, ep, choice, work, name)
        caches.append(cache)
        working_peaks.append(peak)
    betas0 = [np.zeros(1+feature_dim(c.kind)) for c in caches]
    shared0 = shared_model_arrays(theta, env.a, deployed_choices, betas0)
    waiting_cache = sum(c.nbytes() for c in caches)
    shared_bytes = numeric_bytes(shared0)
    waiting_bytes = waiting_cache + shared_bytes
    # No learner has received support/query roles or their targets during returns.
    feature_data, query_tapes, feedback_working = [], [], 0
    for cache, choice in zip(caches, choices):
        f, tape, working = prepare_feedback(cache, theta[1], choice, work)
        feature_data.append(f)
        query_tapes.append(tape)
        feedback_working += working
    support = np.arange(NS)
    query = np.arange(NS, NS+NQ)
    ys_child = [env.reveal(ep, support) for ep in episodes]
    ys = compose(*ys_child)[0]  # Only the whole-program label reaches adaptation.
    work.add("labels_used", NS)
    betas = adapt_pair([c.u[:NS] for c in caches], [f[:NS] for f in feature_data], ys, work)
    qs = [q_value(b, c.u[NS:], f[NS:]) for b, c, f in zip(betas, caches, feature_data)]
    prediction, dc1, dc2 = compose(*qs)
    emit("query_prediction_frozen", seed=seed, trial=trial, method=name,
         contexts=list(contexts), length=length, prediction=prediction.tolist(),
         cache_kinds=[c.kind for c in caches], waiting_numeric_bytes=waiting_bytes)
    yq_child = [env.reveal(ep, query) for ep in episodes]
    yq = compose(*yq_child)[0]  # First access to these query targets in this arm.
    work.add("labels_used", NQ)
    residual = prediction-yq
    loss = float(np.mean(.5*residual*residual))
    cot = residual/NQ
    lambda_q = [cot*dc1, cot*dc2]
    ga, gb = np.zeros(P), np.zeros(P)
    for b, cache, choice, qcot, tape in zip(betas, caches, choices, lambda_q, query_tapes):
        ga += np.sum(qcot[:, None]*cache.ju[NS:], axis=0)
        feature_cot = qcot[:, None]*(b[0]**2*b[1:])[None, :]
        gb += backward_cache(cache, theta[1], choice, feature_cot, tape, work)
    # Declared partial-SGD producer update: beta held fixed, no hypergradient.
    updated = [theta[0]-.01*ga, theta[1]-.01*gb]
    assert all(np.all(np.isfinite(x)) for x in updated)
    post_loss, post_working = post_update_query(env, updated, betas, choices, episodes, seed,
                                               trial, contexts, length, work, name)
    feedback_records = dict(features=feature_data, query_tapes=query_tapes,
                            support_labels=ys, query_labels=yq,
                            native_beta=betas, producer_grad=[ga, gb], updated=updated)
    feedback_total = numeric_bytes(shared0, [c.arrays() for c in caches], feedback_records)
    # Working peaks include the cumulative persistent cache already produced.
    prefix = 0
    production_peak = waiting_bytes
    for peak, cache in zip(working_peaks, caches):
        production_peak = max(production_peak, shared_bytes+prefix+peak)
        prefix += cache.nbytes()
    # Unique allocations are deduplicated across caches and feedback aliases.
    # Additional BPTT scratch is conservatively covered by feedback_working.
    peak_bytes = max(production_peak, feedback_total+feedback_working+post_working)
    result = dict(seed=seed, trial=trial, method=name, contexts=list(contexts), length=length,
                  loss=loss, post_producer_update_loss=post_loss,
                  prediction=prediction.tolist(), targets=yq.tolist(),
                  cache_kinds=[c.kind for c in caches], waiting_cache_bytes=waiting_cache,
                  shared_model_numeric_bytes=shared_bytes, waiting_numeric_bytes=waiting_bytes,
                  producer_working_peak_numeric_bytes=max(working_peaks),
                  feedback_working_numeric_bytes=feedback_working,
                  post_update_forward_working_numeric_bytes=post_working,
                  declared_peak_numeric_bytes=int(peak_bytes),
                  peak_scope="incomplete/conservative working-state proxy, NOT measured actual peak; waiting bytes exact for declared arrays",
                  work=work.v, producer_partial_gradient_norm=[float(np.linalg.norm(ga)), float(np.linalg.norm(gb))],
                  producer_update_norm=[float(np.linalg.norm(updated[i]-theta[i])) for i in range(2)],
                  beta=[b.tolist() for b in betas],
                  cache_reading_control="bounded ridge on (z,Jz) implemented; arbitrary/higher-order cache inference not implemented",
                  producer_gradient_scope="cache-ridge STOP_GRAD_J surrogate; suffix is exact for its truncated reset algorithm; others exact fixed-beta partial gradient",
                  encoder_native_gradient_scope="formed p is meta-only/fixed within production; no hidden dp cache")
    for delay in DELAYS:
        emit("production_result", **result, delay_events=delay,
             waiting_numeric_byte_time=waiting_bytes*delay,
             delay_simulation="same frozen numerical trace; no sleep or duplicate training")
    # No input/h/tape arrays survive in the result or production learner.
    return result, (ga.copy(), gb.copy())


def suffix_approximation_check(theta_b, seed):
    """Predeclared fading-memory diagnostic; never used in formation/features."""
    rng = np.random.default_rng(seed+9000000)
    x = rng.normal(size=(4, SCOUT_LENGTH, D))
    work = Work()
    full, _ = rnn_forward(theta_b, x, work, False)
    tail, _ = rnn_forward(theta_b, x[:, -LAST_M:], work, False)
    recurrent, _, _ = unpack(theta_b)
    lipschitz = float(np.linalg.norm(recurrent, 2))
    bound = min(2*np.sqrt(N), (lipschitz**LAST_M)*np.sqrt(N))
    observed = float(np.max(np.linalg.norm(full-tail, axis=1)))
    assert observed <= bound+1e-10
    emit("suffix_approximation_check", seed=seed, last_m=LAST_M,
         reset_state="zero at start of retained suffix", recurrent_spectral_norm=lipschitz,
         uniform_state_l2_bound=float(bound), observed_max_state_l2_error=observed,
         scope="initial fixed source version; no universal post-learning contraction claim",
         work=work.v)


def checks():
    rng = np.random.default_rng(9901)
    env = Environment(9902)
    x = rng.normal(size=(1, 5, D))
    work = Work()
    h, tape = rnn_forward(env.theta[0], x, work, True)
    jac = rnn_jacobian(env.theta[0], x, tape, work)
    eps = 1e-5
    fd = np.zeros((N, P))
    for j in range(P):
        plus, minus = env.theta[0].copy(), env.theta[0].copy()
        plus[j] += eps
        minus[j] -= eps
        hp, _ = rnn_forward(plus, x, work, False)
        hm, _ = rnn_forward(minus, x, work, False)
        fd[:, j] = (hp[0]-hm[0])/(2*eps)
    lam = rng.normal(size=(1, N))
    vjp = rnn_vjp(env.theta[0], x, tape, lam, work)
    jac_error = float(np.max(np.abs(jac[0]-fd)))
    vjp_error = float(np.max(np.abs(vjp[0]-lam[0]@fd)))
    feature = rng.normal(0, .2, (NS, N))
    u = rng.normal(0, .2, NS)
    y = rng.normal(0, .2, NS)
    beta = rng.normal(0, .15, 1+N)
    value = q_value(beta, u, feature)
    analytic = q_parameter_grad(beta, feature, (value-y)/NS)
    finite = np.zeros_like(beta)
    for j in range(len(beta)):
        plus, minus = beta.copy(), beta.copy()
        plus[j] += eps
        minus[j] -= eps
        finite[j] = (np.mean(.5*(q_value(plus,u,feature)-y)**2)
                     -np.mean(.5*(q_value(minus,u,feature)-y)**2))/(2*eps)
    beta_error = float(np.max(np.abs(analytic-finite)))
    assert max(jac_error, vjp_error, beta_error) < 1e-6
    w = .5  # Both hard-port/full native gradients were -1 at w0=0.
    full, narrow = w+w*w*3, w
    full_loss, narrow_loss = .5*(full-1.25)**2, .5*(narrow-1.25)**2
    assert full_loss == 0 and narrow_loss == .28125
    emit("verified_checks", rnn_jacobian_error=jac_error, rnn_vjp_error=vjp_error,
         native_beta_gradient_error=beta_error,
         quadratic_counterexample=dict(full_loss=full_loss, hard_port_loss=narrow_loss,
                                        current_output_and_current_gradients_match=True),
         work=work.v)


def main():
    emit("serialization_smoke", seeds=list(SEEDS), n=N, d=D, p_per_block=P,
         scout_length=SCOUT_LENGTH, production_lengths=list(PRODUCTION_LENGTHS),
         native_updates=K, native_lr=LR, delays=list(DELAYS), tolerance=TOL,
         suffix_replay_last_m=LAST_M,
         status="exploratory acquired-frontier pilot; no foundational claim",
         isolation="instruction-only environment/cache interface; not a technical secret boundary",
         cache_control="bounded ridge reader of (z,Jz); arbitrary cache inference still not implemented",
         accounting="NumPy allocation and operation proxies; excludes Python/runtime/RSS/energy")
    checks()
    all_results, all_formations = [], []
    for seed in SEEDS:
        guard()
        env = Environment(seed)
        suffix_approximation_check(env.theta[1], seed)
        formed, decisions, formation = form_all(env, seed)
        # Fixed derangement swaps both nominal context pairs. No hidden b is read.
        permutation = [2, 3, 0, 1]
        reassigned = [formed["gradient_port"][j] for j in permutation]
        # Per-arm attribution, not the aggregate multi-method experiment total.
        dependencies = {
            "current_only": None, "full_return": None, "raw_replay": None,
            "suffix_replay_m16": None,
            "gradient_port": ("gradient_port",), "full_projected": ("gradient_port",),
            "reassigned_relation": ("gradient_port",), "hybrid_replay": ("gradient_port",),
            "hybrid_suffix_m16": ("gradient_port",),
            "pca_port": ("pca_port",), "adaptation_port": ("adaptation_port",),
            "cache_ridge": ("gradient_port", "cache_ridge")}
        attributed = {}
        for method, needs in dependencies.items():
            counts = Work()
            if needs is not None:
                for key, value in formation["shared_acquisition_work"].items():
                    counts.add(key, value)
                for dependency in needs:
                    for key, value in formation["method_specific_work"][dependency].items():
                        counts.add(key, value)
            attributed[method] = counts.v
        formation["per_arm_acquisition_work"] = attributed
        formation["acquisition_attribution_scope"] = (
            "fixed controls use no formation; formed diagnostics pay their reused formation dependencies; "
            "shared scout proxy is conservative, not an optimal standalone implementation")
        all_formations.append(dict(seed=seed, decisions=decisions, **formation))
        emit("formation_summary", seed=seed, **formation)
        for pair_id, contexts in enumerate(itertools.combinations(range(4), 2)):
            for length in PRODUCTION_LENGTHS:
                # Same seeds/inputs/labels/checkpoint for every arm of this trial.
                trial = pair_id*2 + int(length == PRODUCTION_LENGTHS[1])
                methods = {
                    "current_only": [spec("u") for _ in range(4)],
                    "full_return": [spec("full") for _ in range(4)],
                    "raw_replay": [spec("raw") for _ in range(4)],
                    "suffix_replay_m16": [spec("suffix_raw") for _ in range(4)],
                    "gradient_port": formed["gradient_port"],
                    "full_projected": [spec("full_projected", c["p"])
                                       if c["kind"] == "port" else spec(c["kind"])
                                       for c in formed["gradient_port"]],
                    "pca_port": formed["pca_port"],
                    "adaptation_port": formed["adaptation_port"],
                    "cache_ridge": formed["cache_ridge"],
                    "reassigned_relation": reassigned,
                    "hybrid_replay": [spec("u") if formed["gradient_port"][j]["kind"] == "u"
                                      else spec("raw") for j in range(4)],
                    "hybrid_suffix_m16": [spec("u") if formed["gradient_port"][j]["kind"] == "u"
                                          else spec("suffix_raw") for j in range(4)],
                }
                trial_results, trial_gradients = {}, {}
                for name, deployed in methods.items():
                    choices = [deployed[j] for j in contexts]
                    result, gradients = production_pair(env, seed, trial, contexts, length, choices, name, deployed)
                    all_results.append(result)
                    trial_results[name], trial_gradients[name] = result, gradients
                narrow, projected = trial_results["gradient_port"], trial_results["full_projected"]
                prediction_error = float(np.max(np.abs(np.array(narrow["prediction"])-projected["prediction"])))
                beta_error = max(float(np.max(np.abs(np.array(a)-b)))
                                 for a, b in zip(narrow["beta"], projected["beta"]))
                gradient_error = max(float(np.max(np.abs(a-b)))
                                     for a, b in zip(trial_gradients["gradient_port"], trial_gradients["full_projected"]))
                post_loss_error = abs(narrow["post_producer_update_loss"]-projected["post_producer_update_loss"])
                assert max(prediction_error, beta_error, gradient_error, post_loss_error) < 1e-9
                emit("physical_retirement_control_check", seed=seed, trial=trial,
                     contexts=list(contexts), length=length,
                     actual_projected_contexts=sum(methods["full_projected"][j]["kind"] == "full_projected"
                                                   for j in contexts),
                     prediction_error=prediction_error, beta_error=beta_error,
                     producer_gradient_error=gradient_error, post_update_loss_error=post_loss_error,
                     waiting_bytes_narrow=narrow["waiting_numeric_bytes"],
                     waiting_bytes_projected=projected["waiting_numeric_bytes"])
                raw, tail = trial_results["raw_replay"], trial_results["suffix_replay_m16"]
                emit("suffix_vs_raw_external_check", seed=seed, trial=trial, length=length,
                     last_m=LAST_M,
                     max_prediction_difference=float(np.max(np.abs(np.array(raw["prediction"])-tail["prediction"]))),
                     query_loss_gap=tail["loss"]-raw["loss"],
                     post_update_loss_gap=tail["post_producer_update_loss"]-raw["post_producer_update_loss"],
                     waiting_bytes_raw=raw["waiting_numeric_bytes"],
                     waiting_bytes_suffix=tail["waiting_numeric_bytes"],
                     outcome_not_used_for_tuning=True)
        emit("environment_work", seed=seed, work=env.teacher_work.v,
             note="environment oracle generation is separate from learner observations")
    aggregate = {}
    for name in sorted({r["method"] for r in all_results}):
        rows = [r for r in all_results if r["method"] == name]
        aggregate[name] = dict(count=len(rows), mean_loss=float(np.mean([r["loss"] for r in rows])),
                               mean_waiting_numeric_bytes=float(np.mean([r["waiting_numeric_bytes"] for r in rows])),
                               max_declared_peak_numeric_bytes=max(r["declared_peak_numeric_bytes"] for r in rows))
    summary = dict(status="completed", seconds=time.monotonic()-START,
                   config=dict(seeds=list(SEEDS), n=N, d=D, p_per_block=P, native_updates=K,
                               native_lr=LR, scout_length=SCOUT_LENGTH,
                               production_lengths=list(PRODUCTION_LENGTHS), delays=list(DELAYS),
                               suffix_replay_last_m=LAST_M),
                   formation=all_formations, results=all_results, aggregate=aggregate,
                   important_limits=["bounded ridge cache reader only; no arbitrary-cache-reading/higher-order learner",
                                     "fixed small MAML-Rep-style port learner, not exact OML reproduction",
                                     "partial producer SGD holds adapted beta fixed",
                                     "cache-ridge producer metadata gradients use explicit STOP_GRAD_J surrogate",
                                     "waiting declared arrays counted exactly; working peak is incomplete/conservative proxy",
                                     "NumPy-array and core-operation proxies, not full system memory/FLOPs",
                                     "constructed synthetic acquisition evidence is not novelty or breakthrough"])
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    emit("completed", seconds=summary["seconds"], production_rows=len(all_results), aggregate=aggregate)


if __name__ == "__main__":
    START = time.monotonic()
    RAW = (OUT/"raw.jsonl").open("x")
    try:
        main()
    except Exception as exc:
        emit("failed", error_type=type(exc).__name__, message=str(exc),
             seconds=time.monotonic()-START)
        raise
