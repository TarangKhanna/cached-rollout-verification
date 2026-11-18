"""Independent NumPy finite qualification. No model, optimizer library or GPU.

All verifier/fresh coins are integrated; randomness is independent old-policy
update groups. This is not a source-faithful GRPO/Adam implementation.
"""
import itertools
import json
import math
import resource
import sys
import time
import numpy as np

Y = np.array(list(itertools.product((0, 1), repeat=3)))
IDX = np.array(list(itertools.product(range(8), repeat=3)))
R = (Y.sum(1) >= 2).astype(float)
F = np.empty((8, 3, 4))
for i, y in enumerate(Y):
    for t in range(3):
        F[i, t] = (1, 0 if t == 0 else 2*y[t-1]-1,
                   t-1, (2*y[:t].sum()-t)/3)
PREFIX = [np.all(Y[:, None, :t] == Y[None, :, :t], axis=2) for t in range(3)]
DIFF = [Y[:, None, t] != Y[None, :, t] for t in range(3)]
TH = np.array([-.2, .35, .15, .25])
rr = R[IDX]
sd = rr.std(1, keepdims=True)
ADV = np.divide(rr-rr.mean(1, keepdims=True), sd,
                out=np.zeros_like(rr), where=sd > 0)
MIX = (sd[:, 0] > 0).astype(float)


def actor(theta):
    z = F @ theta
    one = 1/(1+np.exp(-z))
    tok = np.where(Y, one, 1-one)
    return tok.prod(1), ((Y-one)[:, :, None]*F).sum(1), tok


def kernels(oldtok, newtok):
    """Sum mutually exclusive first-rejection events, including same-bit fallback."""
    accept = np.minimum(1, newtok/oldtok)
    survive = np.ones(8)
    exact, fallback = np.zeros((8, 8)), np.zeros((8, 8))
    regen = np.zeros(8)
    for t in range(3):
        mass = survive*(1-accept[:, t])
        suffix = newtok[:, t+1:].prod(1)
        exact += mass[:, None]*PREFIX[t]*DIFF[t]*suffix[None, :]
        fallback += mass[:, None]*PREFIX[t]*(newtok[:, t]*suffix)[None, :]
        regen += mass*(3-t)
        survive *= accept[:, t]
    exact += np.diag(survive)
    fallback += np.diag(survive)
    return exact, fallback, regen


def group_weights(rows):
    return rows[0, IDX[:, 0]]*rows[1, IDX[:, 1]]*rows[2, IDX[:, 2]]


def new_reward(theta):
    z = np.einsum('itd,nd->nit', F, theta)
    p1 = 1/(1+np.exp(-z))
    return np.where(Y[None], p1, 1-p1).prod(2) @ R


q, oldscore, qt = actor(TH)
QG = q[IDX].prod(1)
OLDG = (ADV[:, :, None]*oldscore[IDX]).mean(1)
BASE = QG @ OLDG
PROJ = BASE/np.linalg.norm(BASE)


def observe(s, theta):
    p, score, tok = actor(theta)
    k, ell, regen = kernels(qt, tok)
    assert np.max(np.abs(k.sum(1)-1)) < 1e-11
    assert np.max(np.abs(ell.sum(1)-1)) < 1e-11
    exact_err = np.max(np.abs(q @ k-p))
    assert exact_err < 1e-11
    grads = (ADV[:, :, None]*score[IDX]).mean(1)
    reserve_law = q @ ell
    w = np.stack([p[IDX].prod(1), (q@k)[IDX].prod(1),
                  group_weights(k[s]), group_weights(ell[s]),
                  reserve_law[IDX].prod(1)])
    means = w @ grads
    yields = w @ MIX
    filtered_numer = (w*MIX) @ grads
    fresh_cond = filtered_numer[0]/yields[0]
    # At a fixed independent theta E_q[Cn - Csurvival*fresh_cond] = 0.
    matched = filtered_numer[2]-yields[2]*fresh_cond
    g2 = (w[0, :, None]*grads**2).sum(0)-means[0]**2
    ratios = tok/qt
    seqrat = p/q
    clip_token = float((q[:, None]*((ratios < .8)|(ratios > 1.2))).sum()/3)
    clip_seq = float(q @ ((seqrat < .8)|(seqrat > 1.2)))
    klqp = float(q @ np.log(q/p))
    klpq = float(p @ np.log(p/q))
    # k1=-log ratio, k3=ratio-1-log ratio, sampled with old q at reached prefixes.
    # Prefix exposure under q is represented by full-q sequence weighting.
    lr = np.log(ratios)
    k1 = float((q[:, None]*(-lr)).sum()/3)
    k3 = float((q[:, None]*(ratios-1-lr)).sum()/3)
    after = new_reward(theta[None]+.25*grads)
    return dict(means=means, yields=yields, matched=matched,
                reward2=w@after, fresh_var=g2, kl=[klqp, klpq], k1=k1, k3=k3,
                clips=[clip_token, clip_seq],
                regen=[9., 3*float(q@regen), float(regen[s].sum()),
                       float(regen[s].sum()), 3*float(q@regen)],
                exact_error=exact_err, fallback_error=float(np.max(np.abs(q@ell-p))))


def summary(v):
    a = np.asarray(v)
    m = a.mean(0)
    se = a.std(0, ddof=1)/math.sqrt(len(a))
    return dict(mean=m.tolist(), se=se.tolist(),
                ci95=np.stack([m-1.96*se, m+1.96*se], axis=-1).tolist())


def cell(eta, b, seed, reps):
    rng = np.random.default_rng(seed)
    allobs, cv, cvl, matched_cv, raw, rawl = [], [], [], [], [], []
    for _ in range(reps):
        ids = rng.choice(512, size=b, p=QG/QG.sum())
        minus = TH+(eta/b)*OLDG[ids[1:]].sum(0)
        full = minus+(eta/b)*OLDG[ids[0]]
        s = IDX[ids[0]]
        o, leave = observe(s, full), observe(s, minus)
        d = o['means'][2]-o['means'][0]
        dl = o['means'][3]-o['means'][4]
        cv.append(d-(leave['means'][2]-leave['means'][0]))
        cvl.append(dl-(leave['means'][3]-leave['means'][4]))
        matched_cv.append(o['matched']-leave['matched'])
        raw.append(d); rawl.append(dl)
        allobs.append(o)
    c = np.array(cv)
    m, se = c.mean(0), c.std(0, ddof=1)/math.sqrt(reps)
    fresh = np.mean([o['means'][0] for o in allobs], axis=0)
    fresh_norm = np.linalg.norm(fresh)
    # Conservative norm bound using a componentwise approximate simultaneous box.
    upper = np.linalg.norm(np.abs(m)+2.734*se)
    fy = np.mean([o['yields'][2] for o in allobs])
    filtered = np.array(matched_cv)/fy
    noise = math.sqrt(np.mean([o['fresh_var'].sum() for o in allobs])/b)
    out = dict(eta=eta, B=b, seed=seed, independent_updates=reps,
               gradient_cv=summary(cv), projected_gradient_cv=summary(c@PROJ),
               raw_gradient_difference=summary(raw),
               included_fallback_minus_reserve_cv=summary(cvl),
               raw_fallback_inclusion_difference=summary(rawl),
               matched_filtered_gradient_cv=summary(filtered),
               bias_norm=float(np.linalg.norm(m)), fresh_gradient_norm=float(fresh_norm),
               approximate_simultaneous_bias_norm_upper95=float(upper),
               relative_upper95=float(upper/fresh_norm),
               fresh_B_group_stochastic_gradient_rms=float(noise),
               arms_mean_gradient=summary([o['means'] for o in allobs]),
               arms_yield=summary([o['yields'] for o in allobs]),
               toy_second_single_group_step_reward=summary([o['reward2'] for o in allobs]),
               regenerated_suffix_tokens_per_attempt=summary([o['regen'] for o in allobs]),
               kl=summary([o['kl'] for o in allobs]),
               old_token_k1_k3=summary([[o['k1'], o['k3']] for o in allobs]),
               old_exposure_token_sequence_clip_fraction=summary([o['clips'] for o in allobs]),
               max_reserve_exact_error=max(o['exact_error'] for o in allobs),
               mean_reserve_fallback_error=float(np.mean([o['fallback_error'] for o in allobs])),
               included_zero_yield_fraction=float(np.mean([o['yields'][2]<1e-14 for o in allobs])),
               material_score_gate=bool(upper >= .05*fresh_norm and np.linalg.norm(m)>noise))
    return out


def checks():
    # Independent finite-difference score check, not the reference implementation.
    for d in range(4):
        v = np.zeros(4); v[d] = 1e-6
        deriv = (np.log(actor(TH+v)[0])-np.log(actor(TH-v)[0]))/(2e-6)
        assert np.max(np.abs(deriv-oldscore[:, d])) < 1e-8
    # Numeric tree normalization/exact-reserve checks at fixed exogenous actors.
    for t in [TH, TH+np.array([.1, -.2, .3, -.1])]:
        p, _, tok = actor(t)
        k, ell, _ = kernels(qt, tok)
        assert np.max(np.abs(q@k-p)) < 1e-12
        assert np.max(np.abs(k.sum(1)-1)) < 1e-12
        assert np.max(np.abs(ell.sum(1)-1)) < 1e-12
    # Exact all-512-old-group zero-step unconditional identity.
    zero = sum(QG[i]*(observe(s, TH)['means'][2]-BASE) for i,s in enumerate(IDX))
    assert np.linalg.norm(zero) < 1e-12
    # Exact positive control without reference imports.
    gap = np.zeros(4); fallback = 0.
    for i,s in enumerate(IDX):
        o = observe(s, TH+.2*OLDG[i])
        gap += QG[i]*(o['means'][2]-o['means'][0])
        fallback += QG[i]*o['fallback_error']
    assert np.linalg.norm(gap)>1e-3 and fallback>1e-8
    return dict(score_finite_difference='PASS', zero_step_unconditional=zero.tolist(),
                exact_B1_eta02_gap_norm=float(np.linalg.norm(gap)),
                reserve_fallback_error=float(fallback))


if __name__ == '__main__':
    start = time.monotonic()
    print(json.dumps(dict(type='controls', results=checks())), flush=True)
    for j, (eta, b) in enumerate(itertools.product([0., .05, .2, .8], [1,8,32,128])):
        o = cell(eta, b, 2026111701+j, 1024)
        print(json.dumps(dict(type='cell', results=o)), flush=True)
    print(json.dumps(dict(type='resources', wall_seconds=time.monotonic()-start,
                         process=resource.getrusage(resource.RUSAGE_SELF)[:3],
                         python=sys.version, numpy=np.__version__)), flush=True)
