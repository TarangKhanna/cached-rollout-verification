"""Deterministic worked examples of a known conditional-proposal pitfall.

No training experiment or released SPEC-RL code. Exact rational dense kernels are
checked against separate finite formulas/orbits. A literal softmax REINFORCE step
provides an additional floating-point check. Run: python code/worked_examples.py
"""
from fractions import Fraction as F
from itertools import product
import math
import numpy as np


def kernel(q, p, fallback='residual'):
    """Rows indexed by cached token, columns by verified output."""
    z = sum(abs(x-y) for x, y in zip(p, q))/2
    a = [min(F(1), x/y) for x, y in zip(p, q)]
    if fallback == 'residual':
        replacement = [max(F(0), x-y)/z for x, y in zip(p, q)] if z else list(p)
    elif fallback == 'ordinary':
        replacement = p
    else:
        raise ValueError(fallback)
    return [[a[y]*int(x == y)+(1-a[y])*replacement[x]
             for x in range(len(q))] for y in range(len(q))]


def reinforce_dense(n):
    """Enumerate Y and complete output distribution for a single score step."""
    assert n >= 2 and n % 2 == 0
    q = [F(1, n)]*n
    reward = [int(j < n//2) for j in range(n)]
    gap = F(0)
    for y in range(n):
        # eta=2 log(2), baseline=1/2: the common logit offset cancels.
        weights = [F(1)]*n
        weights[y] = F(2)**(2*reward[y]-1)
        p = [w/sum(weights) for w in weights]
        k = kernel(q, p)
        fresh = sum(p[x]*reward[x] for x in range(n))
        reused = sum(k[y][x]*reward[x] for x in range(n))
        gap += q[y]*(reused-fresh)
        assert all(sum(q[j]*k[j][x] for j in range(n)) == p[x] for x in range(n))
        assert sum(abs(p[j]-q[j]) for j in range(n))/2 <= F(1,n)
        # Independent literal gradient/softmax evaluation, not the weight formula.
        one = np.eye(n)[y]
        logits = 2*math.log(2)*(reward[y]-.5)*(one-1/n)
        pf = np.exp(logits-logits.max()); pf /= pf.sum()
        np.testing.assert_allclose(pf, [float(v) for v in p], atol=2e-15, rtol=0)
    return gap


def reinforce_formula(n):
    """Average of the rewarded/unrewarded conditional discrepancies."""
    return F(n,4)*(F(1,n+1)-F(1,2*n-1))


def grpo_target(group, n):
    """Literal integer log2 weight increments from detached G=3 scores."""
    count = sum(y < n//2 for y in group)
    weights = [F(1)]*n
    if count in (1,2):
        for y in group:
            exponent = (2 if y < n//2 else -1) if count == 1 else (1 if y < n//2 else -2)
            weights[y] *= F(2)**exponent
    return [w/sum(weights) for w in weights]


def grpo_dense(n, fallback):
    """All ordered training groups, including collisions; complete token kernels."""
    q = [F(1,n)]*n
    gap = F(0)
    for group in product(range(n), repeat=3):
        p = grpo_target(group,n)
        k = kernel(q,p,fallback)
        fresh = sum(p[:n//2])
        reused = sum(sum(k[y][:n//2]) for y in group)/3
        gap += (reused-fresh)/n**3
        if fallback == 'residual':
            assert all(sum(q[j]*k[j][x] for j in range(n)) == p[x] for x in range(n))
    return gap


def grpo_orbits(n, fallback):
    """Separate count/collision calculation without enumerating verifier rows."""
    n = F(n)
    # mass, rewarded count, denominator increment, negative weight, reward increment
    types = [(F(3,8)*(1-2/n), 1, F(2), F(1,2), F(3)),
             (F(3,8)*2/n, 1, F(9,4), F(1,4), F(3)),
             (F(3,8)*(1-2/n), 2, F(5,4), F(1,4), F(2)),
             (F(3,8)*2/n, 2, F(9,4), F(1,4), F(3))]
    gap = F(0)
    for mass,count,increment,bad_weight,reward_increment in types:
        denominator = n+increment
        fresh = (n/2+reward_increment)/denominator
        rejection = 1-n*bad_weight/denominator
        replacement_reward = 1 if fallback == 'residual' else fresh
        mean = F(count,3)+F(3-count,3)*rejection*replacement_reward
        gap += mass*(mean-fresh)
    # Unanimous reward groups have opposite discrepancies of equal probability.
    return gap


def run_checks():
    for n in (2,4,8,16):
        value = reinforce_dense(n)
        assert value == reinforce_formula(n)
        print(f'REINFORCE N={n}: exact gap={value} ({float(value):.10f})')
    # Exact finite kernels versus an independent orbit route, not simulated rollouts.
    for n in (4,6):
        for fallback in ('residual','ordinary'):
            assert grpo_dense(n,fallback) == grpo_orbits(n,fallback)
    large = 10**8
    assert abs(float(reinforce_formula(large))-1/8) < 1e-7
    assert abs(float(grpo_orbits(large,'residual'))-7/32) < 1e-7
    assert abs(float(grpo_orbits(large,'ordinary'))-7/64) < 1e-7
    # Limits from direct reward-count averaging (mixed counts each have mass 3/8).
    residual_mean = F(1,8)+F(3,8)*(F(2,3)+F(11,12))
    ordinary_mean = F(1,8)+F(3,8)*(F(1,2)+F(19,24))
    assert residual_mean-F(1,2) == F(7,32)
    assert ordinary_mean-F(1,2) == F(7,64)
    print('REINFORCE exact-residual limiting discrepancy = 1/8 = 0.125')
    print('G=3 exact-residual limiting discrepancy = 7/32 = 0.21875')
    print('G=3 ordinary-target limiting discrepancy = 7/64 = 0.109375')
    print('PASS dense rational kernels, collision orbits, literal REINFORCE update and limit algebra')
    print('These are worked one-token examples, not measurements of released training.')


if __name__ == '__main__':
    run_checks()
