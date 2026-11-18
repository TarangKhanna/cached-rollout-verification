#!/usr/bin/env python3
"""Independent control-variate batch-influence test, NumPy only.

Does not import the supplied research package. Proposal q, GRPO score,
and exact 1st-rejection sequence kernel come from independent_finite_reference.
This is exploratory, not preregistered or neural policy training.
"""
import math
import numpy as np
from independent_finite_reference import (
    TH0, TRIPLES, REWARD, policy, grpo_gradient, next_kernel,
)
q, oldscore, qtok = policy(TH0)
old_grads = grpo_gradient(oldscore, TRIPLES)
qgroup = np.prod(q[TRIPLES],axis=1)
qgroup /= qgroup.sum()


def projected_gradient_at(theta):
    p,score,ptok=policy(theta)
    group_grads=grpo_gradient(score,TRIPLES)
    fweights=np.prod(p[TRIPLES],axis=1)
    return fweights@group_grads

base_g = projected_gradient_at(TH0)
projection = base_g / np.linalg.norm(base_g)


def discrepancy(group, theta):
    p,score,ptok=policy(theta)
    K=next_kernel(qtok,ptok)
    assert np.max(np.abs(q@K-p))<1e-11
    gs=grpo_gradient(score,TRIPLES)
    fresh=np.prod(p[TRIPLES],axis=1)@gs
    included=np.prod(K[np.asarray(group)[:,None],TRIPLES.T].T,axis=1)@gs
    return float(projection@(included-fresh))


def run():
    rng=np.random.default_rng(20261117)
    for eta in (0.05,0.20):
        for B in (1,2,4,8,16,32,64):
            vals=[]
            for _ in range(300):
                ix=rng.choice(len(TRIPLES),size=B,p=qgroup)
                s=TRIPLES[ix[0]]
                minus=TH0 + (eta/B)*old_grads[ix[1:]].sum(axis=0)
                full=minus + (eta/B)*old_grads[ix[0]]
                # E[discrepancy(S1,minus)|others]=0 exactly:
                # S1 independent of minus and q*K_minus=p_minus.
                vals.append(discrepancy(s,full)-discrepancy(s,minus))
            v=np.asarray(vals)
            print(f'eta={eta:.2f} B={B:2d} included_minus_fresh={v.mean():+.8f} '
                  f'+/- {1.96*v.std(ddof=1)/math.sqrt(len(v)):.8f} (95% approx) '
                  f'B*delta={B*v.mean():+.7f}',flush=True)

if __name__=='__main__':
    run()
