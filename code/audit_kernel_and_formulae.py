#!/usr/bin/env python3
"""Additional independent kernel / proof invariants and algebraic checks.
Assumption: finite alphabet; independent verifier coins. This is a diagnostic,
not a proof and not a production SPEC-RL implementation.
"""
import numpy as np
from math import exp, log

def tv(p,q):return np.abs(p-q).sum()/2

def ker(p,q):
    if np.allclose(p,q,atol=1e-14,rtol=0):return np.eye(len(p))
    a=np.minimum(1,p/q)
    g=np.maximum(p-q,0)
    z=g.sum()
    assert z>0
    return np.diag(a)+(1-a)[:,None]*g[None,:]/z

rng=np.random.default_rng(20261117)
count=0
for dim in (3,4,8,32):
 for trial in range(500):
    q=rng.dirichlet(np.ones(dim))
    p=rng.dirichlet(np.ones(dim))
    pp=rng.dirichlet(np.ones(dim))
    if trial%7==0:
        p[trial%dim]=0;p/=p.sum()
    if trial%11==0:
        pp[(trial+1)%dim]=0;pp/=pp.sum()
    K=ker(p,q);Kp=ker(pp,q)
    assert np.max(np.abs(K.sum(axis=1)-1))<1e-12
    assert np.max(np.abs(q@K-p))<2e-12
    for i in range(dim):
        assert tv(K[i],Kp[i])<=min(1,tv(p,pp)/q[i])+2e-11
    h=rng.dirichlet(np.ones(dim));d=h-q;a=np.minimum(1,p/q)
    g=np.maximum(p-q,0);r=g/g.sum()
    signed=(h@K-p)-(d*a-np.dot(d,a)*r)
    assert np.max(np.abs(signed))<2e-12
    assert tv(h@K,p)<=tv(h,q)+1e-12
    count+=1

for epsilon in (.1,.01,.001):
    q=np.array([epsilon,(1-epsilon)/2,(1-epsilon)/2])
    p=np.array([epsilon/2,1/2,(1-epsilon)/2])
    pp=np.array([epsilon/2,(1-epsilon)/2,1/2])
    K=ker(p,q);Kp=ker(pp,q)
    ratio=np.max(np.abs(np.log(p/pp)))
    assert abs(tv(K[0],Kp[0])-.5)<1e-12
    assert abs(tv(p,pp)/q[0]-.5)<1e-12
    print(f'Sharp example epsilon={epsilon}: target max-log-ratio {ratio:.8f}; row TV {tv(K[0],Kp[0]):.8f}')

for theta in (.001,.01,.1,log(4)):
    t=exp(-theta/2)
    B=(3-2*t-t*t)/8
    BE=(1-t)/6
    print(f'theta={theta:.7f} B={B:.10f} BE={BE:.10f} '
          f'linear approx {theta/4:.10f},{theta/12:.10f}')
    if theta==log(4):
        assert abs(B-7/32)<1e-12 and abs(BE-1/12)<1e-12
print(f'PASS {count} finite randomized kernel checks; algebraic formula limits and sharp example')
