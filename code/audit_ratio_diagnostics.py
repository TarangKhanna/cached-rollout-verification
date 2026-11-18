#!/usr/bin/env python3
"""Independently recompute weighted cached-token and whole-sequence clip fractions."""
import numpy as np
from independent_finite_reference import TH0,TRIPLES,REWARD,PHI,SEQS,policy,grpo_gradient
q,s,qtok=policy(TH0)
q1=1/(1+np.exp(-np.einsum('itd,d->it',PHI,TH0)))
oldG=grpo_gradient(s,TRIPLES)
for eta in (0.025,0.05,0.2,0.8):
    tseq=ttok=0.; total=0.
    for ix,S in enumerate(TRIPLES):
        w=np.prod(q[S])
        p,_,_=policy(TH0+eta*oldG[ix])
        p1=1/(1+np.exp(-np.einsum('itd,d->it',PHI,TH0+eta*oldG[ix])))
        seq_ratio=p[S]/q[S]
        token_ratio=np.where(SEQS[S]==1,p1[S]/q1[S],(1-p1[S])/(1-q1[S]))
        tseq+=w*np.mean((seq_ratio<.8)|(seq_ratio>1.2))
        ttok+=w*np.mean((token_ratio<.8)|(token_ratio>1.2))
        total+=w
    print(f'eta={eta:.3f} response clip fraction={tseq:.8f}; token clip fraction={ttok:.8f}; total weight={total:.10f}')
