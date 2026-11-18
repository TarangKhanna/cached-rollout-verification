#!/usr/bin/env python3
"""Independent derivation-oriented finite check of shared-policy/speculative RL.

No imports from the supplied SPEC-RL package. Python/NumPy only.
Exact T=3 group and two-optimizer-step enumeration; independent first-rejection
kernel via first-divergence formula. Does NOT claim LLM applicability.
"""
import itertools
import math
import numpy as np

T=3
SEQS=np.array(list(itertools.product([0,1], repeat=T)), dtype=np.int8)
M=len(SEQS)
TH0=np.array([-0.2,0.35,0.15,0.25])
REWARD=(SEQS.sum(axis=1)>=2).astype(float)
PHI=np.zeros((M,T,4))
for i,y in enumerate(SEQS):
    for t in range(T):
        PHI[i,t]=[1., 2*y[t-1]-1 if t else 0., t-1., (2*sum(y[:t])-t)/T]

TRIPLES=np.array(list(itertools.product(range(M),repeat=3)))

def policy(theta):
    p1=1/(1+np.exp(-np.einsum('itd,d->it',PHI,theta)))
    tok=np.where(SEQS==1,p1,1-p1)
    p=tok.prod(axis=1)
    s=np.einsum('it,itd->id', SEQS-p1, PHI)
    return p,s,tok

def grpo_gradient(score, group):
    r=REWARD[group]
    mean=r.mean(axis=1,keepdims=True)
    sd=np.sqrt(((r-mean)**2).mean(axis=1,keepdims=True))
    a=np.divide(r-mean,sd,out=np.zeros_like(r),where=sd>1e-12)
    return np.einsum('ni,nid->nd',a,score[group])/3

def next_kernel(q_tok,p_tok):
    """Independent first-divergence tree: residual at binary rejection flips bit."""
    K=np.zeros((M,M))
    for i,y in enumerate(SEQS):
        survives=1.
        for t in range(T):
            a=min(1., p_tok[i,t]/q_tok[i,t])
            reject=survives*(1-a)
            if reject>0:
                for j,x in enumerate(SEQS):
                    if np.array_equal(x[:t],y[:t]) and x[t]!=y[t]:
                        K[i,j]+=reject * float(np.prod(p_tok[j,t+1:]))
            survives*=a
        K[i,i]+=survives
    return K

def reward_of_thetas(thetas):
    logits=np.einsum('itd,nd->nit',PHI,thetas)
    p1=1/(1+np.exp(-logits))
    probs=np.where(SEQS[None]==1,p1,1-p1).prod(axis=2)
    return probs@REWARD


def case(eta):
    q,score_old,q_tok=policy(TH0)
    acc=dict(kl=0.,fmix=0.,imix=0.,gain=0.,gain_cond=0.,grad_f=np.zeros(4),grad_i=np.zeros(4))
    memo={}
    for S in TRIPLES:
        r=REWARD[S]; st=r.std()
        adv=(r-r.mean())/st if st>1e-12 else np.zeros(3)
        new=TH0+eta*(adv[:,None]*score_old[S]).mean(axis=0)
        key=tuple(np.round(new,13))
        if key not in memo:
            p,score,p_tok=policy(new)
            K=next_kernel(q_tok,p_tok)
            assert np.max(np.abs(q@K-p))<1.e-12
            g=grpo_gradient(score,TRIPLES)
            after=reward_of_thetas(new[None]+0.25*g)
            w_f=np.prod(p[TRIPLES],axis=1)
            rho=float(p@REWARD)
            mf=3*rho*(1-rho)
            F2=float(w_f@after)
            F2cond=(F2-(1-mf)*rho)/mf
            g_f=w_f@g
            memo[key]=(p,score,K,g,after,mf,rho,F2,F2cond,g_f)
        p,score,K,g,after,mf,rho,F2,F2cond,g_f=memo[key]
        w_s=float(np.prod(q[S]))
        w_i=np.prod(np.array([K[S[k],TRIPLES[:,k]] for k in range(3)]),axis=0)
        I2=float(w_i@after)
        I_mix=1-float(np.prod(K[S]@REWARD))-float(np.prod(K[S]@(1-REWARD)))
        g_i=w_i@g
        acc['kl']+=w_s*float(q@np.log(q/p))
        acc['fmix']+=w_s*mf;acc['imix']+=w_s*I_mix
        acc['gain']+=w_s*(F2-I2)
        # Exact matched actor/survival: both arms weighted by reuse mixed probability.
        acc['gain_cond']+=w_s*(I_mix*F2cond-(I2-(1-I_mix)*rho))
        acc['grad_f']+=w_s*g_f
        acc['grad_i']+=w_s*g_i
    return dict(eta=eta,kl=acc['kl'],mixed_ratio=acc['imix']/acc['fmix'],
                fresh_minus_reuse_after2=acc['gain'],
                actor_matched_mixed_only_gap=acc['gain_cond']/acc['imix'],
                gradient_gap_norm=np.linalg.norm(acc['grad_f']-acc['grad_i']),
                grad_projection_delta=(acc['grad_f']-acc['grad_i'])@acc['grad_f']/np.linalg.norm(acc['grad_f']),
                avg_fresh_gradient=acc['grad_f'],avg_reuse_gradient=acc['grad_i'])

if __name__=='__main__':
    for eta in [0.,.025,.05,.2,.8]:
        x=case(eta)
        print(f"eta={eta:g} KL={x['kl']:.10f} yield_ratio={x['mixed_ratio']:.7f} "
              f"after2gap={x['fresh_minus_reuse_after2']:.10f} "
              f"matchedgap={x['actor_matched_mixed_only_gap']:.10f} "
              f"gradient_gap_norm={x['gradient_gap_norm']:.7f} "
              f"gradient_gap_proj={x['grad_projection_delta']:.7f}")
