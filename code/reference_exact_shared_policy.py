#!/usr/bin/env python3
"""Exact, finite, shared-parameter autoregressive GRPO + residual verifier lab.

No trained language-model weights and NO claim of production SPEC-RL fidelity.
Multi-token binary sequences, a shared four-parameter conditional logit model,
endogenous 3-sample detached binary-advantage score update, all old groups
enumerated, fresh/held-out/exact-residual/ordinary-fallback arms compared at
same frozen updated actor. No Monte Carlo or inferred confidence intervals.

Python 3, numpy. Run: python code/reference_exact_shared_policy.py --fast --output results/exact.json
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import itertools
import json
import math
import time
from functools import lru_cache
from pathlib import Path
import numpy as np

SQRT2=math.sqrt(2.)
DEFAULT_THETA=np.array([-0.2,0.35,0.15,0.25],dtype=float)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z,-45,45)))


@dataclass
class Space:
    T:int
    seq:np.ndarray
    f:np.ndarray
    counts:np.ndarray
    reward:np.ndarray
    prefix_at:tuple


def make_space(T:int,task:str='majority') -> Space:
    assert 2<=T<=6
    seq=np.array(list(itertools.product((0,1),repeat=T)), dtype=np.int8)
    M=len(seq)
    f=np.empty((M,T,4),float)
    for j,y in enumerate(seq):
        for t in range(T):
            f[j,t]=[1.0, (2*int(y[t-1])-1) if t else 0.0,
                    2*t/max(T-1,1)-1.0, (2*sum(y[:t])-t)/float(T)]
    counts=seq.sum(axis=1)
    if task=='majority':
        reward=(counts>=math.ceil((T+1)/2)).astype(float)
    elif task=='all_ones':reward=(counts==T).astype(float)
    elif task=='parity':reward=(counts%2==0).astype(float)
    elif task=='adjacent11':reward=np.array([int(any(y[t]==y[t+1]==1 for t in range(T-1))) for y in seq],float)
    else:raise ValueError(task)
    prefix_at=tuple({k:np.array([i for i in range(M) if tuple(seq[i,:len(k)])==k],int)
                     for k in itertools.product((0,1),repeat=t)} for t in range(T+1))
    return Space(T,seq,f,counts,reward,prefix_at)


def actor(space:Space,theta:np.ndarray):
    z=space.f@theta
    p1=sigmoid(z)
    prob=np.prod(np.where(space.seq==1,p1,1-p1),axis=1)
    score=np.sum((space.seq-p1)[:,:,None]*space.f,axis=1)
    assert abs(float(prob.sum())-1)<1.e-11
    return prob,score,p1


def fresh_stats(space,theta):
    prob,score,p1=actor(space,theta)
    rho=float(prob@space.reward)
    drho=(prob*space.reward)@score
    mixed=3*rho*(1-rho)
    grad=SQRT2*drho
    return rho,mixed,grad,prob,score,p1


def cached_group_update(space,S,old_score,eta,theta0):
    r=space.reward[list(S)]
    s=math.sqrt(float(np.mean((r-r.mean())**2)))
    a=(r-r.mean())/s if s>1e-14 else np.zeros(3)
    return theta0+eta*(a[:,None]*old_score[list(S)]).mean(axis=0)


def verifier(space:Space, qprob:np.ndarray, qtoken1:np.ndarray,
             pprob:np.ndarray, ptoken1:np.ndarray, fallback:bool=False):
    """Whole-sequence K[y,x]; q K_exact = p, up to floating rounding.

    Token-by-token: verify old y_t while prefixes match; on first rejection
    sample from true positive-part residual (or ordinary p for fallback), then
    sample the remaining suffix from the same frozen p.
    """
    seq=space.seq; M,T=seq.shape
    K=np.zeros((M,M),float)
    expected_suffix=np.zeros(M,float)
    expected_rejected=np.zeros(M,float)
    expected_accepted=np.zeros(M,float)
    for yi,y in enumerate(seq):
        keep=1.
        for t in range(T):
            qy=float(qtoken1[yi,t]) if y[t] else 1-float(qtoken1[yi,t])
            py=float(ptoken1[yi,t]) if y[t] else 1-float(ptoken1[yi,t])
            a=min(1.,py/qy)
            rej=keep*(1-a)
            if rej>0:
                expected_rejected[yi]+=rej
                expected_suffix[yi]+=rej*(T-t-1)
                prefix=tuple(int(v) for v in y[:t])
                if fallback:
                    ids=space.prefix_at[t][prefix]
                    K[yi,ids]+=rej*(pprob[ids]/pprob[ids].sum())
                else:
                    ids=space.prefix_at[t][prefix]
                    q1=float(qtoken1[yi,t]); p1=float(ptoken1[yi,t]);
                    positive=np.array([max(0.,(1-p1)-(1-q1)),max(0.,p1-q1)])
                    Z=float(sum(positive))
                    if not Z>0: raise AssertionError('positive residual missing')
                    for next_bit in (0,1):
                        if positive[next_bit]<=0:continue
                        ids_next=ids[seq[ids,t]==next_bit]
                        K[yi,ids_next]+=rej*(positive[next_bit]/Z)*(pprob[ids_next]/pprob[ids_next].sum())
            keep*=a
            expected_accepted[yi]+=keep
        K[yi,yi]+=keep
    if np.max(np.abs(K.sum(axis=1)-1))>2e-10:raise AssertionError('invalid verifier kernel')
    if not fallback and np.max(np.abs(qprob@K-pprob))>2e-10:
        raise AssertionError('independent proposal did not reproduce p exactly')
    return K,expected_suffix,expected_rejected,expected_accepted


def expected_gradient_independent_members(space,member_laws,score):
    """Exact E[next detached standardized GRPO gradient] for three independent
       (not identically distributed) arbitrary trajectory laws and binary reward.
       Also computes exact mixed-group survival probability.
    """
    r=space.reward
    rr=member_laws@r # shape (3,)
    pos=(member_laws*r[None,:])@score
    neg=(member_laws*(1-r)[None,:])@score
    grad=np.zeros(score.shape[1],float)
    for i in range(3):
        j,k=[m for m in range(3) if m!=i]
        zero=(1-rr[j])*(1-rr[k]); two=rr[j]*rr[k]
        one=1-zero-two
        grad+=((SQRT2*zero+one/SQRT2)*pos[i]
              +(-SQRT2*two-one/SQRT2)*neg[i])/3.
    mixed=1-float(np.prod(rr))-float(np.prod(1-rr))
    expected_reward=float(rr.mean())
    return grad,mixed,expected_reward


def sample_group_gradient(space,score,indices):
    """Vectorized exact next GRPO estimator per group of 3 sequence indices."""
    rewards=space.reward[indices]
    mean=rewards.mean(axis=1)
    sd=np.sqrt(np.mean((rewards-mean[:,None])**2,axis=1))
    A=np.zeros_like(rewards)
    mask=sd>1e-12
    A[mask]=(rewards[mask]-mean[mask,None])/sd[mask,None]
    return np.mean(A[:,:,None]*score[indices],axis=1)


def actor_reward_batch(space,thetas):
    """Exact reward expectation of many actors by enumerating all sequences."""
    thetas=np.atleast_2d(thetas)
    out=np.zeros(len(thetas),float)
    # Batch to avoid giant tensor when extended.
    for start in range(0,len(thetas),1024):
        th=thetas[start:start+1024]
        logits=np.einsum('mtd,bd->bmt',space.f,th,optimize=True)
        p1=sigmoid(logits)
        prob=np.prod(np.where(space.seq[None,:,:]==1,p1,1-p1),axis=2)
        out[start:start+len(th)]=prob@space.reward
    return out


@lru_cache(None)
def all_triples(M):
    return np.array(list(itertools.product(range(M),repeat=3)),dtype=np.int16)


def exact_second_step(space,theta,score,p,K,cache_indices,beta=0.25):
    """Exact expected *population reward* after one additional STOCHASTIC GRPO step.
    p baseline iid group; verified used cache laws K[each member, :].
    One realized actor theta in both cases. No optimizer state/clipping.
    """
    triples=all_triples(len(p))
    g=sample_group_gradient(space,score,triples)
    after=actor_reward_batch(space,theta[None,:]+beta*g)
    fresh_law=p[triples].prod(axis=1)
    # i-th x_i. Avoid accidental broadcasting over member dimension.
    cache_law=K[cache_indices[0],triples[:,0]]*K[cache_indices[1],triples[:,1]]*K[cache_indices[2],triples[:,2]]
    return float(fresh_law@after),float(cache_law@after)


def run_exact(T=3,task='majority',eta=0.8,beta=0.25,theta0=DEFAULT_THETA,
              compute_second=False,max_triples=100000):
    start=time.perf_counter()
    space=make_space(T,task)
    q,old_score,old_p1=actor(space,theta0)
    M=len(q)
    old_triples=all_triples(M)
    if len(old_triples)>max_triples:raise ValueError('exact enumeration exceeds budget')
    results={name:0.0 for name in (
        'fresh_group_yield','included_group_yield','fallback_group_yield','reserve_fallback_group_yield',
        'fresh_reward','included_reward','fallback_reward','reserve_fallback_reward',
        'fresh_grad_norm','grad_delta_norm_weighted','grad_delta_component_0',
        'same_actor_grad_discrepancy_sq','policy_kl_qp','policy_kl_pq','p_old_tv',
        'mean_sample_abs_logratio','mean_sample_k1','mean_sample_k3','sample_clipfrac_02',
        'sample_token_mean_abs_logratio','sample_token_k1','sample_token_k3','sample_token_clipfrac_02',
        'residual_rejection_prob','fallback_rejection_prob','regenerated_suffix_tokens_per_seq',
        'fresh_after_second','included_after_second','fallback_after_second','reserve_fallback_after_second','unverified_after_second',
        'unverified_group_yield','unverified_grad_delta_norm',
        'matched_filter_two_step_fresh_advantage_numerator')}
    grad_delta=np.zeros(len(theta0)); g_fresh_mean=np.zeros(len(theta0)); g_inc_mean=np.zeros(len(theta0))
    filt_delta=np.zeros(len(theta0)); weighted_filt_fresh=np.zeros(len(theta0)); weighted_filt_included=np.zeros(len(theta0))
    memo={}; cache_reuse=0; n_cases=0
    for S in old_triples:
        S=tuple(map(int,S)); weight=float(np.prod(q[list(S)]))
        newtheta=cached_group_update(space,S,old_score,eta,theta0)
        key=tuple(np.round(newtheta,13))
        if key not in memo:
            rho,frmix,frgrad,p,score,p1=fresh_stats(space,newtheta)
            k,reg,rej,accepted=verifier(space,q,old_p1,p,p1,fallback=False)
            ko,reg_o,rej_o,_=verifier(space,q,old_p1,p,p1,fallback=True)
            memo[key]=(rho,frmix,frgrad,p,score,k,ko,reg,rej,rej_o,newtheta,p1)
        else:cache_reuse+=1
        rho,frmix,frgrad,p,score,k,ko,reg,rej,rej_o,newtheta,p1=memo[key]
        ginc,ymix,meanr=expected_gradient_independent_members(space,k[list(S),:],score)
        gfo,ymixo,meanro=expected_gradient_independent_members(space,ko[list(S),:],score)
        reserve_fo=q@ko
        gfr,ymixr,meanrr=expected_gradient_independent_members(space,np.stack([reserve_fo]*3),score)
        grraw,ymixraw,_=expected_gradient_independent_members(space,np.eye(M)[list(S),:],score)
        # Matched-filter comparison reweights actor S by its probability of
        # surviving the included-reuse mixed-reward filter.
        # For both arms, discard unanimous (zero-gradient) groups.
        filt_delta+=weight*(ginc-(ymix/frmix)*frgrad)
        weighted_filt_fresh+=weight*(ymix/frmix)*frgrad
        weighted_filt_included+=weight*ginc
        grad_delta+=weight*(ginc-frgrad)
        g_inc_mean+=weight*ginc
        g_fresh_mean+=weight*frgrad
        results['fresh_group_yield']+=weight*frmix
        results['included_group_yield']+=weight*ymix
        results['fallback_group_yield']+=weight*ymixo
        results['reserve_fallback_group_yield']+=weight*ymixr
        results['unverified_group_yield']+=weight*ymixraw
        results['fresh_reward']+=weight*rho
        results['included_reward']+=weight*meanr
        results['fallback_reward']+=weight*meanro
        results['reserve_fallback_reward']+=weight*meanrr
        results['fresh_grad_norm']+=weight*np.linalg.norm(frgrad)
        results['grad_delta_norm_weighted']+=weight*np.linalg.norm(ginc-frgrad)
        results['unverified_grad_delta_norm']+=weight*np.linalg.norm(grraw-frgrad)
        results['same_actor_grad_discrepancy_sq']+=weight*np.dot(ginc-frgrad,ginc-frgrad)
        results['policy_kl_qp']+=weight*float(q@np.log(q/p))
        results['policy_kl_pq']+=weight*float(p@np.log(p/q))
        results['p_old_tv']+=weight*float(np.abs(p-q).sum()/2.)
        ratios=p[list(S)]/q[list(S)]
        logs=np.log(ratios)
        results['mean_sample_abs_logratio']+=weight*float(np.mean(abs(logs)))
        results['mean_sample_k1']+=weight*float(np.mean(-logs))
        results['mean_sample_k3']+=weight*float(np.mean(ratios-1-logs))
        results['sample_clipfrac_02']+=weight*float(np.mean((ratios<.8)|(ratios>1.2)))
        # Distinguish complete response sequence ratios from actual token
        # conditional probability ratios, as logged by PPO/GRPO trainers.
        old_one=old_p1[list(S),:]
        new_one=p1[list(S),:]
        cached_tokens=space.seq[list(S),:]
        token_ratios=np.where(cached_tokens==1,new_one/old_one,(1-new_one)/(1-old_one))
        token_logs=np.log(token_ratios)
        results['sample_token_mean_abs_logratio']+=weight*float(np.mean(abs(token_logs)))
        results['sample_token_k1']+=weight*float(np.mean(-token_logs))
        results['sample_token_k3']+=weight*float(np.mean(token_ratios-1-token_logs))
        results['sample_token_clipfrac_02']+=weight*float(np.mean((token_ratios<.8)|(token_ratios>1.2)))
        results['residual_rejection_prob']+=weight*float(np.mean(rej[list(S)]))
        results['fallback_rejection_prob']+=weight*float(np.mean(rej_o[list(S)]))
        results['regenerated_suffix_tokens_per_seq']+=weight*float(np.mean(reg[list(S)]))
        if compute_second:
            fresh2,inc2=exact_second_step(space,newtheta,score,p,k,S,beta=beta)
            _,fall2=exact_second_step(space,newtheta,score,p,ko,S,beta=beta)
            _,raw2=exact_second_step(space,newtheta,score,p,np.eye(M),S,beta=beta)
            _,resfo2=exact_second_step(space,newtheta,score,p,np.tile(reserve_fo,(M,1)),S,beta=beta)
            results['fresh_after_second']+=weight*fresh2
            results['included_after_second']+=weight*inc2
            results['fallback_after_second']+=weight*fall2
            results['reserve_fallback_after_second']+=weight*resfo2
            results['unverified_after_second']+=weight*raw2
            # Same selection on realized actor S: weight both arms by
            # P(reused mixed | S), and discard zero-gradient unanimous groups.
            fresh2_cond=(fresh2-(1-frmix)*rho)/frmix
            # Do not divide by zero for deterministic unanimous cached groups.
            inc_mixed_numerator=inc2-(1-ymix)*rho
            results['matched_filter_two_step_fresh_advantage_numerator']+=weight*(ymix*fresh2_cond-inc_mixed_numerator)
        n_cases+=1
    ret={
        'status':'exact_population_enumeration_no_sampling_uncertainty',
        'model':'four_shared_logit_parameters_autoregressive_binary_tokens',
        'T':T,'task':task,'eta':eta,'beta_second_step':beta,
        'old_theta':theta0.tolist(),'old_reward':float(q@space.reward),
        'old_group_count':n_cases,'distinct_realized_actors':len(memo),
        'actor_cache_hits':cache_reuse,
        **{k:float(v) for k,v in results.items()},
        'mean_gradient_fresh':g_fresh_mean.tolist(),
        'mean_gradient_included':g_inc_mean.tolist(),
        'mean_grad_delta':grad_delta.tolist(),
        'mean_grad_delta_norm':float(np.linalg.norm(grad_delta)),
        'gradient_alignment':float(np.dot(g_fresh_mean,g_inc_mean)/(np.linalg.norm(g_fresh_mean)*np.linalg.norm(g_inc_mean)+1e-30)),
        'group_yield_ratio':results['included_group_yield']/results['fresh_group_yield'],
        'matched_mixed_filter_gradient_delta':(filt_delta/results['included_group_yield']).tolist(),
        'matched_mixed_filter_gradient_delta_norm':float(np.linalg.norm(filt_delta/results['included_group_yield'])),
        'matched_filter_included_gradient':(weighted_filt_included/results['included_group_yield']).tolist(),
        'matched_filter_fresh_gradient':(weighted_filt_fresh/results['included_group_yield']).tolist(),
        'max_verification_fraction_for_mixed_yield_break_even':float(results['included_group_yield']/results['fresh_group_yield']-results['regenerated_suffix_tokens_per_seq']/T),
        'regenerated_tokens_per_attempted_group':3*results['regenerated_suffix_tokens_per_seq'],
        'fresh_second_minus_included_second':results['fresh_after_second']-results['included_after_second'] if compute_second else None,
        'matched_filter_two_step_fresh_advantage':results['matched_filter_two_step_fresh_advantage_numerator']/results['included_group_yield'] if compute_second else None,
        'elapsed_sec':time.perf_counter()-start,
        'limits':['single_detached_score_step','no_Adam_or_clipping','binary_synthetic_reward',
                  'sequence_exact_residual_verifier_not_released_SPEC_RL','no_wall_clock_GPU_perf']}
    return ret


def test_core():
    for task in ('majority','all_ones','parity','adjacent11'):
        space=make_space(3,task)
        oldtheta=DEFAULT_THETA
        q,score,q1=actor(space,oldtheta)
        for th in (oldtheta,oldtheta+np.array([.5,-.3,.2,-.1])):
            _,_,ref,p,s,p1=fresh_stats(space,th)
            K,_,_,_=verifier(space,q,q1,p,p1)
            assert np.allclose(q@K,p,atol=1.e-12)
            # Exact next-gradient formula compared with full group summation.
            sublaws=np.stack([K[0],K[4],K[7]])
            g,m,r=expected_gradient_independent_members(space,sublaws,s)
            triples=all_triples(len(p))
            w=sublaws[0,triples[:,0]]*sublaws[1,triples[:,1]]*sublaws[2,triples[:,2]]
            g_full=w@sample_group_gradient(space,s,triples)
            assert np.max(abs(g-g_full))<1.e-12,(g,g_full)
            rr=space.reward[triples]
            m_full=w@((rr.min(axis=1)<rr.max(axis=1)).astype(float))
            assert abs(m-m_full)<1.e-12
            # independent reserve proposal -> exact fresh gradient and yield
            gr,yr,_=expected_gradient_independent_members(space,np.stack([p,p,p]),s)
            assert np.max(abs(gr-ref))<1.e-12,(gr,ref)
            assert abs(yr-3*(p@space.reward)*(1-p@space.reward))<1e-12
    for eta in (0.,.2,1.0):
        ret=run_exact(T=2,eta=eta,compute_second=False)
        if eta==0.:
            assert np.linalg.norm(ret['mean_grad_delta'])<1e-12
            assert abs(ret['included_group_yield']-ret['fresh_group_yield'])<1e-12
    return 'PASS exact qK=p, group gradient analytic-vs-exhaustive, fresh identity, zero-update control'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fast',action='store_true',help='T=2,3,4; only T=2,3 exact second-step')
    ap.add_argument('--output',default='results/exact_shared.json')
    ap.add_argument('--test',action='store_true')
    args=ap.parse_args()
    print(test_core(),flush=True)
    configs=[(2,'majority',.2,True),(2,'majority',.8,True),(2,'majority',2.0,True),
             (3,'majority',.2,True),(3,'majority',.8,True),(3,'majority',2.0,True),
             (3,'all_ones',.8,True),(3,'parity',.8,True),
             (4,'majority',.8,False),(4,'adjacent11',.8,False)]
    if not args.fast:configs.extend([(4,'majority',2.,False),(4,'all_ones',.8,False)])
    if args.test:configs=[]
    outputs=[]
    for T,task,eta,second in configs:
        t0=time.perf_counter()
        r=run_exact(T=T,task=task,eta=eta,compute_second=second)
        outputs.append(r)
        print(f"T={T} task={task} eta={eta} cases={r['old_group_count']} "
              f"yield={r['group_yield_ratio']:.5f} next_grad_delta={r['mean_grad_delta_norm']:.6g} "
              f"KL={r['policy_kl_qp']:.5g} after2gap={r['fresh_second_minus_included_second']} "
              f"time={time.perf_counter()-t0:.3f}s",flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'schema':'spec_rl_shared_exact_v1','runs':outputs},indent=2,sort_keys=True)+'\n')
    print('WROTE',path,flush=True)

if __name__=='__main__':main()
