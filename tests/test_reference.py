#!/usr/bin/env python3
"""Independent reference verification; stdlib unittest + numpy only."""
import itertools
import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import reference_exact_shared_policy as lab


def reference_kernel_row(space,y,old_theta,new_theta,fallback=False):
    """Reference tree expansion, not the prefix-probability matrix algorithm."""
    P={}
    def walk(prefix,idx,prob,verifying):
        if idx==space.T:
            P[prefix]=P.get(prefix,0.)+prob
            return
        pfeat=space.f[np.flatnonzero(np.all(space.seq[:,:len(prefix)]==np.asarray(prefix),axis=1))[0],idx]
        p1=float(lab.sigmoid(pfeat@new_theta)); q1=float(lab.sigmoid(pfeat@old_theta))
        if not verifying:
            walk(prefix+(0,),idx+1,prob*(1-p1),False)
            walk(prefix+(1,),idx+1,prob*p1,False)
            return
        tok=int(y[idx]); pv=p1 if tok else 1-p1; qv=q1 if tok else 1-q1
        a=min(1.,pv/qv)
        walk(prefix+(tok,),idx+1,prob*a,True)
        if a<1:
            if fallback:
                walk(prefix+(0,),idx+1,prob*(1-a)*(1-p1),False)
                walk(prefix+(1,),idx+1,prob*(1-a)*p1,False)
            else:
                residual=[max(0.,q1-p1),max(0.,p1-q1)]
                z=sum(residual)
                for x in (0,1):
                    if residual[x]>0:
                        walk(prefix+(x,),idx+1,prob*(1-a)*residual[x]/z,False)
    walk((),0,1.,True)
    return np.array([P.get(tuple(map(int,s)),0.) for s in space.seq],float)


class ExactReferences(unittest.TestCase):
    def setUp(self):
        self.old=lab.DEFAULT_THETA.copy()

    def test_independent_recursive_verifier(self):
        for T in (2,3,4):
            sp=lab.make_space(T,'majority')
            q,qs,q1=lab.actor(sp,self.old)
            for new in (self.old+np.array([.4,-.25,.1,.3]),self.old-np.array([.35,.1,.2,-.15])):
                p,_,p1=lab.actor(sp,new)
                for fallback in (False,True):
                    K,_,_,_=lab.verifier(sp,q,q1,p,p1,fallback=fallback)
                    for i,y in enumerate(sp.seq):
                        ref=reference_kernel_row(sp,y,self.old,new,fallback)
                        self.assertLess(max(abs(ref-K[i])),1.e-12)
                    if not fallback:
                        self.assertLess(max(abs(q@K-p)),1.e-12)

    def test_independent_group_expectation(self):
        rng=np.random.default_rng(20261117)
        sp=lab.make_space(3,'all_ones')
        p,score,_=lab.actor(sp,self.old+np.array([.7,.1,-.5,.2]))
        laws=rng.dirichlet(np.ones(len(p)),size=3)
        got,yield_g,rew=lab.expected_gradient_independent_members(sp,laws,score)
        expected=np.zeros(4)
        ym=rm=0.
        for ids in itertools.product(range(len(p)),repeat=3):
            w=float(np.prod([laws[i,ids[i]] for i in range(3)]))
            rv=sp.reward[list(ids)]
            sd=float(np.std(rv))
            a=(rv-rv.mean())/sd if sd>1e-12 else np.zeros(3)
            expected+=w*(a@score[list(ids)])/3
            ym+=w*(min(rv)<max(rv))
            rm+=w*float(np.mean(rv))
        self.assertTrue(np.allclose(got,expected,atol=1.e-12))
        self.assertAlmostEqual(yield_g,ym,places=12)
        self.assertAlmostEqual(rew,rm,places=12)

    def test_two_step_independent_manual_reference(self):
        sp=lab.make_space(3,'majority')
        q,sold,q1=lab.actor(sp,self.old)
        S=(1,2,7)
        th=lab.cached_group_update(sp,S,sold,.4,self.old)
        p,score,p1=lab.actor(sp,th)
        K,*_=lab.verifier(sp,q,q1,p,p1)
        fresh,inc=lab.exact_second_step(sp,th,score,p,K,S,beta=.2)
        manual_fresh=manual_inc=0.
        for ids in itertools.product(range(len(p)),repeat=3):
            r=sp.reward[list(ids)]
            sd=float(np.std(r));a=(r-r.mean())/sd if sd>1e-14 else np.zeros(3)
            gg=(a@score[list(ids)])/3.
            theta2=th+.2*gg
            held=lab.actor(sp,theta2)[0]@sp.reward
            manual_fresh+=np.prod(p[list(ids)])*held
            manual_inc+=np.prod([K[S[j],ids[j]] for j in range(3)])*held
        self.assertAlmostEqual(fresh,manual_fresh,places=12)
        self.assertAlmostEqual(inc,manual_inc,places=12)

    def test_fresh_gradient_equals_reward_derivative(self):
        sp=lab.make_space(4,'adjacent11')
        theta=self.old+np.array([.22,-.31,.03,.07])
        rho,mixed,grad,*_=lab.fresh_stats(sp,theta)
        for d in range(4):
            h=1.e-5
            e=np.eye(4)[d]
            plus=lab.actor(sp,theta+h*e)[0]@sp.reward
            minus=lab.actor(sp,theta-h*e)[0]@sp.reward
            self.assertAlmostEqual((plus-minus)/(2*h)*math.sqrt(2),grad[d],places=8)

    def test_computed_finite_two_step_effect(self):
        # Exact exhaustive, not estimator CI. Confirm it disappears at zero engagement.
        a=lab.run_exact(3,'majority',eta=0.,compute_second=True)
        self.assertAlmostEqual(a['fresh_second_minus_included_second'],0,places=12)
        b=lab.run_exact(3,'majority',eta=.05,compute_second=True)
        self.assertGreater(b['fresh_second_minus_included_second'],.001)
        self.assertLess(b['policy_kl_qp'],.0006)
        self.assertAlmostEqual(b['sample_clipfrac_02'],0.,places=12)
        self.assertAlmostEqual(b['sample_token_clipfrac_02'],0.,places=12)
        self.assertGreater(b['matched_mixed_filter_gradient_delta_norm'],0.006)
        self.assertGreater(b['reserve_fallback_group_yield'],0.)

    def test_cached_actor_token_diagnostics_independent(self):
        """Catch actor-memo p1 leakage: recompute every original group independently."""
        sp=lab.make_space(3,'majority')
        q,old_score,old_p1=lab.actor(sp,self.old)
        eta=0.2
        original=lab.run_exact(3,'majority',eta=eta,compute_second=False)
        token_clip=token_k1=token_k3=token_abs=0.0
        sequence_clip=0.0
        for group in itertools.product(range(len(q)),repeat=3):
            weight=float(np.prod(q[list(group)]))
            # Force a fresh actor evaluation each time: no lab memo or verifier.
            new=lab.cached_group_update(sp,group,old_score,eta,self.old)
            p,_,p1=lab.actor(sp,new)
            seq=sp.seq[list(group)]
            r=np.where(seq==1,p1[list(group)]/old_p1[list(group)],
                       (1-p1[list(group)])/(1-old_p1[list(group)]))
            l=np.log(r)
            token_clip+=weight*float(np.mean((r<.8)|(r>1.2)))
            token_k1+=weight*float(np.mean(-l))
            token_k3+=weight*float(np.mean(r-1-l))
            token_abs+=weight*float(np.mean(abs(l)))
            group_ratios=p[list(group)]/q[list(group)]
            sequence_clip+=weight*float(np.mean((group_ratios<.8)|(group_ratios>1.2)))
        self.assertAlmostEqual(original['sample_token_clipfrac_02'],token_clip,places=12)
        self.assertAlmostEqual(original['sample_token_k1'],token_k1,places=12)
        self.assertAlmostEqual(original['sample_token_k3'],token_k3,places=12)
        self.assertAlmostEqual(original['sample_token_mean_abs_logratio'],token_abs,places=12)
        self.assertAlmostEqual(original['sample_clipfrac_02'],sequence_clip,places=12)
        self.assertGreater(sequence_clip,token_clip)
        self.assertGreater(original['actor_cache_hits'],0)

    def test_same_actor_reserve(self):
        sp=lab.make_space(3,'majority')
        q,_,q1=lab.actor(sp,self.old)
        new=self.old+np.array([.3,.15,-.1,.2])
        p,score,p1=lab.actor(sp,new)
        K,*_=lab.verifier(sp,q,q1,p,p1)
        independent_reserve=q@K
        np.testing.assert_allclose(independent_reserve,p,atol=1.e-12)
        g,m,r=lab.expected_gradient_independent_members(sp,np.stack([independent_reserve]*3),score)
        _,_,target,*_=lab.fresh_stats(sp,new)
        np.testing.assert_allclose(g,target,atol=1.e-12)


if __name__=='__main__':unittest.main(verbosity=2)
