#!/usr/bin/env python3
"""Independent finite-enumeration check for a proposed SPEC-RL GRPO-gradient extension.

THIS IS NOT A REPRODUCTION OF REAL LLM GRPO. It exactly enumerates the
single-step uniform-N tabular construction and its residual kernel.
No neural weights or external dependencies beyond numpy.
"""
import itertools
import math
import numpy as np

ETA = 3.0 * math.sqrt(2.0) * math.log(2.0)
MIXED_GRADIENT = 2.0 * math.sqrt(2.0) / 3.0


def enumerate_exact(N, eta=ETA):
    assert N >= 4 and N % 2 == 0
    rs = np.r_[np.ones(N//2), np.zeros(N//2)]
    q = np.ones(N) / N
    n_groups, total_mixed_reuse, total_mixed_fresh, total_gap = 0, 0.0, 0.0, 0.0
    total_rejections, total_tv = 0.0, 0.0
    mean_k1, k1_count, mean_k3, clip_count = 0.0, 0, 0.0, 0
    for Y in itertools.product(range(N), repeat=3):
        rewards=rs[list(Y)]
        d=rewards-rewards.mean()
        s=math.sqrt(float(np.mean(d*d)))
        A=(d/s) if s>0 else np.zeros(3)
        z=np.zeros(N)
        for idx, adv in zip(Y, A):
            z[idx]+=(eta/3)*float(adv)
        w=np.exp(z)
        p=w/w.sum()
        rpos=np.maximum(p-q, 0)
        tv=float(rpos.sum())
        residual_reward=float(rpos @ rs / tv) if tv>1e-14 else 0.0
        acc=np.minimum(1, p[list(Y)]/q[list(Y)])
        pu=acc*rewards+(1-acc)*residual_reward
        # Conditional independence of the verifier coins and residual draws
        mixed_reuse = 1-float(np.prod(pu))-float(np.prod(1-pu))
        reward_p=float(p@rs)
        mixed_fresh=3*reward_p*(1-reward_p)
        total_mixed_reuse+=mixed_reuse
        total_mixed_fresh+=mixed_fresh
        total_gap+=float(np.mean(pu))-reward_p
        total_rejections+=float(np.mean(1-acc))
        total_tv+=tv
        if 0<rewards.sum()<3 and len(set(Y))==3:
            # sample-level diagnostics at the *pre-update cached group*
            ratios=p[list(Y)]/q[list(Y)]
            mean_k1+=float(np.mean(-np.log(ratios)))
            mean_k3+=float(np.mean(ratios-1-np.log(ratios)))
            clip_count+=sum((ratios<0.8)|(ratios>1.2))
            k1_count+=1
        n_groups+=1
    a=total_mixed_reuse/n_groups
    b=total_mixed_fresh/n_groups
    return {
        'N':N, 'mixed_reuse':a, 'mixed_fresh':b,
        'directional_gradient_reuse':a*MIXED_GRADIENT,
        'directional_gradient_fresh':b*MIXED_GRADIENT,
        'gradient_ratio':a/b,
        'reward_gap':total_gap/n_groups,
        'rejection_rate':total_rejections/n_groups,
        'mean_policy_TV':total_tv/n_groups,
        'mixed_sample_k1':mean_k1/k1_count,
        'mixed_sample_k3':mean_k3/k1_count,
        'mixed_sample_clip_fraction':clip_count/(3*k1_count),
    }


def independent_derivative_check():
    N=16
    y=np.array([0,9,12])  # one rewarded, two unrewarded
    rewards=np.r_[np.ones(N//2),np.zeros(N//2)][y]
    sd=np.sqrt(np.mean((rewards-rewards.mean())**2))
    A=(rewards-rewards.mean())/sd
    logits=np.zeros(N)
    for i in range(3): logits[y[i]]+=ETA*A[i]/3
    v=np.r_[np.ones(N//2),-np.ones(N//2)]
    def surrogate(phi):
        zz=logits+phi*v
        logp=zz-np.log(np.exp(zz-np.max(zz)).sum())-np.max(zz)
        return np.mean(A*logp[y])
    eps=1.e-6
    numeric=(surrogate(eps)-surrogate(-eps))/(2*eps)
    assert abs(numeric-MIXED_GRADIENT)<1.e-8,(numeric,MIXED_GRADIENT)


def main():
    independent_derivative_check()
    theta=2*math.log(2)
    t=math.exp(-theta/2)
    expected_reuse=3*t/4
    expected_fresh=3/4
    expected_ratio=t
    expected_reward_gap=(3-2*t-t*t)/8
    print('Analytic large-N predictions:')
    print('  reuse mixed',expected_reuse,'fresh mixed',expected_fresh)
    print('  reward gap',expected_reward_gap, 'projected gradient ratio',expected_ratio)
    assert math.isclose(expected_reward_gap,7/32,abs_tol=1.e-12)
    assert math.isclose(MIXED_GRADIENT*(expected_fresh-expected_reuse),math.sqrt(2)/4,abs_tol=1.e-12)
    print('Finite exact exhaustive enumerations:')
    for N in (4,8,16,32):
        data=enumerate_exact(N)
        print(' ',data)
        assert data['gradient_ratio'] < .51 and data['gradient_ratio']>.40
        assert data['mixed_sample_clip_fraction']==1.0
        assert abs(data['reward_gap']) <= data['rejection_rate']+data['mean_policy_TV']+1.e-11
    print('PASS: independent score derivative, exact finite enumeration, limiting algebra and ratio diagnostics')

if __name__=='__main__':main()
