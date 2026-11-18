# Worked examples of an established conditional-proposal problem

Arnal et al. Appendix C.1 already distinguishes a reused sample's old nominal distribution from its law conditional on subsequent iterates. These examples illustrate that principle; they are not a claim of a new RL discovery.

## Single-sample REINFORCE

Uniform q on even N; R=1 on half the vocabulary. Hold baseline 1/2 fixed and update logits by `2*log(2)*(R(Y)-1/2)*(one_hot(Y)-q)`. After cancelling the common offset, the sampled token weight is 2 if rewarded and 1/2 otherwise; other weights are 1.

| Sample class (probability 1/2 each) | Exact-residual verified reward | Fresh reward at the same actor |
|---|---|---|
| Rewarded | 1 | (N/2+1)/(N+1) |
| Unrewarded | N/[2(2N-1)] | N/(2N-1) |

The second row follows from acceptance N/(2N-1) and a residual uniform over the other N-1 tokens. Subtraction and averaging give `N*(N-2)/(4*(N+1)*(2*N-1))`, tending to **1/8**. At N=2 the gap is zero: the two samples happen to produce the same actor, an important finite sanity check. For either sampled weight w, D=N-1+w, TV<=1/N, KL(q||p)=log(D/N)-log(w)/N and KL(p||q)=(w/D)*log(w)-log(D/N), both O(1/N).

## Three-response standardized-score specialization

At eta=3*sqrt(2)*log(2), detached population-standardized advantages produce weights (4,1/2,1/2) or (2,2,1/4). Repeated samples multiply weights at the same coordinate. Each mixed reward count has probability 3/8 and collisions vanish as N grows.

| Mixed reward count | Exact residual: limiting output mean | Ordinary-target rejection: limiting output mean |
|---|---:|---:|
| 1 rewarded | 1/3+(2/3)(1-1/2)=2/3 | 1/3+(2/3)(1-1/2)(1/2)=1/2 |
| 2 rewarded | 2/3+(1/3)(1-1/4)=11/12 | 2/3+(1/3)(1-1/4)(1/2)=19/24 |

Add the unanimous-success mass 1/8, weight each mixed case by 3/8, then subtract fresh limit 1/2. The gaps are **7/32** and **7/64**. The latter is specific to this endogenous one-token example; it is not an independent-cache fallback identity or a released training measurement. That independent-proposal identity is `q K_ordinary - p = TV(p,q)*(p-r)`, already distinct from conditional cache dependence. SPEC-RL issue 14 predates these illustrations.

## What the executable checks do

`code/worked_examples.py`: `reinforce_dense` exhaustively averages complete rational kernel rows and checks literal floating-point score logits; `reinforce_formula` supplies a separate finite expression. `grpo_dense` enumerates ordered groups and full token kernels; `grpo_orbits` separately aggregates reward counts and collisions. `run_checks` compares both routes and the limiting reward-count algebra. Finite cases include repeats, full support and independent verifier coins. No sampling experiment or production implementation is substituted for these calculations.

A finite numerical limit check is not an all-N proof: the displayed formulas and vanishing collision bound supply the analytic argument. These are fixed-step, support-growing illustrations. Correct conditional proposals and independent verification can sample exactly; no impossibility claim follows.
