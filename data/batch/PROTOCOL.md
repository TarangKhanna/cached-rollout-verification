# Finite qualification protocol

Scientific excerpt of the protocol frozen on 17 November 2026 before the archived experiment. Administrative instructions are omitted; comparisons, seeds, thresholds and controls are unchanged. Original full protocol SHA-256: `32eb371bed6646585c30418f86120a887568ac15744d64fb9b46e2568eb165f7`.

## Actor, cells and replication

Independent implementation: binary T=3 four-shared-parameter actor,
G=3 population std, zero-variance advantage zero, detached GRPO score,
SGD first update theta=theta0+eta*mean old group gradients. B={1,8,32,128},
eta={0,.05,.20,.80}. .80 is stress/engagement positive control, not realistic KL.
1024 independent update replicas per cell, fixed seeds 2026111701+cell_index.
No extra seeds/arms/metrics after outcome inspection. All 512 next groups integrated.
Independent seed-20261117 supplied check is separate replication, not pooled.
## Verifier arms

At each realized actor evaluate A fresh p, B reserve exact residual,
C included exact residual, D included ordinary-target fallback, E reserve ordinary
fallback. Reserve conditional expectations are integrated over independent q;
no reserve cache enters training. Exact binary residual flips the first rejected
bit; fallback samples p at that position. Target suffix is generated from p.
Verifier randomness integrated analytically; no reused seed. Record qK-p and qL-p.
## Primary statistics

Primary statistic C-A mean next-score gradient projected on the normalized
old fresh expected gradient. Also all four vector components and bias norm;
mean/95% normal Monte Carlo intervals by independent policy-update replica.
Use leave-one-group-out paired control variate for C-A: D(S,theta)-D(S,theta_minus),
with theta_minus excluding S contribution with denominator B unchanged.
Keep raw estimates; at eta=0 raw unconditional bias must contain zero.
Record fresh single-B-group stochastic gradient noise, population next-gradient
norm, and componentwise approximate simultaneous bounds (Bonferroni z=2.734).
## Filtering and operation counts

Group mixed-yield and matched-actor filtering: compute conditional survival,
filtered gradient numerator and compare with fresh conditional gradient at the SAME
actor, weighted by included survival. Conditional filtered contrast is pooled
numerator / included survival; do not pool independent retained actor distributions.
Record attempted groups, first-rejection regenerated tokens and expected refill
cost as toy operation counts only; acquiring old cache is an explicit sunk/additive
cost and no GPU-hour/HBM/KV efficiency inference follows. Retain unfiltered results.
## Scope

Finite exact reference measures expected population reward after one stochastic
second SGD group update eta2=.25. New batch experiment measures next mean-score
gradient only; not neural degradation, applied Adam/PPO or multi-update learning.
Do not manufacture source parity. Source qualification requires processed logprob,
sample/population std+epsilon, clipping, optimizer accumulation, masking and backend
parity with actual pinned code/config. Report missing components individually.
## Controls

Positive controls: finite B=1 eta=.20 gradient gap >1e-3; reserve ordinary
fallback has nonzero exact qL-p (>1e-8) when p!=q. Reserve exact qK-p <1e-10;
zero-step mean C-A includes zero. A null is not accepted if these fail.
## Prespecified materiality threshold

Operational materiality gate: systematic vector next-gradient change >=5% of
fresh expected-gradient norm and exceeds fresh B-group stochastic-noise scale;
or >=10% net useful-group loss per measured compute. These are local decision
thresholds. At B>=32, use upper confidence bound on the vector bias, not mere
non-significance. Toy yield/token counts cannot satisfy the measured-compute gate.
Missing source parity prevents declaring a real-model null.

Hypotheses: Hnull B dilution leaves practically immaterial toy effect; Hmechanism
substantial independent-cache-controlled effect persists at large B; Hfallback
ordinary-target fallback creates separate reserve mismatch. No claim of novelty
for the smoothness/leave-one-group-out bound.
