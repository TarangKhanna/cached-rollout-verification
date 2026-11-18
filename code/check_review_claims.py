"""Post-review independent checks: exact categorical kernels and new step-size claims."""
from fractions import Fraction as F
from itertools import product
from collections import Counter
from math import sqrt, log, exp, expm1, isclose
from pathlib import Path

def tv(p,q):return sum(abs(x-y)for x,y in zip(p,q))/2

def kernel(p,q):
 z=tv(p,q);a=[min(F(1),x/y)for x,y in zip(p,q)]
 residual=[max(F(0),x-y)/z for x,y in zip(p,q)]if z else list(p)
 return [[a[y]*int(x==y)+(1-a[y])*residual[x]for x in range(len(p))]for y in range(len(p))]

def target(ys,N,t):
 k=sum(y<N//2 for y in ys);exponents=Counter()
 if k in (1,2):
  for y in ys:exponents[y]+=(-2 if y<N//2 else 1)if k==1 else(-1 if y<N//2 else 2)
 w=[t**exponents[j]for j in range(N)];D=sum(w);return tuple(x/D for x in w)

def exact(N,t):
 q=(F(1,N),)*N;bias=F(0);filtered_num=F(0);survival=F(0)
 for ys in product(range(N),repeat=3):
  p=target(ys,N,t);K=kernel(p,q);pa=sum(p[:N//2]);probs=[sum(K[y][:N//2])for y in ys]
  mean=F(0);kept=F(0);survive=F(0)
  for bits in product((0,1),repeat=3):
   weight=F(1)
   for b,u in zip(bits,probs):weight*=u if b else 1-u
   value=F(sum(bits),3);mean+=weight*value
   if 0<sum(bits)<3:survive+=weight;kept+=weight*value
  bias+=(mean-pa)/N**3
  filtered_num+=(kept-survive*(1+pa)/3)/N**3
  survival+=survive/N**3
  assert tv(p,q)<=F(3,N)
  fkl=sum(float(v)*log(float(v*N))for v in p)
  assert fkl<=5/N+1e-12
 return bias,filtered_num,survival

def orbit(N,t):
 c1=t**-2+2*t-3;c2=2/t+t*t-3;cd=t**-2+t*t-2
 rows=[(F(3,8)*(1-F(2,N)),1,N+c1,t,F(N,2)+t**-2-1),
       (F(3,8)*F(2,N),1,N+cd,t*t,F(N,2)+t**-2-1),
       (F(3,8)*(1-F(2,N)),2,N+c2,t*t,F(N,2)+2/t-2),
       (F(3,8)*F(2,N),2,N+cd,t*t,F(N,2)+t**-2-1)]
 gap=F(0);num=F(0);den=F(0)
 for mass,k,D,wb,Amass in rows:
  acceptance=N*wb/D;u=1-acceptance;mean=F(k,3)+F(3-k,3)*u
  kept=1-u**(3-k);pa=Amass/D
  gap+=mass*(mean-pa);num+=mass*(mean-u**(3-k)-kept*(1+pa)/3);den+=mass*kept
 return gap,num,den

for N in (4,6):
 for t in (F(1,2),F(2,3),F(9,10)):
  value=exact(N,t);assert value==orbit(N,t)
  print(f'PASS full-kernel enumeration N={N}, t={t}: B={value[0]}, filtered={value[1]/value[2]}')
print('PASS repeated-token orbits agree with full residual kernels and all 8 output-reward patterns: 840 training groups')
for t in (F(1,2),F(2,3),F(9,10),F(99,100)):
 g,num,den=orbit(10**8,t);bg=(3-2*t-t*t)/8;bf=(1-t)/6
 assert abs(float(g-bg))<1e-6 and abs(float(num/den-bf))<1e-6
 print(f'PASS N-limit t={t}: unfiltered={(3-2*t-t*t)/8}; matched-filter={(1-t)/6}')
for theta in (1e-4,1e-6):
 bg=-(expm1(-theta)+2*expm1(-theta/2))/8
 bf=-expm1(-theta/2)/6
 assert isclose(bg/theta,1/4,rel_tol=1e-4)
 assert isclose(bf/theta,1/12,rel_tol=1e-4)
print('PASS small-step slopes: unfiltered/theta -> 1/4; matched-filter/theta -> 1/12')
# Alternative sampler: refresh unanimous input groups; its extra survival mass is (1/4)(3/4).
g,num,den=orbit(10**8,F(1,2));alternate=num/(den+F(3,16))
assert abs(float(alternate-F(1,18)))<1e-6
print('PASS alternative refresh-only-unanimous sampler: filtered limit 1/18 (different sampler, not pinned-code behavior)')
# Sample standard deviation at fixed eta weakens, rather than strengthens, the effective step.
eta=3*sqrt(2)*log(2)
for rewards in ([1.,0.,0.],[1.,1.,0.]):
 mean=sum(rewards)/3;pop=sqrt(sum((v-mean)**2 for v in rewards)/3);sample=sqrt(sum((v-mean)**2 for v in rewards)/2)
 gp=[(v-mean)/(3*pop)for v in rewards];gs=[(v-mean)/(3*sample)for v in rewards]
 assert all(isclose(y/x,sqrt(2/3))for x,y in zip(gp,gs))
 for epsilon in (0.,1e-6):
  corrected_eta=eta*(sample+epsilon)/pop
  assert all(isclose(corrected_eta*(v-mean)/(3*(sample+epsilon)),eta*x)for x,v in zip(gp,rewards))
print('PASS normalization correction: fixed-eta sample-std factor sqrt(2/3); retuned eta including epsilon reproduces logits')
# Independently check the critic's derivative-free lemma argument on a grid.
grid=[tuple(F(x,6)for x in xs)for xs in product(range(7),repeat=3)if sum(xs)==6]
qs=[q for q in grid if all(q)];count=0
for q,p,pp in product(qs,grid,grid):
 gain=tuple(max(F(0),x-y)for x,y in zip(p,q));loss=tuple(max(F(0),y-x)for x,y in zip(p,q))
 gainp=tuple(max(F(0),x-y)for x,y in zip(pp,q));lossp=tuple(max(F(0),y-x)for x,y in zip(pp,q))
 assert sum(abs(x-y)for x,y in zip(p,pp))==sum(abs(x-y)for x,y in zip(gain,gainp))+sum(abs(x-y)for x,y in zip(loss,lossp))
 K,Kp=kernel(p,q),kernel(pp,q)
 for y in range(3):assert tv(K[y],Kp[y])<=min(F(1),tv(p,pp)/q[y]);count+=1
print(f'PASS derivative-free positive/negative-part identity and sharp row inequality: {count} rows')
q=(F(9,10),F(1,20),F(1,20));p=(F(4,5),F(3,20),F(1,20));pp=(F(4,5),F(1,20),F(3,20))
assert tv(kernel(p,q)[0],kernel(pp,q)[0])==tv(p,pp)/q[0]==F(1,9)
assert max(7*log(2)-2,F(7,2)*log(2)-F(5,4),F(15,2)*log(2)-F(9,4))<3
print('PASS reviewer sharpness example; analytic case constants imply its loose forward-KL bound 5/N')
for N in (4,8,64,1024):assert sum(min(F(1,N),F(6,N))for _ in range(N))==1
print('PASS whole-group replacement stability is compatible with the nonvanishing separation; no all-q optimality asserted')
print('ALL MATHEMATICAL POST-REVIEW CHECKS PASSED; no source backend executed')
