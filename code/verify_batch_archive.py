"""Checks existing finite qualification; does not rerun the Monte Carlo experiment."""
from pathlib import Path
import hashlib,json
import numpy as np
import qualify_batch as ours
import independent_finite_reference as ref
root=Path(__file__).resolve().parents[1]/'data/batch'
for eta in [.05,.2]:
 for i,s in enumerate(ours.IDX):
  theta=ours.TH+eta*ours.OLDG[i]
  p,score,tok=ours.actor(theta);rp,rs,rt=ref.policy(theta)
  k,ell,_=ours.kernels(ours.qt,tok)
  np.testing.assert_allclose(p,rp,atol=1e-12,rtol=0)
  np.testing.assert_allclose(score,rs,atol=1e-12,rtol=0)
  np.testing.assert_allclose(k,ref.next_kernel(ours.qt,rt),atol=1e-12,rtol=0)
rows=[json.loads(x) for x in (root/'qualification.jsonl').read_text().splitlines()]
cells=[x['results'] for x in rows if x['type']=='cell']
assert len(cells)==16 and sum(c['independent_updates'] for c in cells)==16384
for c in cells:
 assert c['max_reserve_exact_error']<1e-10
 if c['B']>=32:assert c['relative_upper95']<.05
 m=np.array(c['gradient_cv']['mean']);se=np.array(c['gradient_cv']['se'])
 np.testing.assert_allclose(np.linalg.norm(np.abs(m)+2.734*se)/c['fresh_gradient_norm'],c['relative_upper95'],atol=1e-14)
print('PASS 1024 independent finite actor/score/kernel cross-checks')
print('PASS saved 16-cell / 16384-update qualification; all B>=32 upper bounds <5%')
for c in cells:
 if c['B']==128:print('B=128 eta=',c['eta'],'approximate upper95 percent=',100*c['relative_upper95'])
