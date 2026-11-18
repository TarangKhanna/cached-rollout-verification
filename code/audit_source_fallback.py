"""Independent check of one-token ordinary-target fallback versus exact residual.
No published code is imported. This is a probability-kernel identity, not a
reproduction of released SPEC-RL training. Run with numpy.
"""
import numpy as np

rng = np.random.default_rng(20261117)
for n in (2, 3, 8, 16):
    for _ in range(500):
        q = rng.dirichlet(np.ones(n))
        p = rng.dirichlet(np.ones(n))
        h = rng.dirichlet(np.ones(n))
        a = np.minimum(1., p/q)
        z = 0.5 * np.abs(p-q).sum()
        r = np.maximum(p-q,0)/z if z > 0 else p
        k = np.diag(a) + (1-a)[:,None]*r[None,:]
        kt = np.diag(a) + (1-a)[:,None]*p[None,:]
        assert np.allclose(q@k,p,atol=2e-13)
        assert np.allclose(q@kt-p,z*(p-r),atol=2e-13)
        assert np.isclose(.5*np.abs(q@kt-p).sum(),z*.5*np.abs(p-r).sum(),atol=2e-13)
        assert np.allclose(h@kt-p,(h-q)@kt+z*(p-r),atol=2e-13)
        assert np.allclose(k.sum(axis=1),1,atol=2e-13)
q=np.array([.5,.5]); p=np.array([.75,.25]); a=np.minimum(1,p/q)
kt=np.diag(a)+(1-a)[:,None]*p[None,:]
print('PASS 2,000 randomized exact and ordinary fallback kernel checks')
print('q=(0.5,0.5), p=(0.75,0.25), ordinary qK=',(q@kt).tolist())
print('TV(ordinary,p)=',float(.5*np.abs(q@kt-p).sum()))
