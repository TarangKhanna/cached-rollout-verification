"""Deterministic regression checks for the exposition's finite illustrations."""
import sys
from pathlib import Path
from fractions import Fraction as F
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import worked_examples as ex

class WorkedExamples(unittest.TestCase):
    def test_reinforce_dense_and_finite_expression(self):
        for n in (2,4,8): self.assertEqual(ex.reinforce_dense(n),ex.reinforce_formula(n))
        self.assertEqual(ex.reinforce_dense(2),0)
        self.assertEqual(ex.reinforce_dense(4),F(2,35))
    def test_grpo_kernels_and_collision_orbits(self):
        for fallback in ('residual','ordinary'):
            self.assertEqual(ex.grpo_dense(4,fallback),ex.grpo_orbits(4,fallback))
    def test_independent_reserve_exactness_and_fallback_difference(self):
        q=[F(1,2)]*2;p=[F(3,4),F(1,4)]
        k=ex.kernel(q,p);ordinary=ex.kernel(q,p,'ordinary')
        self.assertEqual([sum(q[y]*k[y][x] for y in range(2)) for x in range(2)],p)
        self.assertEqual([sum(q[y]*ordinary[y][x] for y in range(2)) for x in range(2)],[F(11,16),F(5,16)])

if __name__=='__main__': unittest.main()
