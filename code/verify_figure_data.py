"""Compare stored finite figure points to an independently enumerated model."""
from pathlib import Path
import json
import numpy as np
import independent_finite_reference as ref
root=Path(__file__).resolve().parents[1]
for row in json.loads((root/'data/finite_exact_values.json').read_text()):
 out=ref.case(row['eta'])
 for field,source in [('fresh_second_minus_included_second','fresh_minus_reuse_after2'),('matched_filter_two_step_fresh_advantage','actor_matched_mixed_only_gap'),('policy_kl_qp','kl')]:
  np.testing.assert_allclose(row[field],out[source],atol=1e-12,rtol=1e-10)
 np.testing.assert_allclose(row['included_group_yield']/row['fresh_group_yield'],out['mixed_ratio'],atol=1e-12)
print('PASS all five exact finite figure-data points against independent enumeration')
