#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONDONTWRITEBYTECODE=1
python code/worked_examples.py
python code/audit_kernel_and_formulae.py
python code/check_review_claims.py
python code/audit_source_fallback.py
python code/independent_finite_reference.py
python code/audit_ratio_diagnostics.py
python code/spec_rl_gradient_extension.py
python code/reference_exact_shared_policy.py --test --output "${TMPDIR:-/tmp}/rollout_reuse_reference.json"
python -m unittest discover -s tests -v
python code/verify_batch_archive.py
python code/verify_saved_readout.py
python code/verify_figure_data.py
python code/validate_release.py
