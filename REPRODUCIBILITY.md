# Reproduction details

Use Python 3.12 and `pip install -r requirements.txt`, then `bash run_checks.sh`.

| Check | Coverage |
|---|---|
| `code/worked_examples.py` | Rational probability kernels versus separate formulas/collision orbits; literal REINFORCE softmax update; limits 1/8, 7/32 and 7/64 |
| Exact finite model and 10 unit tests | Three-token shared-parameter model; first-rejection kernel and expected reward after a second stochastic score update |
| Kernel/stability checks | 2,000 randomized kernel and 2,000 fallback cases; 23,520 rational row-sensitivity cases |
| `code/verify_batch_archive.py` | Independent finite actor/score/kernel checks and the saved 16-cell qualification |
| `code/verify_saved_readout.py` | All 24 archived ratio-bootstrap contrasts; saved statistics only |

Critical saved results agree with the note:

- Finite second-update fresh-minus-included reward: **0.0011106005**, first eta=.05, second step=.25; matched-filter gap **0.0003572267**.
- B=128 approximate per-cell 95% gradient-bound ratios: **0.0249% / 0.0848%** at eta=.05/.20. Estimated denominator; not an all-cell or source-faithful guarantee.
- Qwen frozen-readout projected contrast: **-1.4475%**, interval **[-3.3357%, 0.4515%]**, mean KL .0090465; raw estimate **+0.2612%**. Inconclusive, not an equivalence test.

The tests do not rerun the 16,384-update Monte Carlo sweep or the neural experiment. The readout check does not recreate inference, sampling histories, dense gradients or full-policy training. No verl/vLLM/AdamW/PPO parity, measured GPU/refill cost or learning-efficiency advantage is established.

To regenerate the retained finite-data figures:

```bash
pip install -r requirements-figures.txt
python code/plot_manuscript.py
```

Build the note with separately installed Tectonic 0.17.0: `tectonic --keep-logs manuscript.tex`. Its first build may download standard TeX fonts/styles. These optional plotting dependencies are pinned to the previously verified environment. PDF bytes can vary with timestamps; numerical outputs are the reproduction target.

Inspected official source pin: `97da99ed54d663b238c417c728bc824c79cfab6f`. Static inspection is not an executable source-faithful result. The notebook-free scripts are independent implementations; no official SPEC-RL code or model assets are included.
