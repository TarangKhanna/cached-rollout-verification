# Cached rollout verification

Can a policy reuse a rollout that helped train it and still sample exactly from its updated distribution?

This project makes a known conditional-proposal problem concrete with small executable examples, exact probability calculations, and fresh/reserve-cache controls. The underlying dependence is prior work, including [Arnal et al., Appendix C.1](https://proceedings.mlr.press/v306/arnal26a.html). This is a technical exposition, not a new RL algorithm.

[Read the three-page note](manuscript.pdf) · [Worked examples](docs/WORKED_EXAMPLES.md)

## What the code checks

- **Sampling:** three controlled one-token examples approach expected-reward differences of **1/8**, **7/32**, and **7/64** compared with fresh samples from the same updated policy. Exact kernels and a separate calculation agree.
- **Learning:** exhaustive enumeration measures the next stochastic update in a four-parameter autoregressive model.
- **Controls:** included caches, independent reserve caches and fresh samples separate training dependence from an ordinary-target fallback mismatch.

The practical evidence is negative or inconclusive: larger-batch finite controls attenuate the effect, and Qwen statistics recomputed from saved metrics give an interval that includes zero. No neural training was rerun. This does **not** demonstrate degradation of released SPEC-RL or reproduce full-policy LLM training.

## Run

Python 3.12:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python code/worked_examples.py
bash run_checks.sh
```

Expected: **10 unit tests pass**, plus deterministic mathematical checks and checks of saved statistics. No GPU, model download or training run is required. This is an unofficial educational project, not the SPEC-RL implementation.

[Reproduction details](REPRODUCIBILITY.md) · [Provenance](PROVENANCE.md) · [AI assistance](AI_ASSISTANCE.md)

By Tarang Khanna. Parts of the derivations, code, analysis and writing were AI-developed; see the [disclosure](AI_ASSISTANCE.md). [MIT License](LICENSE).
