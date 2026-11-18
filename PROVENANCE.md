# Provenance

This repository contains self-contained worked examples and preserved finite/readout evidence, not the official SPEC-RL implementation. The conditional-proposal problem is established prior art; see Arnal et al., Appendix C.1, and the note's references. Ordinary-target fallback mismatch was already discussed in SPEC-RL issue 14.

The reference finite implementation and its seven original tests were supplied by Tarang in the prior research archive. Separate audit implementations, the batch qualification and the saved-data verifier cross-check those results; this does not imply independent human replication. `code/worked_examples.py` adds explicit rational kernels and a separate formula/orbit calculation for the exposition. The saved readout verifier extracts only the statistics calculation from the original archive; model weights, prompts, features and inference code are not distributed.

The project was human-directed and developed with substantial AI assistance. Parts of the derivations, implementations, analysis and text were AI-developed. Distinct implementations are computational cross-checks, not independent human replication or peer review. The complete disclosure is in AI_ASSISTANCE.md; automated tests do not certify independent human derivation.

Tarang Khanna authorized public redistribution and MIT licensing. LICENSE covers the original repository material, including the exposition and figures, to the extent rights are held. It does not license the cited publications, official SPEC-RL implementation or model assets. No third-party implementation code or third-party figures are vendored. Copyright ownership/protectability of AI-assisted content is not certified by this record.
