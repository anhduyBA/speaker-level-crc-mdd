# Speaker-level conformal risk control for mispronunciation warnings: release package

This package accompanies the ICCIES 2027 submission *Speaker-Level Conformal Risk Control for Mispronunciation Warnings: A Cross-Corpus Study with Frozen HuBERT Features* (Bui Anh Duy, Truong Long).

It contains the analysis code, the written protocols and the locked result summaries. **It contains no audio, labels or model checkpoints.** The corpora must be obtained from their original distributors under their own licences.

The results were produced by the authors with this code. They have **not** been reproduced independently.

## Contents
| Path | What it is |
|---|---|
| `code/vpa/` | All modules needed by the reported experiments (training, scoring, CRC calibration, analyses) |
| `code/tests/` | 60 unit tests; run with `cd code && python -m unittest discover -s tests` |
| `protocols/` | Protocols written before each run (contracts and pre-declared analysis notes) |
| `results/<study>/locked_protocol.json` | Hash-locked protocol of each run (data, checkpoints, code, analysis plan) |
| `results/<study>/summary.json`, `findings_*.md` | Reported numbers and short findings |
| `make_figures.py`, `framework.html` | Figures 2–3 and the Fig. 1 design diagram |
| `MANIFEST.sha256` | SHA-256 of every file in this package |

Main locked protocols (SHA-256):
- **EpaDB convergence study (v4):** `3cced4638896aa078554fd09ba193a1f3dfc4ca52111b01357e31039119b1c22`
- **L2-ARCTIC CRC study:** `97309bf74675413a8dd0a1ca699bd11e511cca3548ce3a89a466274608b1d1fa`

## External resources
- **HuBERT phone classifier:** `coml/hubert-phoneme-classification`, revision `9251d0fdc406d973ce1dbb7738ac6f7b124fbc65` (Poli et al., EMNLP 2024).
- **EpaDB:** Hugging Face `KoelLabs/EpaDB`, revision `52404544712cdd5f45a020a1a86dfe27d5c9cbd8` (Vidal et al., Interspeech 2019; CC BY-NC 4.0). Only 10 development speakers are used.
- **L2-ARCTIC v5.0:** https://psi.engr.tamu.edu/l2-arctic-corpus/ (Zhao et al., Interspeech 2018; CC BY-NC 4.0).

## Run order (from `code/`)
The paths in the modules assume the original repository layout. Raw data goes under `data/raw` and `data/public`; private intermediates are written to `data/derived`.

1. **EpaDB heads.**
   - Run the SSL-MD v2 feature extraction and training (`vpa.sslmd_training_v2`) and v3 (`vpa.sslmd_convergence_v3`). Their protocols are in `protocols/`.
   - Then run `python -m vpa.sslmd_convergence_v4 lock`, then `fit`, then `evaluate`.
2. **Exploratory EpaDB CRC:** `python -m vpa.sslmd_risk_epadb_v1`
3. **L2-ARCTIC:** `python -m vpa.sslmd_l2arctic_crc_v1 labels`, then `lock`, then `score`, then `evaluate`.
4. **Descriptive and sensitivity analyses:**
   - `python -m vpa.sslmd_l2arctic_phone_analysis_v1`
   - `python -m vpa.sslmd_l2arctic_phone_posthoc_v1`
   - `python -m vpa.sslmd_l2arctic_sensitivity_v1`

Each analysis first checks that it reproduces the locked thresholds and means, and stops if it does not.

## Environment used
- CPU only, 4 threads.
- Python 3.12, torch 2.12.0+cpu, transformers 5.12.0, numpy 2.2.6, scipy 1.17.1, scikit-learn 1.8.0. See `requirements.txt`.
- Seeds: 1729, 2718, 31415, 57721, 65537. Bootstrap seed: 3191. Subsampling seed: 4242.
- Training the 50 EpaDB heads took about 22,600 s in total; scoring L2-ARCTIC took 990 s.

## Scope and limitations
- The CRC guarantee is marginal over exchangeable speakers, not a per-speaker cap.
- The EpaDB study is retrospective on 10 speakers.
- Each L2-ARCTIC utterance has a single annotator.
- Insertions are not scored, and alignment is automatic.
- Sensitivity and per-phone analyses are labelled as such; they are not pre-registered endpoints.
