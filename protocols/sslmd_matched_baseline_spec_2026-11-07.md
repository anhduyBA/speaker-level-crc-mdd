# SSL-MD matched baseline — specification v1, 07/11/2026

**Status: specified, not fitted.** This is a paper-informed adaptation to our target/alignment contract. It is not a reproduction of the published experiment and is not a novel architecture. The source-version audit makes old development usable under an explicit operational contract; it does not authenticate the final-test annotation release.

## Source recipe and availability

[Vidal, Riera and Ferrer (2023), §§3–4.3](https://arxiv.org/html/2307.16324) combine SSL encoder/transformer representations using learned weights, followed by a linear phone-output layer. MD uses sigmoid outputs and frame supervision from automatic alignments; phone scores average pre-activation frame outputs. Their SSL training uses AdamW, learning rate 1e-4 (1e-5 for Large), batch size64, with epochs selected by cross-validation. EpaDB uses30 development/20 test speakers; reported metrics are phone-macro1−AUC and ActCost, with speaker bootstrap. Their aligner is Kaldi/PyKaldi TDNN-F. Our below choices deliberately differ in backbone, labels, alignment, splits and primary endpoint.

Availability evidence: `research_plan_revision_2026_11_07/upstream_availability.json` pins the author's public SSL repository to `1b3f7794ef16cb190c1955cc5a3223992f523f16`; that snapshot has README only and an empty frozen-database link. GOP-FT source is separately pinned to `0a0abc30d6dde454cf7ab563bc07107b885580fa`. Do not claim these missing SSL recipe files were run.

## Fixed development scope

Use only the existing10 EpaDB development speakers/634 utterances. Reuse the five6-fit/2-cal/2-evaluation speaker folds in `epadb_matched_head_v1/locked_protocol.json`. Calibration/final speakers outside these10 are excluded from all extraction, fitting, tuning and stopping in this development study. Historical final20 stays exposed and is not reopened as a confirmation set.

Target inputs are existing gold-free `author_first` references and the current HuBERT39 inventory. Training labels are the existing all-path-certified canonical-position labels in `epadb_development_v1/certified_labels.json`, preserving every unknown/unsupported position as a mask. The source-regenerated flags from the forensic audit are **not** training data. Keep raw source reference/observation/flag versions distinct from mapped/certified targets. Record utterance and position coverage for all634 utterances, even when no supervised frame is eligible.

The frozen `coml/hubert-phoneme-classification` checkpoint is native-trained and already used by our controls: revision `9251d0fdc406d973ce1dbb7738ac6f7b124fbc65`, weight SHA256 `8c06c6a0e45bd6cfebfac66b2ac7a7111a692a96682c09d0cc715b6e710c868d`. This is not the paper's original s3prl pretrained upstream. All new acoustic arms share this same frozen encoder. The old raw/conditional79 results are context controls; a hidden-layer gain over posterior features alone is not architectural novelty.

## Arms and implementation contract

1. **Frame-weighted phone prior:** fit a phone-only intercept on the identical eligible frames/weights. Use a deterministic convex fit; do not manufacture five identical seed replicates. Keep the existing position-level phone-only control too, labelled with its different objective.
2. **Last transformer layer → linear39 MD:** one output per canonical phone, selected by gold-free reference identity, frame BCE with error=1. Encoder in eval mode, frozen; no waveform augmentation.
3. **Learned layer mix → identical linear39 MD:** softmax-normalized scalar mixture over12 transformer outputs, then the same head. The12-vs-paper encoder-plus-transformer mixture difference is explicit. No new projector/MLP is added. Match head initialization with arm2 within each seed. Report the12 extra scalars and any unavailable feature layer in the difference ledger.

Use the existing frozen automatic posterior alignment to associate canonical positions with frames; no TextGrid timing or manual phone sequence enters extraction/scoring. Keep hard-assignment and expected-occupancy semantics separate. Primary smoke implementation uses the existing hard alignment; zero-frame positions remain unsupported rather than receiving a made-up score. A frame shared between alternatives must not silently acquire conflicting supervision. Log a unique eligible frame/position map and validate it before training. Compare arms on identical support; publish each arm's abstentions.

MD target1 means error; the paper's correctness orientation is inverted explicitly. At inference average selected **logits** over each phone's eligible frames. Convert with sigmoid only where the predeclared threshold/calibration interface requires a bounded score. Never average sigmoid then label it average logit. BCE selects the reference-phone output only; other outputs do not receive fabricated negatives.

## Optimization and selection budget proposed for the smoke gate

Fixed seed list: `[1729,2718,31415,57721,65537]`. Use identical shuffled utterance order, paired initializations and supervised frame masks for arms2/3 per seed. Proposed optimizer: AdamW, lr1e-4, weight_decay0.01; maximum20epochs. CPU physical batch1 utterance, accumulation4, is a resource adaptation from the paper. Weight frames so each fit speaker has equal total contribution; no class weighting in the primary fit. Fit any feature centering/scaling using fit speakers only, identically across acoustic arms. Padding never contributes to the loss. No backbone updates.

For each fold/seed, use the two fold-cal speakers to choose one epoch by speaker-macro pAUC at maxFPR0.1; earliest epoch breaks ties. Those are empirical development selection/calibration scores, not CRC guarantees. If the budget cannot support all predeclared epochs/seeds/folds, amend the compute budget before evaluating the held fold; do not drop slow or weak seeds after results. No learning-rate or architecture search in this specification. A candidate with extra hidden layers requires a separate capacity-matched control before any superiority claim.

Use mean paired held-speaker differences against **both** phone prior and last-layer MD as the main development diagnostic. Report five seeds, every fold and unknown coverage; aggregate paired-seed scores before speaker bootstrap to avoid treating seeds as independent people. Existing conditional79 and SA remain comparators. Phone-macro ActCost is secondary under the existing declared support rule; it must not be compared numerically to the paper as if splits/targets matched. No test-optimal threshold is an operational endpoint.

## Next executable gate: smoke before fits

Select20 utterances by SHA256 ordering of `sslmd-smoke-v1|row_id` within the10 development speakers, with all634 eligible IDs declared before selection. This is a runtime/adapter check only, not performance-based selection. Check checkpoint hash,12 hidden tensors, frame lengths, padding masks, reference/frame map, output polarity and no manual-input dependency. A short synthetic gradient check must show zero encoder gradients and finite head/mix gradients. An alignment/target invariant failure stops the fit stage.

Measure elapsed time, process peak RAM, feature-cache bytes and available disk/RAM with other user apps left open. Extract one model at a time,4CPU threads. Prefer float32 cache for this initial check; estimate full634 storage from actual frames. Keep at least4GB system available RAM and10GB free disk after estimated cache; if not, choose streaming/chunking or a smaller predeclared budget before fitting. Those limits are resource rules, not scientific exclusion criteria.

Only after this check write a hashed run protocol with the measured compute budget and source/feature/label IDs. This document authorizes no claim that the smoke,50 neural fits (2arms×5folds×5seeds), fresh validation or full GOP-FT reproduction has already happened. A02 results cannot replace the fresh validation gate; absence of a stable acoustic gain leads to a narrower evaluation/robustness claim.
