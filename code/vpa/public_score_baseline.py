"""Locked public-gold phone-score regression; not MDD or Vietnamese evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


METHODS = ("global_train_mean", "reference_phone_train_mean", "ridge_gop84_plus_reference_phone_onehot")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def unique_utterance_order(keys: list[str]) -> list[str]:
    output, seen = [], set()
    last = None
    for key in keys:
        utterance, position = key.rsplit(".", 1)
        if not position.isdigit():
            raise ValueError("invalid phone index")
        if utterance != last:
            if utterance in seen:
                raise ValueError("utterance keys must be contiguous")
            output.append(utterance)
            seen.add(utterance)
            last = utterance
    return output


def prepare_split(root: Path, prefix: str, official_split: str, scores: dict,
                  verify_raw_csv: bool = True) -> tuple[dict, dict]:
    feature_path = root / f"gopt_features/seq_data_librispeech/{prefix}_feat.npy"
    label_path = root / f"gopt_features/seq_data_librispeech/{prefix}_label_phn.npy"
    keys_path = root / f"gopt_features/raw_kaldi_gop/librispeech/{prefix}_keys_phn.csv"
    x = np.load(feature_path, allow_pickle=False)
    label = np.load(label_path, allow_pickle=False)
    if x.shape != (2500, 50, 84) or label.shape != (2500, 50, 2):
        raise ValueError("unexpected published feature schema")
    keys = keys_path.read_text(encoding="utf-8").splitlines()
    utterances = unique_utterance_order(keys)
    speakers = dict(line.split() for line in (root / f"official/{official_split}/utt2spk").read_text().splitlines())
    if len(utterances) != 2500 or set(utterances) != set(speakers):
        raise ValueError("feature utterance IDs do not match the official split")
    mask = label[:, :, 1] >= 0
    reference_mapping = defaultdict(set)
    flat_utt, flat_spk, flat_index, official_gold, symbols = [], [], [], [], []
    for index, utterance in enumerate(utterances):
        gold = [float(v) for word in scores[utterance]["words"] for v in word["phones-accuracy"]]
        phones = [re.sub(r"[012]$", "", phone) for word in scores[utterance]["words"] for phone in word["phones"]]
        valid_positions = np.flatnonzero(mask[index])
        if len(gold) != len(valid_positions) or not np.array_equal(valid_positions, np.arange(len(gold))):
            raise ValueError("phone counts or padding do not match official annotations")
        if not np.allclose(label[index, valid_positions, 1], gold, atol=1e-12, rtol=0):
            raise ValueError("provided scores differ from original official expert means")
        for position, phone in enumerate(phones):
            reference_mapping[int(label[index, position, 0])].add(phone)
            flat_utt.append(utterance)
            flat_spk.append(speakers[utterance])
            flat_index.append(position)
        official_gold.extend(gold)
        symbols.extend(phones)
    if any(len(v) != 1 for v in reference_mapping.values()):
        raise ValueError("reference phone ID does not map one-to-one to canonical phone symbols")
    array = x[mask].astype(np.float64)
    phone_ids = label[:, :, 0][mask].astype(int)
    if not np.isfinite(array).all() or not np.isfinite(official_gold).all():
        raise ValueError("nonfinite features or scores")
    # The raw CSV is [reference phone identity, 84 acoustic GOP features].
    # Verify that seq_data stores those feature columns unchanged, without author normalization.
    raw_equal = None
    if verify_raw_csv:
        with zipfile.ZipFile(root / "gopt_data.zip") as archive:
            with archive.open(f"raw_kaldi_gop/librispeech/{prefix}_feats.csv") as stream:
                raw_features = np.loadtxt(stream, delimiter=",")
        if raw_features.shape != (len(array), 85):
            raise ValueError("raw GOP CSV has unexpected shape")
        raw_equal = bool(np.array_equal(array, raw_features[:, 1:]))
        if not raw_equal:
            raise ValueError("sequence features were changed from raw acoustic GOP columns")
        if not np.array_equal(np.array(flat_utt), np.array([k.rsplit(".", 1)[0] for k in keys])):
            raise ValueError("flattened features do not match raw key row order")
        if not np.array_equal(np.array(flat_index), np.array([int(k.rsplit(".", 1)[1]) for k in keys])):
            raise ValueError("flattened phone positions do not match raw key order")
    data = {"features": array, "reference_phone_id": phone_ids, "scores": np.asarray(official_gold),
            "utterance_ids": np.asarray(flat_utt), "speaker_ids": np.asarray(flat_spk),
            "phone_indices": np.asarray(flat_index), "canonical_symbols": np.asarray(symbols)}
    integrity = {"n_utterances": len(utterances), "n_speakers": len(set(speakers.values())),
                 "n_valid_phones": len(array), "feature_dimension": array.shape[1],
                 "n_original_zero_scores": int(np.sum(data["scores"] == 0)),
                 "valid_scores_exactly_match_official_means": True,
                 "sequence_features_equal_raw_csv_acoustic_columns": raw_equal,
                 "reference_phone_mapping": {str(k): next(iter(v)) for k, v in sorted(reference_mapping.items())},
                 "feature_sha256": sha256(feature_path), "label_sha256": sha256(label_path),
                 "keys_sha256": sha256(keys_path)}
    return data, integrity


def fit_predict(train: dict, query: dict) -> tuple[dict, dict]:
    target = train["scores"]
    mean = float(target.mean())
    phone_means = {int(phone): float(target[train["reference_phone_id"] == phone].mean())
                   for phone in np.unique(train["reference_phone_id"])}
    scaler = StandardScaler().fit(train["features"])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64).fit(train["reference_phone_id"][:, None])
    design = np.concatenate((scaler.transform(train["features"]), encoder.transform(train["reference_phone_id"][:, None])), axis=1)
    query_design = np.concatenate((scaler.transform(query["features"]), encoder.transform(query["reference_phone_id"][:, None])), axis=1)
    model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky").fit(design, target)
    raw_ridge = model.predict(query_design)
    predictions = {METHODS[0]: np.full(len(query["scores"]), mean),
                   METHODS[1]: np.array([phone_means.get(int(p), mean) for p in query["reference_phone_id"]]),
                   METHODS[2]: np.clip(raw_ridge, 0, 2)}
    state = {"global_train_mean": mean, "phone_ids": np.array(sorted(phone_means)),
             "phone_means": np.array([phone_means[p] for p in sorted(phone_means)]),
             "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
             "encoder_categories": encoder.categories_[0], "ridge_coef": model.coef_,
             "ridge_intercept": model.intercept_,
             "n_raw_ridge_predictions_outside_score_bounds": int(np.sum((raw_ridge < 0) | (raw_ridge > 2)))}
    return predictions, state


def subset(data: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in data.items()}


def pcc(gold: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(gold) < 2 or np.ptp(gold) == 0 or np.ptp(predicted) == 0:
        return None
    return float(np.corrcoef(gold, predicted)[0, 1])


def metrics(data: dict, predicted: np.ndarray) -> dict:
    gold = data["scores"]
    by_speaker = [{"speaker_id": speaker, "n_phones": int(np.sum(data["speaker_ids"] == speaker)),
                   "mae": float(np.abs(gold[data["speaker_ids"] == speaker] - predicted[data["speaker_ids"] == speaker]).mean())}
                  for speaker in sorted(set(data["speaker_ids"]))]
    return {"n_phones": len(gold), "n_speakers": len(by_speaker),
            "pooled_phone_pearson": pcc(gold, predicted),
            "pooled_phone_spearman": float(spearmanr(gold, predicted).statistic) if np.ptp(predicted) else None,
            "phone_mae": float(np.abs(gold - predicted).mean()),
            "phone_rmse": float(np.sqrt(np.mean((gold - predicted) ** 2))),
            "speaker_macro_mae": float(np.mean([r["mae"] for r in by_speaker])),
            "by_speaker": by_speaker}


def bootstrap(data: dict, predictions: dict) -> dict:
    speakers = sorted(set(data["speaker_ids"]))
    draws = np.random.default_rng(1729).integers(0, len(speakers), size=(2000, len(speakers)))
    samples = {}
    for name in (METHODS[1], METHODS[2]):
        y, p = data["scores"], predictions[name]
        sufficient, speaker_mae = [], []
        for speaker in speakers:
            mask = data["speaker_ids"] == speaker
            a, b = y[mask], p[mask]
            sufficient.append([len(a), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])
            speaker_mae.append(np.abs(a - b).mean())
        sums = np.asarray(sufficient)[draws].sum(axis=1)
        n, sy, sp, syy, spp, syp = sums.T
        corr = (syp - sy * sp / n) / np.sqrt((syy - sy * sy / n) * (spp - sp * sp / n))
        mae = np.asarray(speaker_mae)[draws].mean(axis=1)
        samples[name] = {"pooled_phone_pearson": corr, "speaker_macro_mae": mae}
    output = {"unit": "official test speaker", "n_speakers": len(speakers), "n_resamples": 2000,
              "seed": 1729, "interval": "percentile95", "scope": "fixed model predictions, not training randomness",
              "methods": {}, "ridge_minus_phone_mean": {}}
    for method, values in samples.items():
        output["methods"][method] = {key: np.quantile(value, [0.025, 0.975]).tolist() for key, value in values.items()}
    for key in ("pooled_phone_pearson", "speaker_macro_mae"):
        difference = samples[METHODS[2]][key] - samples[METHODS[1]][key]
        output["ridge_minus_phone_mean"][key] = {"ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
                                                "point_difference": metrics(data, predictions[METHODS[2]])[key] - metrics(data, predictions[METHODS[1]])[key]}
    return output


def run(args: argparse.Namespace) -> dict:
    if (args.output_dir / "summary.json").exists():
        raise ValueError("final public test results already exist; use stored outputs rather than rerunning/selecting on test")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["ridge"]["alpha"] != 1.0 or protocol["test_metrics_accessed_when_locked"]:
        raise ValueError("unexpected or already-test-informed protocol")
    root, started = args.data_root, time.perf_counter()
    if sha256(root / "gopt_data.zip") != protocol["archive_sha256"]:
        raise ValueError("public archive changed")
    scores = json.loads((root / "official/resource/scores.json").read_text(encoding="utf-8"))
    train, train_integrity = prepare_split(root, "tr", "train", scores)
    test, test_integrity = prepare_split(root, "te", "test", scores)
    if set(train["speaker_ids"]) & set(test["speaker_ids"]) or set(train["utterance_ids"]) & set(test["utterance_ids"]):
        raise ValueError("official train/test overlap")
    if train_integrity["reference_phone_mapping"] != test_integrity["reference_phone_mapping"]:
        raise ValueError("reference phone map changes between splits")
    validation_ids = set(protocol["splits"]["validation_speaker_ids"])
    val_mask = np.isin(train["speaker_ids"], list(validation_ids))
    fit, validation = subset(train, ~val_mask), subset(train, val_mask)
    if len(set(fit["speaker_ids"])) != 100 or len(set(validation["speaker_ids"])) != 25:
        raise ValueError("train-only validation split changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "data_integrity.json", {"train": train_integrity, "test": test_integrity,
              "official_speaker_overlap": 0, "official_utterance_overlap": 0,
              "feature_provenance": "all84 NPY acoustic columns equal raw Kaldi GOP CSV columns1: exactly; reference phone ID is appended separately, no expert-score feature column",
              "upstream_limit": "exact Kaldi acoustic-model binary/version and its preprocessing are not pinned by supplied archive; no author global normalization used here"})
    validation_predictions, _ = fit_predict(fit, validation)
    validation_metrics = {name: metrics(validation, value) for name, value in validation_predictions.items()}
    save_json(args.output_dir / "validation_results.json", {"fixed_alpha": 1.0, "selection_performed": False,
              "n_fit_speakers": 100, "n_validation_speakers": 25, "metrics": validation_metrics})
    print(json.dumps({"phase": "validation_complete", "alpha_unchanged": 1.0}), flush=True)
    # Final test is first predicted here, after fixed train-only validation.
    test_predictions, fitted_state = fit_predict(train, test)
    test_metrics = {name: metrics(test, value) for name, value in test_predictions.items()}
    intervals = bootstrap(test, test_predictions)
    np.savez_compressed(args.private_dir / "test_predictions.npz", scores=test["scores"],
                        speaker_ids=test["speaker_ids"], utterance_ids=test["utterance_ids"],
                        phone_indices=test["phone_indices"], reference_phone_id=test["reference_phone_id"],
                        **test_predictions)
    np.savez_compressed(args.private_dir / "fitted_ridge_state.npz", **fitted_state)
    results = {"date_local": datetime.now(timezone(timedelta(hours=7))).date().isoformat(),
               "scope": "public expert phoneme accuracy score regression; not MDD, HuBERT or Vietnamese performance",
               "protocol_sha256": sha256(args.protocol), "implementation_sha256": sha256(Path(__file__)),
               "n_official_train_speakers": 125, "n_official_test_speakers": 125,
               "n_official_train_utterances": 2500, "n_official_test_utterances": 2500,
               "n_train_phone_scores": len(train["scores"]), "n_test_phone_scores": len(test["scores"]),
               "reference_phone_classes": 39, "gop_feature_dimension": 84,
               "ridge_alpha": 1.0, "hyperparameter_selection_performed": False,
               "n_final_test_prediction_passes": 1,
               "validation_metrics": validation_metrics, "test_metrics": test_metrics,
               "speaker_bootstrap": intervals,
               "n_raw_ridge_test_predictions_outside_bounds": fitted_state["n_raw_ridge_predictions_outside_score_bounds"],
               "wall_seconds": time.perf_counter() - started,
               "downloaded_audio": False, "new_inhouse_gold_labels_generated": False,
               "limits": ["The gold corpus is read English by Mandarin L1 speakers, not Vietnamese code-switched technical answers.",
                          "The target is expert phone accuracy score0–2; no binary phone-error threshold or MDD F1 is claimed.",
                          "This is a fixed ridge baseline on author GOP features, not a reproduction of the GOPT Transformer.",
                          "Official test is speaker-disjoint; bootstrap conditions on the fitted model and fixed public test corpus."]}
    save_json(args.output_dir / "summary.json", results)
    print(json.dumps({"phase": "test_complete", "wall_seconds": results["wall_seconds"],
                      "test_metrics": {name: {key: value for key, value in m.items() if key != "by_speaker"} for name, m in test_metrics.items()}}, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
