"""Exploratory speaker-level conformal intervals/CRC on official TRAIN only.

The official TEST has already been observed by another study. This module never
prepares its feature split, evaluates it, or reuses the old fitted Ridge state.
"""
from __future__ import annotations

import os
for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "4"

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .public_score_baseline import METHODS, fit_predict, metrics, prepare_split, save_json, sha256, subset


def valid_phone_mask(labels: np.ndarray) -> np.ndarray:
    """Only -1 is padding; zero is a real target, never an empty-feature proxy."""
    values = np.asarray(labels)[..., 1]
    if not np.isfinite(values).all() or np.any((values < 0) & (values != -1)) or np.any(values > 2):
        raise ValueError("score must be -1 padding or in [0,2]")
    return values >= 0


def speaker_split(speaker_ids: np.ndarray, seed: int = 1729,
                  n_fit: int = 75, n_calibration: int = 25) -> dict:
    speakers = np.asarray(sorted(set(np.asarray(speaker_ids).tolist())))
    if n_fit < 1 or n_calibration < 1 or n_fit + n_calibration >= len(speakers):
        raise ValueError("need nonempty fit, calibration, and evaluation sets")
    order = np.random.default_rng(seed).permutation(speakers)
    return {"fit": sorted(order[:n_fit].tolist()),
            "calibration": sorted(order[n_fit:n_fit + n_calibration].tolist()),
            "evaluation": sorted(order[n_fit + n_calibration:].tolist())}


def conformal_quantile(scores: np.ndarray, alpha: float = .1, bound: float = 2.) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not (0 < alpha < 1):
        raise ValueError("nonempty one-dimensional scores and alpha in (0,1) required")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > bound)):
        raise ValueError("nonconformity scores outside known bounds")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return (float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float(bound), rank)


def score_intervals(predicted: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 2)) or not (0 <= q <= 2):
        raise ValueError("predictions/q outside score bounds")
    return np.maximum(0, predicted - q), np.minimum(2, predicted + q)


def warning_rows(gold: np.ndarray, priority: np.ndarray, speakers: np.ndarray, threshold: float) -> list[dict]:
    gold, priority, speakers = np.asarray(gold), np.asarray(priority), np.asarray(speakers)
    if not (gold.shape == priority.shape == speakers.shape) or not len(gold):
        raise ValueError("nonempty matched vectors required")
    if (not np.isfinite(gold).all() or not np.isfinite(priority).all()
            or np.any((gold < 0) | (gold > 2)) or np.any((priority < 0) | (priority > 1))):
        raise ValueError("gold/priority outside bounds")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold outside [0,1]")
    rows = []
    for speaker in sorted(set(speakers.tolist())):
        mask = speakers == speaker
        acceptable = gold[mask] == 2
        flag = priority[mask] > threshold  # Equality abstains: required for right continuity.
        a, b = int(acceptable.sum()), int((~acceptable).sum())
        fp, tp, n_flag = int((flag & acceptable).sum()), int((flag & ~acceptable).sum()), int(flag.sum())
        rows.append({"speaker_id": str(speaker), "n_phones": int(mask.sum()),
                     "aggregate_score_2": a, "aggregate_score_below_2": b,
                     "flags": n_flag, "false_flags": fp, "true_flags_score_derived": tp,
                     "crc_loss": fp / a if a else 0.,
                     "false_fraction_among_flags": fp / n_flag if n_flag else None,
                     "full_recall_score_below_2": tp / b if b else None})
    return rows


def warning_metrics(rows: list[dict]) -> dict:
    totals = {k: sum(r[k] for r in rows) for k in
              ("n_phones", "aggregate_score_2", "aggregate_score_below_2", "flags", "false_flags", "true_flags_score_derived")}
    out = dict(totals)
    out.update({"n_speakers": len(rows),
                "n_speakers_without_score2": sum(r["aggregate_score_2"] == 0 for r in rows),
                "speaker_macro_crc_loss": float(np.mean([r["crc_loss"] for r in rows])),
                "pooled_false_flag_rate_on_score2": totals["false_flags"] / totals["aggregate_score_2"] if totals["aggregate_score_2"] else None,
                "false_fraction_among_flags_NOT_controlled": totals["false_flags"] / totals["flags"] if totals["flags"] else None,
                "precision_score_derived_NOT_controlled": totals["true_flags_score_derived"] / totals["flags"] if totals["flags"] else None,
                "full_recall_score_below_2": totals["true_flags_score_derived"] / totals["aggregate_score_below_2"] if totals["aggregate_score_below_2"] else None,
                "flag_coverage": totals["flags"] / totals["n_phones"],
                "abstention_coverage": 1 - totals["flags"] / totals["n_phones"]})
    return out


def calibrate_crc(gold: np.ndarray, priority: np.ndarray, speakers: np.ndarray,
                  target: float = .1, tolerance: float = 1e-12) -> dict:
    """Exact monotone finite candidate search; denominator is fixed per speaker."""
    # Validate through the reference implementation, retaining no-positive speakers.
    at_one = warning_rows(gold, priority, speakers, 1.)
    n = len(at_one)
    if not 0 < target < 1 or not 0 <= tolerance <= 1e-10:
        raise ValueError("invalid risk target/tolerance")
    gold, priority, speakers = np.asarray(gold), np.asarray(priority), np.asarray(speakers)
    acceptable = gold == 2
    denominators = {r["speaker_id"]: r["aggregate_score_2"] for r in at_one}
    weights = np.array([1 / denominators[str(s)] for s in speakers[acceptable]])
    order = np.argsort(priority[acceptable], kind="stable")
    sorted_p, sorted_weights = priority[acceptable][order], weights[order]
    # Tail at searchsorted(..., side='right') excludes values equal to threshold.
    tail = np.r_[np.cumsum(sorted_weights[::-1])[::-1], 0.]
    candidates = np.unique(np.r_[0., priority, 1.])
    correction = (tail[np.searchsorted(sorted_p, candidates, side="right")] + 1) / (n + 1)
    passing = np.flatnonzero(correction <= target + tolerance)
    threshold = float(candidates[passing[0]]) if len(passing) else 1.
    rows = warning_rows(gold, priority, speakers, threshold)
    corrected = (sum(r["crc_loss"] for r in rows) + 1) / (n + 1)
    if len(passing) and corrected > target + tolerance:
        raise AssertionError("optimized search disagrees with direct speaker loss")
    return {"threshold": threshold, "n_calibration_speakers": n,
            "target": target, "bound": 1., "finite_sample_correction": 1 / (n + 1),
            "corrected_empirical_risk": corrected,
            "empirical_macro_loss": float(np.mean([r["crc_loss"] for r in rows])),
            "n_candidates": len(candidates), "empty_feasible_set_fallback": not len(passing),
            "calibration_rows": rows}


def describe(values: list) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n_defined": 0, "mean": None, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    points = np.quantile(values, [0, .25, .5, .75, 1])
    return {"n_defined": len(values), "mean": float(values.mean()),
            **dict(zip(("min", "q25", "median", "q75", "max"), map(float, points)))}


def interval_evaluation(gold: np.ndarray, predicted: np.ndarray, speakers: np.ndarray, q: float) -> tuple[dict, list[dict]]:
    lower, upper = score_intervals(predicted, q)
    covered = np.abs(gold - predicted) <= q  # Inclusive nonconformity comparison preserves ties.
    rows = []
    for s in sorted(set(speakers.tolist())):
        m = speakers == s
        rows.append({"speaker_id": str(s), "n_phones": int(m.sum()),
                     "n_covered_phones": int(covered[m].sum()),
                     "all_sampled_phones_covered": bool(covered[m].all()),
                     "phone_coverage": float(covered[m].mean()),
                     "mean_interval_width": float((upper[m] - lower[m]).mean()),
                     "sum_interval_width": float((upper[m] - lower[m]).sum()),
                     "sum_absolute_error": float(np.abs(gold[m] - predicted[m]).sum()),
                     "sum_squared_error": float(((gold[m] - predicted[m]) ** 2).sum()),
                     "max_absolute_error": float(np.abs(gold[m] - predicted[m]).max())})
    return {"q": q, "n_speakers": len(rows), "n_phones": len(gold),
            "simultaneously_covered_speakers": sum(r["all_sampled_phones_covered"] for r in rows),
            "simultaneous_speaker_coverage": float(np.mean([r["all_sampled_phones_covered"] for r in rows])),
            "pooled_phone_coverage": float(covered.mean()),
            "speaker_macro_phone_coverage": float(np.mean([r["phone_coverage"] for r in rows])),
            "mean_interval_width": float((upper - lower).mean()),
            "phone_interval_width_distribution": describe((upper - lower).tolist()),
            "speaker_mean_interval_width_distribution": describe([r["mean_interval_width"] for r in rows])}, rows


def evaluation_bootstrap(interval_rows: list[dict], crc_rows: list[dict],
                         draws: int = 2000, seed: int = 1729) -> dict:
    if [r["speaker_id"] for r in interval_rows] != [r["speaker_id"] for r in crc_rows]:
        raise ValueError("bootstrap rows must match speakers")
    ix = np.random.default_rng(seed).integers(0, len(interval_rows), size=(draws, len(interval_rows)))
    def sums(rows, key):
        return np.array([r[key] for r in rows], dtype=float)[ix].sum(axis=1)
    def means(rows, key):
        return sums(rows, key) / len(rows)
    def ratios(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    n = sums(interval_rows, "n_phones")
    values = {"phone_mae": sums(interval_rows, "sum_absolute_error") / n,
              "phone_rmse": np.sqrt(sums(interval_rows, "sum_squared_error") / n),
              "speaker_macro_mae": np.array([r["sum_absolute_error"] / r["n_phones"] for r in interval_rows])[ix].mean(axis=1),
              "simultaneous_speaker_coverage": means(interval_rows, "all_sampled_phones_covered"),
              "pooled_phone_coverage": sums(interval_rows, "n_covered_phones") / n,
              "speaker_macro_phone_coverage": means(interval_rows, "phone_coverage"),
              "mean_interval_width": sums(interval_rows, "sum_interval_width") / n,
              "crc_speaker_macro_loss": means(crc_rows, "crc_loss"),
              "crc_pooled_false_flag_rate_on_score2": ratios(sums(crc_rows, "false_flags"), sums(crc_rows, "aggregate_score_2")),
              "false_fraction_among_flags_NOT_controlled": ratios(sums(crc_rows, "false_flags"), sums(crc_rows, "flags")),
              "full_recall_score_below_2": ratios(sums(crc_rows, "true_flags_score_derived"), sums(crc_rows, "aggregate_score_below_2")),
              "flag_coverage": sums(crc_rows, "flags") / n}
    cis = {}
    for k, v in values.items():
        finite = v[np.isfinite(v)]
        cis[k] = {"n_defined_resamples": len(finite),
                  "percentile95": [float(x) for x in np.quantile(finite, [.025, .975])] if len(finite) else None}
    return {"draws": draws, "seed": seed, "unit": "evaluation speaker",
            "scope": "descriptive fixed model/calibration; not prospective conformal certification or training/calibration uncertainty",
            "intervals": cis}


def run(args: argparse.Namespace) -> dict:
    start = time.monotonic()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol)
    if args.private_dir.exists() and (args.private_dir / "predictions.npz").exists():
        raise RuntimeError("Refusing to overwrite an observed run; verify saved predictions instead")
    for rel, expected in protocol["input_sha256"].items():
        if sha256(args.data_root / rel) != expected:
            raise ValueError(f"locked input hash mismatch: {rel}")
    if protocol["models"]["alpha"] != 1 or protocol["conformal"]["alpha"] != .1 or protocol["crc"]["target"] != .1:
        raise ValueError("This implementation only supports the locked alpha1 / risk.10 protocol")
    # scores.json is a shared official file. Only TRAIN keys are selected or scored.
    train_utterances = {line.split()[0] for line in (args.data_root / "official/train/utt2spk").read_text().splitlines()}
    source_scores = json.loads((args.data_root / "official/resource/scores.json").read_text(encoding="utf-8"))
    train_scores = {k: source_scores[k] for k in train_utterances}
    del source_scores
    data, integrity = prepare_split(args.data_root, "tr", "train", train_scores, verify_raw_csv=True)
    raw_labels = np.load(args.data_root / "gopt_features/seq_data_librispeech/tr_label_phn.npy", allow_pickle=False)
    if int(valid_phone_mask(raw_labels).sum()) != len(data["scores"]):
        raise AssertionError("valid zero/padding handling disagrees with upstream validator")
    split = speaker_split(data["speaker_ids"], protocol["split"]["seed"])
    if any(split[k] != protocol["split"][k] for k in split):
        raise ValueError("speaker IDs disagree with pre-calibration lock")
    if len(set(data["speaker_ids"])) != 125:
        raise ValueError("unexpected official TRAIN speaker count")
    args.private_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.private_dir / "speaker_splits.json", split)
    masks = {k: np.isin(data["speaker_ids"], v) for k, v in split.items()}
    if not np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1):
        raise AssertionError("speaker partitions overlap or omit phones")
    fit = subset(data, masks["fit"])
    query = subset(data, ~masks["fit"])
    with threadpool_limits(limits=4):
        predictions, state = fit_predict(fit, query)
    predicted = predictions[METHODS[2]]
    if len(state["encoder_categories"]) != 39:
        raise ValueError("fit partition does not contain the locked 39 reference phone IDs")
    np.savez_compressed(args.private_dir / "fitted_ridge_state.npz", **state)
    cal_mask = np.isin(query["speaker_ids"], split["calibration"])
    eval_mask = np.isin(query["speaker_ids"], split["evaluation"])
    cal, evaluation = subset(query, cal_mask), subset(query, eval_mask)
    cal_pred, eval_pred = predicted[cal_mask], predicted[eval_mask]
    calibration_rows = [{"speaker_id": s, "n_phones": int((cal["speaker_ids"] == s).sum()),
                         "max_absolute_error": float(np.abs(cal["scores"][cal["speaker_ids"] == s] - cal_pred[cal["speaker_ids"] == s]).max())}
                        for s in split["calibration"]]
    q, rank = conformal_quantile(np.array([r["max_absolute_error"] for r in calibration_rows]), .1)
    crc = calibrate_crc(cal["scores"], 1 - cal_pred / 2, cal["speaker_ids"], .1, protocol["crc"]["comparison_tolerance"])
    crc_cal_rows = crc.pop("calibration_rows")
    primary, interval_rows = interval_evaluation(evaluation["scores"], eval_pred, evaluation["speaker_ids"], q)
    trivial, _ = interval_evaluation(evaluation["scores"], eval_pred, evaluation["speaker_ids"], 2.)
    crc_eval_rows = warning_rows(evaluation["scores"], 1 - eval_pred / 2, evaluation["speaker_ids"], crc["threshold"])
    crc_eval = warning_metrics(crc_eval_rows)
    no_flags = warning_metrics(warning_rows(evaluation["scores"], 1 - eval_pred / 2, evaluation["speaker_ids"], 1.))
    point = metrics(evaluation, eval_pred)
    point.pop("by_speaker")
    boot = evaluation_bootstrap(interval_rows, crc_eval_rows)
    curves = []
    for t in np.linspace(0, 1, 101):
        row = {"threshold": float(t), **warning_metrics(warning_rows(evaluation["scores"], 1 - eval_pred / 2, evaluation["speaker_ids"], float(t)))}
        flagged = 1 - eval_pred / 2 > t
        row["flagged_phone_score_mae_NOT_controlled"] = float(np.abs(evaluation["scores"][flagged] - eval_pred[flagged]).mean()) if flagged.any() else None
        row["abstained_phone_score_mae_NOT_controlled"] = float(np.abs(evaluation["scores"][~flagged] - eval_pred[~flagged]).mean()) if (~flagged).any() else None
        curves.append(row)
    save_json(args.output_dir / "descriptive_threshold_curve.json", {"scope": "evaluation descriptive only; never used for threshold reselection", "rows": curves})
    lower, upper = score_intervals(predicted, q)
    partitions = np.where(cal_mask, "calibration", "evaluation")
    np.savez_compressed(args.private_dir / "predictions.npz", scores=query["scores"], predicted=predicted,
                        speaker_ids=query["speaker_ids"], utterance_ids=query["utterance_ids"],
                        phone_indices=query["phone_indices"], reference_phone_id=query["reference_phone_id"],
                        partition=partitions, interval_lower=lower, interval_upper=upper,
                        priority=1 - predicted / 2, flagged=(1 - predicted / 2 > crc["threshold"]))
    save_json(args.private_dir / "calibration.json", {"conformal_rows": calibration_rows, "crc_rows": crc_cal_rows})
    save_json(args.private_dir / "evaluation_speaker_results.json", {"interval_rows": interval_rows, "crc_rows": crc_eval_rows})
    summary = {
        "date_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
        "source_sha256": sha256(Path(__file__)), "methodology_sha256": protocol["methodology_sha256"],
        "task": protocol["experiment"], "input_integrity": integrity,
        "splits": {k: {"n_speakers": len(split[k]), "n_utterances": len(set(data["utterance_ids"][masks[k]])),
                        "n_valid_phones": int(masks[k].sum()),
                        "n_zero_scores": int((data["scores"][masks[k]] == 0).sum()),
                        "n_aggregate_score2": int((data["scores"][masks[k]] == 2).sum())} for k in split},
        "point_evaluation": point,
        "conformal_calibration": {"alpha": .1, "rank_one_based": rank, "q": q,
                                  "speaker_max_residual_distribution": describe([r["max_absolute_error"] for r in calibration_rows])},
        "conformal_evaluation": primary, "trivial_0_2_comparator": trivial,
        "crc_calibration": crc, "crc_evaluation": crc_eval, "no_flags_comparator": no_flags,
        "evaluation_speaker_distributions": {k: describe([r[k] for r in crc_eval_rows]) for k in
                                           ("n_phones", "aggregate_score_2", "aggregate_score_below_2", "flags", "false_flags", "crc_loss", "false_fraction_among_flags", "full_recall_score_below_2")},
        "descriptive_speaker_bootstrap": boot,
        "model_state": {"n_reference_phone_categories_fit": len(state["encoder_categories"]),
                        "n_raw_predictions_outside_bounds_calibration_plus_evaluation": state["n_raw_ridge_predictions_outside_score_bounds"],
                        "cpu_thread_limit": 4},
        "test_data_prepared_or_evaluated": False, "private_gold_created": False,
        "limits": ["Exploratory numerical demonstration: all official TRAIN labels entered a previous Ridge fit and data/model results were inspected.",
                   "The nominal theorem assumes exchangeable new speakers and model/protocol decisions independent of calibration; this run is not fresh prospective certification.",
                   "One calibration observation is a whole speaker, not an individual correlated phone.",
                   "Score<2 includes accent; aggregate score=2 does not assert unanimous experts. No MDD/SDE label diagnosis or Vietnamese result.",
                   "CRC controls expected per-speaker FP/gold-score2 rate under assumptions; false fraction among issued flags and conditional selected MAE are not controlled.",
                   "Bootstrap holds the fitted model, q, and threshold fixed and has only 25 evaluation speakers."],
        "wall_seconds": time.monotonic() - start,
        "saved_predictions_sha256": sha256(args.private_dir / "predictions.npz")}
    save_json(args.output_dir / "summary.json", summary)
    write_findings(args.output_dir / "findings.md", summary)
    try:
        plot_results(args.output_dir / "risk_summary.png", summary, curves, crc_eval_rows)
    except ImportError as error:
        save_json(args.output_dir / "plot_unavailable.json", {"reason": str(error)})
    print(json.dumps({"phase": "exploratory_train_only_complete", "point": point,
                      "conformal": primary, "crc": crc_eval, "q": q, "threshold": crc["threshold"],
                      "protocol_sha256": protocol_hash}, indent=2))
    return summary


def write_findings(path: Path, s: dict) -> None:
    p, c, r, b = s["point_evaluation"], s["conformal_evaluation"], s["crc_evaluation"], s["descriptive_speaker_bootstrap"]["intervals"]
    false_fraction_text = (f"{r['false_fraction_among_flags_NOT_controlled']:.2%}"
                           if r["false_fraction_among_flags_NOT_controlled"] is not None else "không xác định (không có cảnh báo)")
    path.write_text(f"""# Khoảng điểm và cảnh báo có từ chối: thử nghiệm thăm dò

Chỉ official TRAIN speechocean762 được chia 75 người fit / 25 calibration / 25 evaluation, seed 1729. Không đánh giá lại official TEST. JSON khóa SHA256 `{s['protocol_sha256']}` được ghi trước khi fit/hiệu chuẩn. Lần chạy dùng dữ liệu đã từng được quan sát và đưa vào Ridge trước đây, nên đây là **minh họa số thăm dò**, không phải chứng nhận prospective trên người mới chưa từng được xem.

## Kết quả chính

Ridge alpha=1, GOP84 + 39 one-hot âm tham chiếu, scale chỉ trên fit và clip [0,2]: evaluation có {p['n_phones']:,} âm vị, MAE {p['phone_mae']:.6f}, RMSE {p['phone_rmse']:.6f}, PCC {p['pooled_phone_pearson']:.6f}. Speaker-macro MAE {p['speaker_macro_mae']:.6f}. Đây là dự đoán điểm chất lượng âm vị, không phải MDD/SDE.

Split conformal lấy phần tử thứ {s['conformal_calibration']['rank_one_based']}/25 của residual cực đại theo người, alpha=0.10, q={c['q']:.6f}. Bao phủ đồng thời {c['simultaneously_covered_speakers']}/25 người ({c['simultaneous_speaker_coverage']:.1%}); pooled phone coverage {c['pooled_phone_coverage']:.2%}; độ rộng trung bình {c['mean_interval_width']:.6f} trên miền điểm rộng 2. Khoảng tầm thường [0,2] đạt 25/25 người và độ rộng 2. CI bootstrap mô tả cho simultaneous coverage: {b['simultaneous_speaker_coverage']['percentile95']}; độ rộng: {b['mean_interval_width']['percentile95']}. Khoảng rộng hoặc utility thấp phải được giữ và báo cáo, không đổi alpha/score/phân hoạch sau quan sát.

Định lý split conformal, khi các đối tượng người nói trao đổi được và mọi lựa chọn độc lập khỏi calibration, có phạm vi xác suất biên cho toàn bộ âm vị được lấy mẫu của **một người mới**, không bảo đảm cho từng cá nhân hoặc mọi lời nói tương lai. Lịch sử tái sử dụng dữ liệu ở đây ngăn việc gọi lần chạy này là xác nhận mới của bảo đảm. [Quy tắc split conformal](https://arxiv.org/html/2107.07511v6#A4).

## Cảnh báo dựa trên điểm

Cố định risk mục tiêu=0.10; p=1-score_prediction/2 và cảnh báo khi p>t, còn lại từ chối kết luận. Ngưỡng calibration t={s['crc_calibration']['threshold']:.9f}; empirical macro loss={s['crc_calibration']['empirical_macro_loss']:.6f}; corrected risk={s['crc_calibration']['corrected_empirical_risk']:.6f}, đã cộng 1/26={s['crc_calibration']['finite_sample_correction']:.6f}.

Evaluation phát {r['flags']:,} cảnh báo / {r['n_phones']:,} âm vị ({r['flag_coverage']:.2%}), từ chối {r['abstention_coverage']:.2%}. Có {r['false_flags']:,} FP trên {r['aggregate_score_2']:,} âm vị aggregate score=2; pooled FP/gold2={r['pooled_false_flag_rate_on_score2']:.4%}, speaker-macro loss={r['speaker_macro_crc_loss']:.4%} (CI mô tả {b['crc_speaker_macro_loss']['percentile95']}). Có {r['true_flags_score_derived']:,} cảnh báo trên toàn bộ {r['aggregate_score_below_2']:,} âm vị aggregate score<2; full recall={r['full_recall_score_below_2']:.2%}, tính cả các mẫu bị từ chối trong mẫu số.

**Tỷ lệ cảnh báo phát ra bị sai** FP/(TP+FP) = {false_fraction_text}. Đại lượng này không được CRC khống chế; FPR thấp có thể vẫn đi kèm nhiều cảnh báo sai trong số cảnh báo. Nhóm score<2 gồm cả giọng nặng, không chỉ phát âm sai. Không suy ra rằng cả năm chuyên gia đồng ý ở score=2.

Baseline không bao giờ cảnh báo (t=1): 0 FP, risk 0, recall 0, coverage cảnh báo 0 và từ chối 100%; precision/tỷ lệ sai trong cảnh báo không xác định. Baseline này nhắc rằng đạt risk thấp chưa đủ để chứng minh hữu ích. Đường ngưỡng–risk/recall/coverage và MAE ở các tập bị cảnh báo/từ chối chỉ là thống kê thực nghiệm; không chọn ngưỡng từ đường evaluation.

CRC với mẫu số gold2 cố định và loss từng người trong [0,1] điều khiển **kỳ vọng loss** dưới giả thiết trao đổi được; không bảo đảm 90% mỗi người có risk dưới 10%. Không áp dụng công thức này cho FP/số cảnh báo hay MAE/số mẫu giữ lại vì không nhất thiết đơn điệu. [CRC, phương trình 4 và Định lý 1](https://arxiv.org/html/2208.02814v4#S2.SS1).

## Dữ liệu và giới hạn

Toàn bộ official TRAIN có 47.076 âm vị, 37.707 (80,10%) đạt aggregate score=2, nên luôn dự đoán điểm cao và hiếm cảnh báo có lợi do mất cân bằng. Fit/calibration/evaluation rời nhau theo người; điểm 0 vẫn giữ, -1 là padding. Đây là đặc trưng do tác giả GOPT cung cấp trên tiếng Anh đọc của người Mandarin L1, không phải audio→GOP tái lập hoàn chỉnh hoặc thử nghiệm tiếng Việt.

Summary lưu numerator/denominator, phân vị theo người và 2.000 bootstrap người evaluation, seed 1729. CI giữ cố định mô hình và calibration, không gồm bất định khi học/hiệu chuẩn lại và không thay thế bảo đảm lý thuyết. Predictions, split, calibration và metric từng người ở `data/public/speechocean762/risk_v1/`. Không tạo human gold nội bộ. Cần cohort độc lập và gán nhãn thật trước khi xác nhận hiệu quả tiếng Việt.
""", encoding="utf-8")


def plot_results(path: Path, s: dict, curves: list[dict], speaker_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["flag_coverage"] for r in curves]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1), constrained_layout=True)
    axes[0].bar(["Speaker max CP", "Trivial [0,2]"], [s["conformal_evaluation"]["mean_interval_width"], 2], color=["#187e9b", "#a8b7bf"])
    axes[0].set_ylim(0, 2.1)
    axes[0].set_ylabel("Mean interval width (score range 0-2)")
    axes[0].set_title(f"Simultaneous coverage: {s['conformal_evaluation']['simultaneously_covered_speakers']}/25 speakers")
    axes[1].plot(x, [r["speaker_macro_crc_loss"] for r in curves], label="Macro FP / gold score2")
    axes[1].plot(x, [r["full_recall_score_below_2"] for r in curves], label="Full recall: gold score<2")
    axes[1].scatter([s["crc_evaluation"]["flag_coverage"]], [s["crc_evaluation"]["speaker_macro_crc_loss"]], marker="x", color="black", zorder=5, label="Calibration-chosen threshold")
    axes[1].set_xlabel("Fraction of phones flagged")
    axes[1].set_ylabel("Evaluation rate (descriptive)")
    axes[1].set_title("Warning coverage and empirical risk")
    axes[1].legend(fontsize=8)
    axes[2].hist([r["crc_loss"] for r in speaker_rows], bins=np.linspace(0, 1, 11), color="#187e9b", edgecolor="white")
    axes[2].set_xlabel("FP / gold score2 for one speaker")
    axes[2].set_ylabel("Evaluation speakers")
    axes[2].set_title("Per-speaker loss at chosen threshold")
    fig.suptitle("Exploratory reused official TRAIN data | 75 fit / 25 calibration / 25 evaluation", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


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
