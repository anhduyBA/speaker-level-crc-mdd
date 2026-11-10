"""Inductive control: fixed per-phone rank map built on EpaDB (label-free), applied to L2-ARCTIC SSL-MD scores.

See 05_methodology/sslmd_l2arctic_phone_normalization_v2_note_2026-11-10.md. Run from code/:
    python -m vpa.sslmd_l2arctic_phone_normalization_v2
"""
import json

import numpy as np

from vpa import sslmd_l2arctic_phone_analysis_v1 as A
from vpa import sslmd_convergence_v3 as v3
from vpa import sslmd_smoke as s
from vpa.sslmd_l2arctic_phone_normalization_v1 import phone_rank, describe, loso
from vpa.sslmd_l2arctic_sensitivity_v1 import Calibrator

OUT = A.ROOT / "06_experiment_results/sslmd_l2arctic_phone_normalization_v2"
CACHE_DIRS = [A.ROOT / "data/derived/sslmd_training_v2/hidden", A.ROOT / "data/derived/sslmd_training_v1/hidden", s.PRIVATE / "hidden"]
ALPHAS = (0.05, 0.10, 0.15)
MIN_REF = 20


def reference_logits():
    import torch
    protocol = json.loads((A.SRC / "locked_protocol.json").read_text(encoding="utf-8"))
    heads = []
    for c in protocol["checkpoints"]:
        h = v3.make_head(protocol["arm"])
        with np.load(A.ROOT / c, allow_pickle=False) as z:
            h.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files})
        h.eval(); heads.append(h)
    rows = json.loads((s.PRIVATE / "frame_mapping.json").read_text(encoding="utf-8"))
    phones, logits, missing = [], [], []
    for r in rows:
        stem = r["row_id"].replace(":", "_") + ".npy"
        path = next((d / stem for d in CACHE_DIRS if (d / stem).exists()), None)
        if path is None:
            missing.append(r["row_id"]); continue
        layers = torch.from_numpy(np.load(path))
        assert layers.shape[0] == 12 and layers.shape[1] == r["frames"], (r["row_id"], layers.shape)
        with torch.inference_mode():
            pooled = np.array([s.pool_logits(h(layers).numpy(), r["phones"], r["ranges"]) for h in heads], dtype=object)
        for j, ph in enumerate(r["phones"]):
            vals = [x for x in pooled[:, j] if x is not None]
            if len(vals) == len(heads):
                phones.append(ph); logits.append(float(np.mean(vals)))
    assert not missing, missing[:5]
    return np.array(phones), np.array(logits), len(rows)


def fixed_rank(l2_logit, l2_phone_name, ref_phone_id, ref_logit):
    from vpa.external_hubert import PHONEMES
    ref_name = np.array([PHONEMES[p + 1] for p in ref_phone_id])
    pooled = np.sort(ref_logit)
    out = np.empty_like(l2_logit); fallback = []
    for ph in np.unique(l2_phone_name):
        m = l2_phone_name == ph
        ref = np.sort(ref_logit[ref_name == ph])
        if len(ref) < MIN_REF:
            ref = pooled; fallback.append(str(ph))
        out[m] = np.searchsorted(ref, l2_logit[m], side="right") / len(ref)
    counts = {str(ph): int((ref_name == ph).sum()) for ph in np.unique(ref_name)}
    return out, fallback, counts


def per_phone_table(err, phone, pr, th, speaker, phones):
    t = np.array([th[x] for x in speaker]); flag = pr > t
    rows = {}
    for ph in phones:
        m = phone == ph
        ok, bad = m & ~err, m & err
        warn = flag & m
        rows[ph] = {"errors": int(bad.sum()), "fpr": float((flag & ok).sum() / ok.sum()),
                    "precision": float((flag & bad).sum() / warn.sum()) if warn.sum() else None,
                    "recall": float((flag & bad).sum() / bad.sum()), "warnings": int(warn.sum())}
    return rows


def main():
    phone, err, speaker, pr = A.load()
    people = sorted(set(speaker.tolist()))
    ref_phone, ref_logit, n_utts = reference_logits()
    logit = np.log(pr["sslmd"] / (1 - pr["sslmd"]))
    inductive, fallback, ref_counts = fixed_rank(logit, phone, ref_phone, ref_logit)
    variants = {"raw": pr["sslmd"], "transductive": phone_rank(pr["sslmd"], phone), "inductive_epadb": inductive}
    locked_t = json.loads((A.SRC / "thresholds.json").read_text(encoding="utf-8"))
    res, ths = {}, {}
    for name, score in variants.items():
        cal = Calibrator(err, speaker, score)
        for a in ALPHAS:
            for m in ("crc", "empirical"):
                th = loso(cal, people, a, m)
                if name == "raw":
                    key = f"sslmd__{m}__{a:.2f}"
                    assert all(abs(th[p] - locked_t[key][p]) <= 1e-12 for p in people), key
                res[f"{name}__{m}__{a:.2f}"] = describe(err, speaker, phone, score, th, a)
                ths[f"{name}__{m}__{a:.2f}"] = th
    top = [ph for ph, _ in sorted(((ph, int((err & (phone == ph)).sum())) for ph in np.unique(phone)), key=lambda x: -x[1])[:10]]
    tables = {name: per_phone_table(err, phone, variants[name], ths[f"{name}__crc__0.10"], speaker, top) for name in ("raw", "inductive_epadb")}
    rng = np.random.default_rng(3191); draws = rng.integers(0, len(people), (5000, len(people)))
    diffs = {}
    for name in ("transductive", "inductive_epadb"):
        d = np.array(res[f"{name}__crc__0.10"]["per_speaker_recall"]) - np.array(res["raw__crc__0.10"]["per_speaker_recall"])
        b = d[draws].mean(1)
        diffs[f"{name}_minus_raw"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                                      "positive_speakers": int((d > 0).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({
        "label": "post-hoc exploratory; inductive control with a fixed EpaDB rank map (no L2-ARCTIC data in the map)",
        "reference": {"utterances": n_utts, "positions": int(len(ref_logit)), "phones_with_fallback": fallback, "per_phone_counts": ref_counts},
        "gate_raw_reproduces_locked_thresholds": True, "results": res, "per_phone_crc_0.10": tables, "recall_differences_alpha_0.10": diffs},
        indent=2), encoding="utf-8")
    print("reference positions", len(ref_logit), "fallback", fallback)
    for k, v in res.items():
        if "0.10" in k:
            print(k, round(v["mean_fpr"], 4), v["speakers_over_alpha"], round(v["max_fpr"], 4), round(v["mean_recall"], 4),
                  "AH", round(v["AH_share_of_false_warnings"] or 0, 3), "half", v["phones_for_half_of_false_warnings"],
                  {ph: (round(x["all"], 3), round(x["vietnamese"], 3)) for ph, x in v["consonant_recall"].items()})
    for name, t in tables.items():
        print(name, {ph: (round(r["fpr"], 3), None if r["precision"] is None else round(r["precision"], 3), round(r["recall"], 3)) for ph, r in t.items()})
    print(json.dumps(diffs))


if __name__ == "__main__":
    main()
