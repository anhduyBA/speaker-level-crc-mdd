"""Post-hoc exploratory: per-phone, label-free rank normalization before speaker-level CRC.

See 05_methodology/sslmd_l2arctic_phone_normalization_v1_note_2026-11-10.md. Run from code/:
    python -m vpa.sslmd_l2arctic_phone_normalization_v1
"""
import json

import numpy as np

from vpa import sslmd_l2arctic_phone_analysis_v1 as A
from vpa.sslmd_l2arctic_sensitivity_v1 import Calibrator

OUT = A.ROOT / "06_experiment_results/sslmd_l2arctic_phone_normalization_v1"
ALPHAS = (0.05, 0.10, 0.15)
CONSONANTS = ("T", "D", "L", "DH")
VIETNAMESE = ("HQTV", "PNV", "THV", "TLV")


def phone_rank(score, phone):
    """r = share of same-phone positions (all speakers, no labels) with score <= this one."""
    out = np.empty_like(score)
    for ph in np.unique(phone):
        m = phone == ph
        s = np.sort(score[m])
        out[m] = np.searchsorted(s, score[m], side="right") / m.sum()
    return out


def loso(cal, people, alpha, method):
    th = {p: cal.threshold([q for q in people if q != p], alpha, method) for p in people}
    return th


def describe(err, speaker, phone, pr, th, alpha):
    people = sorted(th)
    t = np.array([th[s] for s in speaker])
    flag = pr > t
    fpr = np.array([(flag & ~err & (speaker == p)).sum() / (~err & (speaker == p)).sum() for p in people])
    rec = np.array([(flag & err & (speaker == p)).sum() / (err & (speaker == p)).sum() for p in people])
    fw = flag & ~err
    by_phone = {ph: int((fw & (phone == ph)).sum()) for ph in np.unique(phone)}
    counts = sorted(by_phone.values(), reverse=True)
    total = sum(counts)
    half = int(np.searchsorted(np.cumsum(counts), total / 2) + 1) if total else 0
    vi = np.isin(speaker, VIETNAMESE)
    cons = {}
    for ph in CONSONANTS:
        m = err & (phone == ph)
        cons[ph] = {"all": float(flag[m].mean()), "vietnamese": float(flag[m & vi].mean())}
    return {"mean_fpr": float(fpr.mean()), "speakers_over_alpha": int((fpr > alpha).sum()), "max_fpr": float(fpr.max()),
            "mean_recall": float(rec.mean()), "total_false_warnings": total,
            "AH_share_of_false_warnings": by_phone.get("AH", 0) / total if total else None,
            "phones_for_half_of_false_warnings": half, "phones_with_any_false_warning": sum(c > 0 for c in counts),
            "consonant_recall": cons, "per_speaker_recall": rec.tolist(), "per_speaker_fpr": fpr.tolist()}


def main():
    phone, err, speaker, pr = A.load()
    people = sorted(set(speaker.tolist()))
    variants = {}
    for s in A.SCORES:
        variants[f"{s}_raw"] = pr[s]
        variants[f"{s}_phone_rank"] = phone_rank(pr[s], phone)
    locked_t = json.loads((A.SRC / "thresholds.json").read_text(encoding="utf-8"))
    res = {}
    for name, score in variants.items():
        cal = Calibrator(err, speaker, score)
        for a in ALPHAS:
            for m in ("crc", "empirical"):
                th = loso(cal, people, a, m)
                if name.endswith("_raw"):   # gate: raw thresholds must equal the locked ones
                    key = f"{name[:-4]}__{m}__{a:.2f}"
                    assert all(abs(th[p] - locked_t[key][p]) <= 1e-12 for p in people), key
                res[f"{name}__{m}__{a:.2f}"] = describe(err, speaker, phone, score, th, a)
    rng = np.random.default_rng(3191)
    draws = rng.integers(0, len(people), (5000, len(people)))
    diffs = {}
    for s in A.SCORES:
        d = np.array(res[f"{s}_phone_rank__crc__0.10"]["per_speaker_recall"]) - np.array(res[f"{s}_raw__crc__0.10"]["per_speaker_recall"])
        boot = d[draws].mean(1)
        diffs[f"{s}_rank_minus_raw_recall"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                                               "positive_speakers": int((d > 0).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"label": "post-hoc exploratory; label-free per-phone rank normalization pooled over all 24 speakers",
                                                   "gate_raw_reproduces_locked_thresholds": True, "results": res, "recall_differences_alpha_0.10": diffs},
                                                  indent=2), encoding="utf-8")
    for k, v in res.items():
        if "0.10" in k:
            print(k, round(v["mean_fpr"], 4), v["speakers_over_alpha"], round(v["max_fpr"], 4), round(v["mean_recall"], 4),
                  "AH", round(v["AH_share_of_false_warnings"] or 0, 3), "half", v["phones_for_half_of_false_warnings"], "any", v["phones_with_any_false_warning"],
                  {ph: (round(x["all"], 3), round(x["vietnamese"], 3)) for ph, x in v["consonant_recall"].items()})
    print(json.dumps(diffs))


if __name__ == "__main__":
    main()
