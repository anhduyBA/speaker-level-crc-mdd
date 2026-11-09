"""Sensitivity analyses on the locked L2-ARCTIC scores (see 05_methodology/sslmd_l2arctic_sensitivity_v1_note_2026-11-09.md).

Run from code/:  python -m vpa.sslmd_l2arctic_sensitivity_v1
"""
import json

import numpy as np

from vpa import sslmd_l2arctic_phone_analysis_v1 as A

OUT = A.ROOT / "06_experiment_results/sslmd_l2arctic_sensitivity_v1"
ALPHAS = (0.05, 0.10, 0.15)
SIZES = (5, 9, 14, 19, 23)
DRAWS, SEED = 50, 4242


class Calibrator:
    """Exact threshold search mirroring public_risk.calibrate_crc / the locked empirical rule."""

    def __init__(self, err, speaker, pr):
        self.people = sorted(set(speaker.tolist()))
        self.by = {}
        for p in self.people:
            m = speaker == p
            acc = np.sort(pr[m][~err[m]])
            self.by[p] = {"acc": acc, "A": len(acc), "all": pr[m], "err": err[m]}

    def _tail(self, cal):
        acc = np.concatenate([self.by[p]["acc"] for p in cal])
        w = np.concatenate([np.full(self.by[p]["A"], 1 / self.by[p]["A"]) for p in cal])
        o = np.argsort(acc, kind="stable")
        acc, w = acc[o], w[o]
        tail = np.r_[np.cumsum(w[::-1])[::-1], 0.0]
        cand = np.unique(np.r_[0.0, np.concatenate([self.by[p]["all"] for p in cal]), 1.0])
        s = tail[np.searchsorted(acc, cand, side="right")]          # sum of speaker losses at each candidate
        pooled = (len(acc) - np.searchsorted(acc, cand, side="right")) / len(acc)
        return cand, s, pooled

    def threshold(self, cal, alpha, method):
        cand, s, pooled = self._tail(cal)
        n = len(cal)
        ok = {"crc": (s + 1) / (n + 1) <= alpha + 1e-12, "empirical": s / n <= alpha + 1e-12,
              "pooled": pooled <= alpha + 1e-12}[method]
        idx = np.flatnonzero(ok)
        return float(cand[idx[0]]) if len(idx) else 1.0

    def apply(self, person, t):
        b = self.by[person]
        flag = b["all"] > t
        return {"fpr": float((flag & ~b["err"]).sum() / (~b["err"]).sum()),
                "recall": float((flag & b["err"]).sum() / b["err"].sum()), "silent": t >= 1.0}


def summarize(rows, alpha):
    fpr = np.array([r["fpr"] for r in rows])
    return {"evaluations": len(rows), "mean_fpr": float(fpr.mean()), "share_over_alpha": float((fpr > alpha).mean()),
            "speakers_over_alpha": int((fpr > alpha).sum()), "max_fpr": float(fpr.max()),
            "mean_recall": float(np.mean([r["recall"] for r in rows])), "share_no_warning": float(np.mean([r["silent"] for r in rows]))}


def sign_flip_p(d):
    """Exact two-sided paired sign-flip permutation p-value over all 2^n sign vectors."""
    d = np.asarray(d, float)
    n, obs = len(d), abs(d.mean())
    hits, chunk = 0, 1 << 18
    bits = (np.arange(chunk)[:, None] >> np.arange(18)) & 1                       # low 18 bits
    low = (1 - 2 * bits) @ d[:18]
    for hi in range(1 << (n - 18)):
        hb = (hi >> np.arange(n - 18)) & 1
        total = low + (1 - 2 * hb) @ d[18:]
        hits += int((np.abs(total / n) >= obs - 1e-15).sum())
    return hits / float(1 << n)


def main():
    _, err, speaker, pr = A.load()
    locked_t = json.loads((A.SRC / "thresholds.json").read_text(encoding="utf-8"))
    summary = json.loads((A.SRC / "summary.json").read_text(encoding="utf-8"))
    cal = {s: Calibrator(err, speaker, pr[s]) for s in A.SCORES}
    people = cal["sslmd"].people

    # Gate: reproduce every locked LOSO threshold and the locked means.
    for s in A.SCORES:
        for a in ALPHAS:
            for m in ("crc", "empirical"):
                key = f"{s}__{m}__{a:.2f}"
                rows = []
                for p in people:
                    t = cal[s].threshold([q for q in people if q != p], a, m)
                    assert abs(t - locked_t[key][p]) <= 1e-12, (key, p, t, locked_t[key][p])
                    rows.append(cal[s].apply(p, t))
                ref = summary["P1_loso"][key]
                assert abs(np.mean([r["fpr"] for r in rows]) - ref["mean_fpr"]) < 1e-9, key
                assert abs(np.mean([r["recall"] for r in rows]) - ref["mean_recall"]) < 1e-9, key

    # S1: pooled-phone thresholding
    s1 = {}
    for s in A.SCORES:
        for a in ALPHAS:
            for m in ("pooled", "empirical", "crc"):
                rows = [cal[s].apply(p, cal[s].threshold([q for q in people if q != p], a, m)) for p in people]
                s1[f"{s}__{m}__{a:.2f}"] = summarize(rows, a)

    # S2: number of calibration speakers
    rng = np.random.default_rng(SEED)
    s2 = {}
    for n in SIZES:
        subsets = {p: ([[q for q in people if q != p]] if n == 23 else
                       [list(rng.choice([q for q in people if q != p], n, replace=False)) for _ in range(DRAWS)]) for p in people}
        for s in A.SCORES:
            for m in ("crc", "empirical"):
                rows = [cal[s].apply(p, cal[s].threshold(c, 0.10, m)) for p in people for c in subsets[p]]
                s2[f"{s}__{m}__n{n}"] = summarize(rows, 0.10)

    # S3: exact sign-flip permutation for U1/U2
    rec = {s: np.array([r["recall"] for r in summary["P1_loso"][f"{s}__crc__0.10"]["per_speaker"]]) for s in ("sslmd", "gop", "prior")}
    order = [r["speaker"] for r in summary["P1_loso"]["sslmd__crc__0.10"]["per_speaker"]]
    for s in ("gop", "prior"):
        assert [r["speaker"] for r in summary["P1_loso"][f"{s}__crc__0.10"]["per_speaker"]] == order
    s3 = {}
    for k, other in (("U1_sslmd_minus_gop", "gop"), ("U2_sslmd_minus_prior", "prior")):
        d = rec["sslmd"] - rec[other]
        s3[k] = {"mean": float(d.mean()), "positive_speakers": int((d > 0).sum()), "sign_flip_p": sign_flip_p(d)}
    ps = sorted(s3, key=lambda k: s3[k]["sign_flip_p"])
    running = 0.0
    for i, k in enumerate(ps):
        running = max(running, min(1.0, (len(ps) - i) * s3[k]["sign_flip_p"]))
        s3[k]["holm_p"] = running

    OUT.mkdir(parents=True, exist_ok=True)
    result = {"source_protocol_sha256": summary["protocol_sha256"], "gate_reproduced_locked_thresholds": True,
              "label": "sensitivity analyses; not pre-registered endpoints", "S1_pooled_phone_threshold": s1,
              "S2_calibration_size_alpha_0.10": s2, "S3_sign_flip_permutation": s3,
              "bootstrap_note": "locked bootstrap resamples held-out speakers with fixed LOSO thresholds (no recalibration)"}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, v in s1.items():
        if "0.10" in k: print("S1", k, round(v["mean_fpr"], 4), v["speakers_over_alpha"], round(v["max_fpr"], 4), round(v["mean_recall"], 4))
    for k, v in s2.items():
        print("S2", k, round(v["mean_fpr"], 4), round(v["share_over_alpha"], 3), round(v["mean_recall"], 4), round(v["share_no_warning"], 3))
    print("S3", json.dumps(s3))


if __name__ == "__main__":
    main()
