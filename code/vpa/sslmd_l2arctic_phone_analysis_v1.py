"""Descriptive per-phone analysis of the locked L2-ARCTIC CRC run (no retraining, no new thresholds).

See 05_methodology/sslmd_l2arctic_phone_analysis_v1_note_2026-11-09.md. Run from code/:
    python -m vpa.sslmd_l2arctic_phone_analysis_v1
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from vpa.external_hubert import PHONEMES

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "06_experiment_results/sslmd_l2arctic_crc_v1"
OUT = ROOT / "06_experiment_results/sslmd_l2arctic_phone_analysis_v1"
SCORES = ("sslmd", "gop")
ALPHA = "0.10"
VIETNAMESE = ("HQTV", "PNV", "THV", "TLV")
MIN_ERRORS_ALL, MIN_ERRORS_VI = 200, 50


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    locked = json.loads((SRC / "scores_locked.json").read_text(encoding="utf-8"))
    rows = []
    for rel, digest in sorted(locked["files_sha256"].items()):
        assert sha(ROOT / rel) == digest, rel
        rows.append(json.loads((ROOT / rel).read_text(encoding="utf-8")))
    phone = np.array([PHONEMES[p + 1] for r in rows for p in r["phones"]])
    err = np.array([y == 1 for r in rows for y in r["labels"]])
    speaker = np.array([r["speaker"] for r in rows for _ in r["labels"]])
    logit = np.array([x for r in rows for x in r["sslmd_logit"]], float)
    pr = {"sslmd": 1 / (1 + np.exp(-logit)), "gop": np.array([x for r in rows for x in r["gop_priority"]], float)}
    return phone, err, speaker, pr


def main():
    phone, err, speaker, pr = load()
    thresholds = json.loads((SRC / "thresholds.json").read_text(encoding="utf-8"))
    summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))
    flag = {}
    for s in SCORES:
        t = np.array([thresholds[f"{s}__crc__{ALPHA}"][sp] for sp in speaker])
        flag[s] = pr[s] > t
        # Gate: reproduce the locked speaker-macro FPR and recall.
        people = sorted(set(speaker.tolist()))
        fpr = np.mean([(flag[s] & ~err & (speaker == p)).sum() / (~err & (speaker == p)).sum() for p in people])
        rec = np.mean([(flag[s] & err & (speaker == p)).sum() / (err & (speaker == p)).sum() for p in people])
        ref = summary["P1_loso"][f"{s}__crc__{ALPHA}"]
        assert abs(fpr - ref["mean_fpr"]) < 1e-9 and abs(rec - ref["mean_recall"]) < 1e-9, (s, fpr, rec)

    phones = sorted(set(phone.tolist()))
    def per_phone(mask):
        out = {}
        for ph in phones:
            m = mask & (phone == ph)
            ok, bad = m & ~err, m & err
            out[ph] = {"acceptable": int(ok.sum()), "errors": int(bad.sum())}
            for s in SCORES:
                out[ph][f"{s}_false_warnings"] = int((flag[s] & ok).sum())
                out[ph][f"{s}_fpr"] = float((flag[s] & ok).sum() / ok.sum()) if ok.sum() else None
                out[ph][f"{s}_recall"] = float((flag[s] & bad).sum() / bad.sum()) if bad.sum() else None
        return out

    allp = per_phone(np.ones_like(err))
    concentration = {}
    for s in SCORES:
        fw = sorted((v[f"{s}_false_warnings"] for v in allp.values()), reverse=True)
        total = sum(fw)
        k = int(np.searchsorted(np.cumsum(fw), total / 2) + 1)
        concentration[s] = {"total_false_warnings": total, "phones_for_half": k, "phones_with_any": sum(x > 0 for x in fw)}
    total_fw = {s: concentration[s]["total_false_warnings"] for s in SCORES}
    top_fw = sorted(allp, key=lambda ph: -allp[ph]["sslmd_false_warnings"])[:10]
    top_fw_rows = [{"phone": ph, **allp[ph], **{f"{s}_share": allp[ph][f"{s}_false_warnings"] / total_fw[s] for s in SCORES}} for ph in top_fw]
    recall_rows = sorted(({"phone": ph, **v, "recall_diff": v["sslmd_recall"] - v["gop_recall"]}
                          for ph, v in allp.items() if v["errors"] >= MIN_ERRORS_ALL), key=lambda r: -r["errors"])
    vi = per_phone(np.isin(speaker, VIETNAMESE))
    vi_rows = sorted(({"phone": ph, **v} for ph, v in vi.items() if v["errors"] >= MIN_ERRORS_VI), key=lambda r: -r["errors"])[:10]

    OUT.mkdir(parents=True, exist_ok=True)
    result = {"source_protocol_sha256": summary["protocol_sha256"], "alpha": float(ALPHA), "descriptive_only": True,
              "gate_reproduced_locked_means": True, "concentration": concentration, "top_false_warning_phones": top_fw_rows,
              "recall_by_phone_min200": recall_rows, "vietnamese_top_error_phones": vi_rows, "per_phone_all": allp,
              "per_phone_vietnamese": vi}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(concentration))
    for r in top_fw_rows:
        print("FW", r["phone"], r["acceptable"], round(r["sslmd_fpr"], 3), round(r["sslmd_share"], 3), round(r["gop_fpr"], 3), round(r["gop_share"], 3))
    for r in recall_rows:
        print("REC", r["phone"], r["errors"], round(r["sslmd_recall"], 3), round(r["gop_recall"], 3))
    for r in vi_rows:
        print("VI", r["phone"], r["errors"], r["acceptable"], round(r["sslmd_recall"], 3), round(r["gop_recall"], 3), round(r["sslmd_fpr"], 3), round(r["gop_fpr"], 3))


if __name__ == "__main__":
    main()
