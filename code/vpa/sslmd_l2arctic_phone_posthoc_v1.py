import json, numpy as np
from scipy.stats import spearmanr
from vpa.external_hubert import PHONEMES
from vpa import sslmd_l2arctic_phone_analysis_v1 as A
phone, err, speaker, pr = A.load()
th = json.load(open(A.SRC/'thresholds.json', encoding='utf-8'))
v4 = json.load(open(A.ROOT/'06_experiment_results/sslmd_convergence_v4/summary.json', encoding='utf-8'))['metrics']['mixedConv']['phones']
epa = {PHONEMES[int(k)+1]: (v['bad'], v['good']+v['bad']) for k, v in v4.items()}
rows = []
for ph in sorted(set(phone.tolist())):
    m = phone == ph
    if ph not in epa or epa[ph][1] < 50 or m.sum() < 50: continue
    r = {'phone': ph, 'epadb_error_rate': epa[ph][0]/epa[ph][1], 'l2_error_rate': float(err[m].mean()), 'n_l2': int(m.sum())}
    for s in A.SCORES:
        t = np.array([th[f'{s}__crc__0.10'][sp] for sp in speaker[m]])
        r[f'{s}_warning_rate'] = float((pr[s][m] > t).mean())
    rows.append(r)
out = {'phones': len(rows)}
for s in A.SCORES:
    w = [r[f'{s}_warning_rate'] for r in rows]
    out[s] = {'rho_vs_epadb_error_rate': float(spearmanr([r['epadb_error_rate'] for r in rows], w)[0]),
              'rho_vs_l2_error_rate': float(spearmanr([r['l2_error_rate'] for r in rows], w)[0])}
out['rho_epadb_vs_l2_error_rate'] = float(spearmanr([r['epadb_error_rate'] for r in rows], [r['l2_error_rate'] for r in rows])[0])
p = A.OUT/'summary.json'; s = json.load(open(p, encoding='utf-8')); s['posthoc_exploratory_phone_rate_correlation'] = out | {'rows': rows}
json.dump(s, open(p, 'w', encoding='utf-8'), indent=2); print(json.dumps(out, indent=1))
