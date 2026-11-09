"""Exploratory speaker-level CRC on v3 SSL-MD out-of-fold scores (EpaDB, leave-one-speaker-out)."""
from datetime import datetime, timezone
import numpy as np
from . import sslmd_smoke as s
from . import sslmd_convergence_v3 as v3
from .public_risk import calibrate_crc, warning_rows

ROOT=s.ROOT;OUT=ROOT/'06_experiment_results/sslmd_risk_epadb_v1'
CONTRACT=ROOT/'05_methodology/sslmd_risk_epadb_v1_contract_2026-11-08.md'
ALPHAS=(.10,.15,.20);ARMS=('mixed60','uniform60','prior')
read,save,sha=s.read,s.save,s.sha


def to_crc(rows):
    """y=0 acceptable -> gold 2; y=1 error -> gold 0; priority = sigmoid(logit)."""
    gold=np.array([2 if r['y']==0 else 0 for r in rows]);speakers=np.array([r['speaker'] for r in rows])
    priority=1/(1+np.exp(-np.clip(np.array([r['score'] for r in rows],float),-500,500)))
    return gold,priority,speakers


def split_crc_feasible(n,alpha):
    """Smallest attainable corrected risk is 1/(n+1) (zero calibration loss)."""
    return 1/(n+1)<=alpha


def empirical_threshold(gold,priority,speakers,alpha):
    """Smallest candidate threshold whose uncorrected speaker-macro loss is <= alpha."""
    candidates=np.unique(np.r_[0.,priority,1.])
    loss=lambda t:np.mean([r['crc_loss'] for r in warning_rows(gold,priority,speakers,float(t))])
    lo,hi=0,len(candidates)-1  # loss is non-increasing in t; loss(1)=0
    while lo<hi:
        mid=(lo+hi)//2
        if loss(candidates[mid])<=alpha:hi=mid
        else:lo=mid+1
    return float(candidates[lo])


def loso(rows,alpha,method):
    gold,priority,speakers=to_crc(rows);out=[]
    for person in sorted(set(speakers.tolist())):
        cal=speakers!=person
        if method=='crc':t=calibrate_crc(gold[cal],priority[cal],speakers[cal],alpha)['threshold']
        else:t=empirical_threshold(gold[cal],priority[cal],speakers[cal],alpha)
        test=~cal;r=warning_rows(gold[test],priority[test],speakers[test],t)[0]
        out.append({'speaker':person,'threshold':t,'fpr':r['crc_loss'],'recall':r['full_recall_score_below_2'],
                    'flags':r['flags'],'n':r['n_phones']})
    return out


def describe(per,alpha,draws):
    f=np.array([r['fpr'] for r in per]);rc=np.array([r['recall'] for r in per])
    return {'mean_fpr':float(f.mean()),'mean_fpr_ci95':np.quantile(f[draws].mean(1),[.025,.975]).tolist(),
            'speakers_over_alpha':int((f>alpha).sum()),'mean_recall':float(rc.mean()),
            'mean_recall_ci95':np.quantile(rc[draws].mean(1),[.025,.975]).tolist(),
            'flag_share':sum(r['flags'] for r in per)/sum(r['n'] for r in per),'per_speaker':per}


def run():
    assert not (OUT/'summary.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    source=v3.PRIVATE/'held_predictions.json';held=read(source)
    rng=np.random.default_rng(3191);draws=rng.integers(0,10,(5000,10))
    results={}
    for arm in ARMS:
        for alpha in ALPHAS:
            for method in ('crc','empirical'):
                results[f'{arm}__{method}__{alpha:.2f}']=describe(loso(held[arm],alpha,method),alpha,draws)
    v3s=read(v3.OUT/'summary.json')
    actcost={arm:{'mean_fpr':v3s['metrics'][arm]['means']['conditional_fpr'],'mean_recall':v3s['metrics'][arm]['means']['recall']} for arm in ARMS}
    save(OUT/'summary.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (source,CONTRACT,ROOT/'code/vpa/sslmd_risk_epadb_v1.py',ROOT/'code/vpa/public_risk.py')},
        'within_fold_split_crc_feasible':{f'{a:.2f}':split_crc_feasible(2,a) for a in ALPHAS},
        'loso_min_attainable_corrected_risk':1/10,'results':results,'v3_actcost_reference':actcost,'exploratory':True})
    for k,v in results.items():print(k,round(v['mean_fpr'],4),v['speakers_over_alpha'],round(v['mean_recall'],4),round(v['flag_share'],4))


if __name__=='__main__':run()
