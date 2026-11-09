"""L2-ARCTIC speaker-level CRC with a fixed SSL-MD head trained only on EpaDB (option A)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import time
import numpy as np
from . import sslmd_smoke as s
from . import sslmd_training_v2 as v2
from . import sslmd_convergence_v3 as v3
from . import sslmd_convergence_v4 as v4
from .public_risk import calibrate_crc, warning_rows

ROOT=s.ROOT;OUT=ROOT/'06_experiment_results/sslmd_l2arctic_crc_v1'
PRIVATE=ROOT/'data/derived/sslmd_l2arctic_crc_v1'
CONTRACT=ROOT/'05_methodology/sslmd_l2arctic_crc_v1_contract_2026-11-08.md'
MANIFEST=ROOT/'data/derived/public_corpora/l2arctic_manifest.jsonl'
CORPUS=ROOT/'data/public/l2arctic_v5'
ALPHAS=(.05,.10,.15);PRIMARY=.10;SCORES=('sslmd','gop','prior')
PILOT={'ABA','ASI','BWC','EBVS','HJK','HQTV'}
STANDARD_TEST={'NJS','TLV','TNI','TXHC','YKWK','ZHAA'}
read,save,sha=s.read,s.save,s.sha


# ---------- labels ----------
def positions(intervals):
    """Canonical positions and labels; None when the utterance must be excluded."""
    from .external_hubert import PHONE_TO_ID
    if any(i['kind'] in ('unparsed','unscorable') for i in intervals):return None,'unparsed_or_unscorable'
    phones,labels=[],[]
    for i in intervals:
        if i['kind'] in ('silence','insertion'):continue
        if i['kind'] not in ('acceptable','substitution','deletion'):raise ValueError(i['kind'])
        if i['canonical_phone']=='AX':return None,'canonical_AX'
        phones.append(PHONE_TO_ID[i['canonical_phone']]-1);labels.append(0 if i['kind']=='acceptable' else 1)
    if not phones:return None,'no_canonical_positions'
    return (phones,labels),None


def labels():
    rows=[];excluded=Counter();inserted=Counter();summary={}
    for line in open(MANIFEST,encoding='utf-8'):
        r=json.loads(line);key=f"{r['speaker_id']}:{r['utterance_id']}"
        got,why=positions(r['phone_intervals'])
        if got is None:excluded[why]+=1;continue
        phones,ys=got;inserted[r['speaker_id']]+=sum(i['kind']=='insertion' for i in r['phone_intervals'])
        rows.append({'row_id':key,'speaker':r['speaker_id'],'l1':r['l1'],'wav':r['paths_relative_to_corpus']['wav'],
                     'wav_sha256':r['sha256']['wav'],'phones':phones,'labels':ys})
    for l1 in sorted({r['l1'] for r in rows}):
        rr=[r for r in rows if r['l1']==l1]
        summary[l1]={'speakers':sorted({r['speaker'] for r in rr}),'utterances':len(rr),
                     'positions':sum(len(r['labels']) for r in rr),'errors':sum(sum(r['labels']) for r in rr)}
    save(PRIVATE/'labels.json',rows)
    save(OUT/'labels_summary.json',{'utterances':len(rows),'speakers':len({r['speaker'] for r in rows}),
         'positions':sum(len(r['labels']) for r in rows),'errors':sum(sum(r['labels']) for r in rows),
         'excluded_utterances':dict(excluded),'insertions_not_scored':sum(inserted.values()),'by_l1':summary,
         'labels_sha256':sha(PRIVATE/'labels.json')})
    print({'utterances':len(rows),'excluded':dict(excluded)},flush=True)


# ---------- lock ----------
def lock():
    assert not (OUT/'locked_protocol.json').exists()
    summary=read(v4.OUT/'summary.json');arm=summary['gate_B_baseline_by_locked_rule'];assert arm in v4.ARMS
    fits=read(v4.OUT/'fits_locked.json');p4=v4.protocol();checkpoints=[]
    for fold in p4['folds']:
        for seed in p4['seeds']:
            rec=read(v4.PRIVATE/'fits'/f"fold{fold['fold']}_{arm}_{seed}.json")
            c=rec['caps']['conv'];assert sha(ROOT/c['checkpoint_path'])==c['checkpoint_sha256']
            checkpoints.append(c['checkpoint_path'])
    assert len(checkpoints)==25
    priors=[(v2.PRIVATE/'fits'/f"fold{f['fold']}_prior.json").relative_to(ROOT).as_posix() for f in p4['folds']]
    provenance=read(s.MODEL/'provenance.json')
    files=[ROOT/'code/vpa/sslmd_l2arctic_crc_v1.py',ROOT/'code/tests/test_sslmd_l2arctic_crc_v1.py',CONTRACT,MANIFEST,
           PRIVATE/'labels.json',OUT/'labels_summary.json',OUT/'software_tests.json',v4.OUT/'summary.json',v4.OUT/'fits_locked.json',
           ROOT/'code/vpa/public_risk.py',ROOT/'code/vpa/phone_alignment.py',ROOT/'code/vpa/external_hubert.py',
           ROOT/'code/vpa/public_hubert_pilot.py',s.MODEL/'provenance.json']+[s.MODEL/n for n in provenance['files']]
    files+=[ROOT/c for c in checkpoints]+[ROOT/q for q in priors]
    save(OUT/'locked_protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'inputs_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))},
        'arm':arm,'checkpoints':checkpoints,'priors':priors,'alphas':list(ALPHAS),'primary_alpha':PRIMARY,
        'endpoints':{'U1':'LOSO CRC alpha0.10 recall sslmd-gop','U2':'LOSO CRC alpha0.10 recall sslmd-prior','multiplicity':'Holm over U1,U2',
                     'validity':'mean LOSO FPR and speakers over alpha (descriptive)','P2':'leave-one-L1-out CRC (robustness)',
                     'P3':'speaker-macro pAUC/AUROC','P4':'standard 6 test speakers P/R/F1 (context only)','sensitivity':'P1 restricted to 18 non-pilot test speakers'},
        'statistics':{'bootstrap_seed':3191,'draws':5000,'unit':'speaker'},
        'no_l2arctic_training_or_selection':True,'scope':'Fixed EpaDB-trained head; L2-ARCTIC used only for calibration/evaluation'})
    print({'locked_arm':arm},flush=True)


def protocol():
    p=read(OUT/'locked_protocol.json')
    for rel,h in p['inputs_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p


# ---------- scoring ----------
def gop_priority(prob,codes,ranges):
    """1 - mean posterior of the canonical class over its aligned frames."""
    return [1.-float(prob[lo:hi,c].mean()) for c,(lo,hi) in zip(codes,ranges,strict=True)]


def score():
    p=protocol();v2.guard();torch=v2.configure_torch()
    from .external_hubert import load_model,conv_output_length
    from .phone_alignment import posterior_align
    from .public_hubert_pilot import waveform
    heads=[]
    for c in p['checkpoints']:
        head=v3.make_head(p['arm'])
        with np.load(ROOT/c,allow_pickle=False) as z:head.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
        head.eval();heads.append(head)
    prior=np.mean([read(ROOT/q)['fit']['coef'] for q in p['priors']],0)
    model=load_model(s.MODEL);transformer=model.model.wav2vec2.encoder.transformer;assert len(transformer.layers)==12
    rows=read(PRIVATE/'labels.json');out=PRIVATE/'scores';out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();done=0
    for r in rows:
        dest=out/(r['row_id'].replace(':','_')+'.json')
        if dest.exists():done+=1;continue
        v2.guard()
        path=CORPUS/r['wav'];assert sha(path)==r['wav_sha256']
        audio=waveform(path);wave=torch.from_numpy(audio)[None];captured=[]
        hooks=[layer.register_forward_hook(lambda _m,_i,o:captured.append(o[0].detach())) for layer in transformer.layers]
        try:
            with torch.inference_mode():
                logits=model(wave);assert len(captured)==12
                assert all(tuple(t.shape)==(1,conv_output_length(len(audio)),768) for t in captured)
                layers=torch.stack([t[0] for t in captured]);prob=logits.softmax(-1)[0].numpy().astype(np.float64)
                codes=[x+1 for x in r['phones']];ranges=posterior_align(prob,codes)['phone_frame_ranges']
                pooled=np.array([s.pool_logits(h(layers).numpy(),r['phones'],ranges) for h in heads],dtype=float)
        finally:
            for h in hooks:h.remove()
        sslmd=pooled.mean(0);gop=gop_priority(prob,codes,ranges)
        save(dest,{'row_id':r['row_id'],'speaker':r['speaker'],'l1':r['l1'],'ranges':ranges,'labels':r['labels'],'phones':r['phones'],
                   'sslmd_logit':sslmd.tolist(),'gop_priority':gop,'prior_logit':[float(prior[q]) for q in r['phones']],
                   'frames':len(prob),'protocol_sha256':sha(OUT/'locked_protocol.json')})
        done+=1;del captured,layers,logits,prob,wave
        if done%50==0:print({'scored':done,'of':len(rows),'seconds':round(time.perf_counter()-started,1)},flush=True)
    files=sorted(out.glob('*.json'));assert len(files)==len(rows)
    save(OUT/'scores_locked.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'utterances':len(files),
         'files_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in files},'resources_end':s.resources()})
    print('All utterances scored.',flush=True)


# ---------- evaluation ----------
def sigmoid(x):return 1/(1+np.exp(-np.clip(np.asarray(x,float),-500,500)))


def table(records):
    """Flat arrays per score: gold (2 acceptable / 0 error), priority, speaker, l1."""
    gold=np.array([2 if y==0 else 0 for r in records for y in r['labels']])
    speakers=np.array([r['speaker'] for r in records for _ in r['labels']]);l1=np.array([r['l1'] for r in records for _ in r['labels']])
    pr={'sslmd':sigmoid([x for r in records for x in r['sslmd_logit']]),'gop':np.array([x for r in records for x in r['gop_priority']],float),
        'prior':sigmoid([x for r in records for x in r['prior_logit']])}
    return gold,pr,speakers,l1


def empirical_threshold(gold,priority,speakers,alpha):
    candidates=np.unique(np.r_[0.,priority,1.]);lo,hi=0,len(candidates)-1
    loss=lambda t:np.mean([r['crc_loss'] for r in warning_rows(gold,priority,speakers,float(t))])
    while lo<hi:
        mid=(lo+hi)//2
        if loss(candidates[mid])<=alpha:hi=mid
        else:lo=mid+1
    return float(candidates[lo])


def apply(gold,priority,speakers,cal,test,alpha,method):
    t=(calibrate_crc(gold[cal],priority[cal],speakers[cal],alpha)['threshold'] if method=='crc'
       else empirical_threshold(gold[cal],priority[cal],speakers[cal],alpha))
    rows=warning_rows(gold[test],priority[test],speakers[test],t)
    return t,[{'speaker':r['speaker_id'],'fpr':r['crc_loss'],'recall':r['full_recall_score_below_2'],'flags':r['flags'],
               'tp':r['true_flags_score_derived'],'fp':r['false_flags'],'errors':r['aggregate_score_below_2'],'n':r['n_phones']} for r in rows]


def summarize(per,alpha):
    f=np.array([r['fpr'] for r in per]);rc=np.array([r['recall'] for r in per]);tp=sum(r['tp'] for r in per);fl=sum(r['flags'] for r in per)
    return {'mean_fpr':float(f.mean()),'speakers_over_alpha':int((f>alpha).sum()),'max_fpr':float(f.max()),'mean_recall':float(rc.mean()),
            'flag_share':fl/sum(r['n'] for r in per),'pooled_precision':tp/fl if fl else None,'per_speaker':per}


def loso(gold,priority,speakers,alpha,method):
    per=[];thresholds={}
    for person in sorted(set(speakers.tolist())):
        t,rows=apply(gold,priority,speakers,speakers!=person,speakers==person,alpha,method);per+=rows;thresholds[person]=t
    return per,thresholds


def boot(values,draws):
    v=np.asarray(values,float);b=v[draws].mean(1)
    return {'mean':float(v.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'bootstrap_p':v4.bootstrap_p(b)}


def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else None;r=tp/(tp+fn) if tp+fn else None
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p and r else None,'tp':tp,'fp':fp,'fn':fn}


def f1_threshold(gold,priority):
    order=np.argsort(-priority,kind='stable');err=(gold[order]<2).astype(int);tp=np.cumsum(err);fp=np.cumsum(1-err);fn=err.sum()-tp
    f1=2*tp/(2*tp+fp+fn);i=int(np.argmax(f1));return float(priority[order][i])-1e-12


def evaluate():
    from sklearn.metrics import roc_auc_score
    p=protocol();assert not (OUT/'summary.json').exists();locked=read(OUT/'scores_locked.json')
    for rel,h in locked['files_sha256'].items():assert sha(ROOT/rel)==h,rel
    records=[read(ROOT/rel) for rel in sorted(locked['files_sha256'])]
    gold,pr,speakers,l1=table(records);people=sorted(set(speakers.tolist()));assert len(people)==24
    rng=np.random.default_rng(p['statistics']['bootstrap_seed']);draws=rng.integers(0,24,(p['statistics']['draws'],24))
    p1={};thresholds={}
    for name in SCORES:
        for alpha in ALPHAS:
            for method in ('crc','empirical'):
                per,t=loso(gold,pr[name],speakers,alpha,method);key=f'{name}__{method}__{alpha:.2f}'
                p1[key]=summarize(per,alpha);thresholds[key]=t
                p1[key]['mean_fpr_ci95']=boot([r['fpr'] for r in per],draws)['ci95'];p1[key]['mean_recall_ci95']=boot([r['recall'] for r in per],draws)['ci95']
    rec=lambda name:np.array([r['recall'] for r in p1[f'{name}__crc__{PRIMARY:.2f}']['per_speaker']])
    endpoints={'U1_sslmd_minus_gop':boot(rec('sslmd')-rec('gop'),draws),'U2_sslmd_minus_prior':boot(rec('sslmd')-rec('prior'),draws)}
    for k,q in zip(endpoints,v4.holm([e['bootstrap_p'] for e in endpoints.values()])):endpoints[k]['holm_p']=q
    p2={}
    for name in SCORES:
        for lang in sorted(set(l1.tolist())):
            _,per=apply(gold,pr[name],speakers,l1!=lang,l1==lang,PRIMARY,'crc');p2[f'{name}__{lang}']=summarize(per,PRIMARY)
    p3={}
    for name in SCORES:
        per=[]
        for person in people:
            m=speakers==person;y=(gold[m]<2).astype(int)
            per.append({'speaker':person,'l1':str(l1[m][0]),'pauc':float(roc_auc_score(y,pr[name][m],max_fpr=.1)),'auroc':float(roc_auc_score(y,pr[name][m]))})
        p3[name]={'mean_pauc':float(np.mean([r['pauc'] for r in per])),'mean_auroc':float(np.mean([r['auroc'] for r in per])),
                  'by_l1':{lang:float(np.mean([r['pauc'] for r in per if r['l1']==lang])) for lang in sorted(set(l1.tolist()))},'per_speaker':per}
    p3['sslmd_minus_gop_pauc']=boot(np.array([r['pauc'] for r in p3['sslmd']['per_speaker']])-np.array([r['pauc'] for r in p3['gop']['per_speaker']]),draws)
    test=np.isin(speakers,list(STANDARD_TEST));cal=~test;p4={}
    for name in SCORES:
        t_crc,_=apply(gold,pr[name],speakers,cal,test,PRIMARY,'crc');t_f1=f1_threshold(gold[cal],pr[name][cal])
        for label,t in (('crc_alpha0.10',t_crc),('f1max_on_18',t_f1)):
            flag=pr[name][test]>t;err=gold[test]<2
            p4[f'{name}__{label}']=prf(int((flag&err).sum()),int((flag&~err).sum()),int((~flag&err).sum()))|{'threshold':t}
    sens={}
    for name in SCORES:
        per=[r for r in p1[f'{name}__crc__{PRIMARY:.2f}']['per_speaker'] if r['speaker'] not in PILOT];sens[name]=summarize(per,PRIMARY)
    save(OUT/'thresholds.json',thresholds)
    save(OUT/'summary.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'arm':p['arm'],'speakers':24,'positions':int(len(gold)),
         'errors':int((gold<2).sum()),'P1_loso':p1,'endpoints':endpoints,'P2_leave_one_l1_out':p2,'P3_ranking':p3,'P4_standard_test_context':p4,
         'sensitivity_non_pilot_18':sens,'new_labels':0,'l2arctic_training':False,
         'limitations':['Marginal speaker-level guarantee; LOSO splits are not independent','4 speakers per L1 in P2',
                        'EpaDB-trained head: domain shift','single annotator labels','P4 not directly comparable with published systems']})
    for k,v in p1.items():
        if k.endswith(f'{PRIMARY:.2f}'):print(k,round(v['mean_fpr'],4),v['speakers_over_alpha'],round(v['mean_recall'],4))
    print({k:(round(v['mean'],4),[round(x,4) for x in v['ci95']],round(v['holm_p'],4)) for k,v in endpoints.items()},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['labels','lock','score','evaluate']);args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);globals()[args.phase]()
