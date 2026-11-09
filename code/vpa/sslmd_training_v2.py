"""Streamed, pinned development SSL-MD baseline. No reserved/final inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import time
import numpy as np
from . import sslmd_smoke as s

ROOT=s.ROOT; OUT=ROOT/'06_experiment_results/sslmd_training_v2'
PRIVATE=ROOT/'data/derived/sslmd_training_v2'
CONTRACT=ROOT/'05_methodology/sslmd_streamed_training_v2_contract_2026-11-07.md'
read,save,sha=s.read,s.save,s.sha


def check_folds(folds,people):
    held=[]
    for f in folds:
        a,b,c=(set(f[k]) for k in ('fit','calibration','evaluation'))
        if (len(a),len(b),len(c))!=(6,2,2) or a&b or a&c or b&c or a|b|c!=set(people):
            raise ValueError('invalid or overlapping speaker fold')
        held.extend(c)
    if sorted(held)!=sorted(people):raise ValueError('each speaker must be evaluated once')


def fit_counts(rows,people):
    totals=Counter()
    for r in rows:
        if r['speaker'] in people:totals[r['speaker']]+=r['eligible_frames']
    if set(totals)!=set(people) or min(totals.values())<=0:raise ValueError('missing fit supervision')
    return dict(totals)


def threshold(y,q):
    y=np.asarray(y);q=np.asarray(q,float)
    if set(y)!={0,1} or y.shape!=q.shape or not np.isfinite(q).all():raise ValueError('two classes required')
    grid=np.unique(np.r_[np.nextafter(q.min(),-np.inf),q])
    good=np.sort(q[y==0]);bad=np.sort(q[y==1]);ng=len(good);nb=len(bad)
    fp=ng-np.searchsorted(good,grid,side='right');fn=np.searchsorted(bad,grid,side='right')
    cost=fn*ng+2*fp*nb;i=np.flatnonzero(cost==cost.min())[-1]
    return float(grid[i])


def macro_pauc(rows):
    from sklearn.metrics import roc_auc_score
    people=sorted({r['speaker'] for r in rows});values=[]
    for person in people:
        rr=[r for r in rows if r['speaker']==person and r['y']>=0]
        if {r['y'] for r in rr}!={0,1}:raise ValueError('speaker ranking needs both classes')
        values.append(roc_auc_score([r['y'] for r in rr],[r['score'] for r in rr],max_fpr=.1))
    return float(np.mean(values))


def pick_epoch(history):
    return max(range(len(history)),key=lambda i:(history[i]['calibration_pauc'],-i))


def protocol():
    p=read(OUT/'locked_protocol.json')
    for rel,h in p['inputs_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p


def guard(preload=False):
    r=s.resources();minimum=(3.5 if preload else 2)*s.GIB
    if r['available_ram_bytes']<minimum:
        raise RuntimeError('Resource gate: available RAM %.3fGiB < %.1fGiB'%(r['available_ram_bytes']/s.GIB,minimum/s.GIB))
    return r


def lock():
    assert not (OUT/'locked_protocol.json').exists();s.protocol()
    budget=read(s.OUT/'locked_training_budget.json');refs=read(s.REFS)
    check_folds(budget['folds'],{r['speaker'] for r in refs})
    files=[Path(__file__),ROOT/'code/tests/test_sslmd_training_v2.py',CONTRACT,
           s.OUT/'locked_training_budget.json',s.OUT/'delivery_manifest_sha256.json',
           s.OUT/'locked_smoke_protocol.json',s.OUT/'extraction_complete.json',
           s.PRIVATE/'frame_mapping.json',s.REFS,s.LABELS,s.PREDS,s.PARQUET,
           ROOT/'data/derived/epadb_admission_v1/speaker_roles.json',
           ROOT/'data/derived/epadb_matched_head_v1/held_predictions.json',OUT/'software_tests.json']
    legacy=ROOT/'06_experiment_results/sslmd_training_v1/delivery_manifest_sha256.json'
    files += [legacy] + [ROOT/rel for rel in read(legacy)['files_sha256']]
    files += [ROOT/rel for rel in read(s.OUT/'locked_smoke_protocol.json')['inputs_sha256']]
    for r in read(s.OUT/'extraction_complete.json')['rows']:
        files += [s.PRIVATE/'hidden'/(r['row_id'].replace(':','_')+suffix) for suffix in ('.npy','.json')]
    save(OUT/'locked_protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'inputs_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))},
        'folds':budget['folds'],'seeds':budget['seeds'],'max_epochs':20,
        'scope':'Retrospective10speaker development; not fresh confirmation/full published reproduction',
        'target':'Existing all-path-certified author_first canonical positions; unknown masked; no regenerated source labels',
        'normalization':'No learned centering/scaling; unchanged native hidden representations shared by arms',
        'weighting':'Equal fit-speaker frame-BCE totals; group estimator Nfitrows/groupSize times summed per-frame loss/(6*speakerEligibleFrames)',
        'optimizer':budget['optimizer_proposal'],
        'prior':'39onehot logistic, no intercept, L2/C1, aggregated frame class counts, equal speaker-frame weights normalized mean1, lbfgs tol1e-10 max2000; five deterministic fits',
        'selection':'maximum2cal-speaker-macro pAUC, earliest tied epoch; held speakers excluded',
        'threshold':'strict score>t; per-phone cal FN_error/Nerror+2FP_error/Nacceptable minimum, largest tie; global cal fallback when phone lacks a class; finite allwarn and nowarn choices',
        'primary':'Mixed-minus-last and mixed-minus-frameprior speaker-macro pAUC; same held positions',
        'ensemble':'Average paired5seed logits per held position; choose empirical thresholds on the corresponding average5seed cal logits, no held tuning',
        'statistics':{'bootstrap_seed':3191,'draws':5000,'unit':'10paired held-development speakers','limits':'conditional on fitted models, selected epochs and calibrators; not training/calibration/prompt uncertainty'},
        'resource_amendment':'Streaming batch1; >=3.5GiB before encoder load, >=2GiB before each utterance/batch; separate encoder/head processes;4threads; original4GiB policy superseded by pinned compute contract only',
        'training_wall_budget_seconds':7200,'fits_expected':{'neural':50,'deterministic_prior':5},
        'numeric_QC_amendment':'V1retained; posterior absolute tolerance1e-5, exact canonical ranges, argmax/mean-priority deltas reported; no performance-based change',
        'no_test_or_reserved_values':True})
    print('Full streamed development protocol locked.',flush=True)


def posterior_qc(prob,old,m):
    from .phone_alignment import posterior_align
    if prob.shape!=old.shape or not np.isfinite(prob).all():raise ValueError('invalid emission array')
    delta=float(np.abs(prob-old).max())
    if delta>1e-5:raise ValueError('posterior difference exceeds1e-5')
    codes=[p+1 for p in m['phones']]
    ranges=posterior_align(prob,codes)['phone_frame_ranges']
    if ranges!=m['ranges']:raise ValueError('canonical alignment changed')
    canonical=[float(abs(prob[lo:hi,c].astype(float).mean()-old[lo:hi,c].astype(float).mean()))
               for c,(lo,hi) in zip(codes,m['ranges'],strict=True)]
    return {'max_posterior_difference':delta,'max_canonical_mean_difference':max(canonical),
            'argmax_frame_changes':int((prob.argmax(1)!=old.argmax(1)).sum()),'canonical_ranges_identical':True}


def configure_torch():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(1729);torch.use_deterministic_algorithms(True)
    return torch


def extract():
    p=protocol();assert not (OUT/'features_locked.json').exists()
    maps=read(s.PRIVATE/'frame_mapping.json');initial=guard(True)
    remaining=sum(m['frames']*12*768*4 for m in maps if not (s.PRIVATE/'hidden'/(m['row_id'].replace(':','_')+'.npy')).exists())
    assert initial['free_disk_bytes']-remaining>=10*s.GIB
    torch=configure_torch()
    import pyarrow.parquet as pq
    from scipy.io import wavfile
    from .external_hubert import load_model,conv_output_length
    model=load_model(s.MODEL);transformer=model.model.wav2vec2.encoder.transformer
    assert transformer.layer_norm_first and all(not x.requires_grad for x in model.parameters())
    refs=read(s.REFS);people=sorted({r['speaker'] for r in refs})
    roles=read(ROOT/'data/derived/epadb_admission_v1/speaker_roles.json')
    table=pq.read_table(s.PARQUET,columns=['audio','speaker_code'],filters=[('speaker_code','in',people)]).to_pylist()
    assert len(table)==len(refs)==634
    audio={f'train:{i}':a for i,a in zip(roles['mapping_train_row_indices'],table,strict=True)}
    receipts=[];started=time.perf_counter();fresh=0
    for m in maps:
        key=m['row_id'];stem=key.replace(':','_');target=PRIVATE/'hidden'/(stem+'.npy');receipt=target.with_suffix('.json')
        old=s.PRIVATE/'hidden'/(stem+'.npy')
        if not old.exists():old=ROOT/'data/derived/sslmd_training_v1/hidden'/(stem+'.npy')
        if old.exists():
            item=read(old.with_suffix('.json'));assert sha(old)==item['sha256'];receipts.append(item|{'path':old.relative_to(ROOT).as_posix(),'origin':'smoke' if 'sslmd_smoke_v1' in old.parts else 'v1_reused'});continue
        if receipt.exists():
            item=read(receipt);assert sha(target)==item['sha256'];receipts.append(item);continue
        assert not target.exists(),'unreceipted feature cache requires review'
        guard();tick=time.perf_counter();a=audio[key];assert a['speaker_code']==m['speaker']
        rate,pcm=wavfile.read(io.BytesIO(a['audio']['bytes']));assert rate==16000 and pcm.dtype==np.int16 and pcm.ndim==1
        wave=torch.from_numpy(np.ascontiguousarray(pcm.astype(np.float32)/32768.))[None]
        captured=[]
        def hook(_module,_inputs,output):captured.append(output[0].detach())
        hooks=[layer.register_forward_hook(hook) for layer in transformer.layers]
        try:
            with torch.inference_mode():
                logits=model(wave);assert len(captured)==12
                assert all(tuple(t.shape)==(1,m['frames'],768) for t in captured)
                assert m['frames']==conv_output_length(len(pcm))
                assert torch.equal(model.aux(captured[-1]),logits)
                features=torch.stack([t[0] for t in captured]).numpy();prob=logits.softmax(-1)[0].numpy()
        finally:
            for h in hooks:h.remove()
        with np.load(s.DEV/'cache/hubert'/(stem+'.npz'),allow_pickle=False) as z:old_prob=z['probabilities']
        qc=posterior_qc(prob,old_prob,m);delta=qc['max_posterior_difference'];assert np.isfinite(features).all()
        target.parent.mkdir(parents=True,exist_ok=True);np.save(target,features,allow_pickle=False)
        item={'row_id':key,'speaker':m['speaker'],'path':target.relative_to(ROOT).as_posix(),'origin':'full',
              'shape':list(features.shape),'sha256':sha(target),'bytes':target.stat().st_size,
              'max_posterior_difference':delta,'QC':qc,'seconds':time.perf_counter()-tick,'resources_after':s.resources()}
        save(receipt,item);receipts.append(item);fresh+=1
        del captured,features,logits,prob,wave
        if fresh%32==0:print({'features_total':len(receipts),'fresh':fresh,'seconds':round(time.perf_counter()-started,1)},flush=True)
    assert len(receipts)==634
    save(OUT/'features_locked.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'rows':receipts,
         'fresh_in_this_invocation':fresh,'new_v2_feature_rows':sum(r['origin']=='full' for r in receipts),'v1_reused_rows':sum(r['origin']=='v1_reused' for r in receipts),'smoke_reused_rows':20,
         'seconds':time.perf_counter()-started,'resources_end':s.resources()})
    print('All634frozen feature caches sealed.',flush=True)


def tensors(row,paths,torch):
    # np.load owns one utterance buffer; no persistent full-cohort tensor cache.
    return torch.from_numpy(np.load(ROOT/paths[row['row_id']],allow_pickle=False))


def score(head,rows,paths,torch):
    records=[];head.eval()
    with torch.no_grad():
        for r in rows:
            h=tensors(r,paths,torch);out=head(h).numpy()
            values=s.pool_logits(out,r['phones'],r['ranges'])
            for j,(y,phone,q) in enumerate(zip(r['labels'],r['phones'],values,strict=True)):
                if q is not None:records.append({'row_id':r['row_id'],'speaker':r['speaker'],'slot':j,'phone':phone,'y':y,'score':q})
            del h,out
    return records


def prior_fit(rows,fit_people):
    from sklearn.linear_model import LogisticRegression
    counts=fit_counts(rows,fit_people);mass=np.zeros((39,2));total=sum(counts.values())
    for r in rows:
        if r['speaker'] not in fit_people:continue
        for phone,y,(lo,hi) in zip(r['phones'],r['labels'],r['ranges'],strict=True):
            if y>=0:mass[phone,y]+=(hi-lo)*total/(len(counts)*counts[r['speaker']])
    xx=[];yy=[];ww=[]
    for phone in range(39):
        for y in (0,1):
            if mass[phone,y]>0:xx.append(np.eye(39)[phone]);yy.append(y);ww.append(mass[phone,y])
    model=LogisticRegression(C=1.,fit_intercept=False,solver='lbfgs',tol=1e-10,max_iter=2000)
    model.fit(xx,yy,sample_weight=ww);assert int(model.n_iter_[0])<2000
    return {'coef':model.coef_[0].tolist(),'iterations':int(model.n_iter_[0]),'mass':mass.tolist(),'fit_frame_counts':counts}


def prior_score(prior,rows):
    return [{'row_id':r['row_id'],'speaker':r['speaker'],'slot':j,'phone':phone,'y':y,'score':prior['coef'][phone]}
            for r in rows for j,(phone,y) in enumerate(zip(r['phones'],r['labels'],strict=True))]


def fit():
    p=protocol();assert not (OUT/'fits_locked.json').exists();guard()
    torch=configure_torch();features=read(OUT/'features_locked.json')
    assert features['protocol_sha256']==sha(OUT/'locked_protocol.json')
    paths={r['row_id']:r['path'] for r in features['rows']}
    for r in features['rows']:assert sha(ROOT/r['path'])==r['sha256']
    rows=read(s.PRIVATE/'frame_mapping.json');started=time.perf_counter();completed=[]
    def check_budget():
        guard()
        if time.perf_counter()-started>p['training_wall_budget_seconds']:raise RuntimeError('Predeclared training wall budget exhausted; incomplete fits retained')
    for fold in p['folds']:
        fid=fold['fold'];parts={k:[r for r in rows if r['speaker'] in fold[k]] for k in ('fit','calibration','evaluation')}
        pp=PRIVATE/'fits'/f'fold{fid}_prior.json'
        if not pp.exists():
            check_budget();prior=prior_fit(rows,fold['fit'])
            save(pp,{'protocol_sha256':sha(OUT/'locked_protocol.json'),'fold':fid,'arm':'prior','seed':None,'fit':prior,
                     'calibration':prior_score(prior,parts['calibration']),
                     'evaluation':prior_score(prior,parts['evaluation'])})
        completed.append(pp);denominators=fit_counts(rows,fold['fit']);n=len(parts['fit'])
        for seed in p['seeds']:
            for arm in ('last','mixed'):
                dest=PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}.json';checkpoint=dest.with_suffix('.npz')
                if dest.exists():
                    old=read(dest);assert sha(checkpoint)==old['checkpoint_sha256'];completed.append(dest);continue
                check_budget();torch.manual_seed(seed);head=s.make_head(arm=='mixed');head.train()
                optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
                rng=np.random.default_rng(seed+10000*fid);history=[];best=None;best_metric=-math.inf;best_epoch=None
                tick=time.perf_counter()
                for epoch in range(1,p['max_epochs']+1):
                    order=rng.permutation(n);epoch_loss=0.;head.train()
                    for offset in range(0,n,4):
                        check_budget();indices=order[offset:offset+4];optimizer.zero_grad(set_to_none=True)
                        for index in indices:
                            r=parts['fit'][int(index)];h=tensors(r,paths,torch)
                            m=s.frame_map(r['phones'],r['labels'],r['ranges'],r['frames'])
                            valid=np.flatnonzero(m['valid']);out=head(h)
                            loss=torch.nn.functional.binary_cross_entropy_with_logits(out[valid,m['phone'][valid]],torch.as_tensor(m['target'][valid],dtype=out.dtype),reduction='sum')
                            weighted=loss/(len(denominators)*denominators[r['speaker']])
                            epoch_loss+=float(weighted.detach());(weighted*(n/len(indices))).backward()
                            del h,out,loss,weighted
                        optimizer.step()
                    cal=score(head,parts['calibration'],paths,torch);value=macro_pauc(cal)
                    history.append({'epoch':epoch,'fit_equal_speaker_frame_BCE':epoch_loss,'calibration_pauc':value})
                    if value>best_metric:
                        best_metric=value;best_epoch=epoch;best={k:v.detach().clone() for k,v in head.state_dict().items()}
                    if epoch%5==0:print({'fold':fid,'seed':seed,'arm':arm,'epoch':epoch,'seconds':round(time.perf_counter()-tick,1)},flush=True)
                assert best_epoch==history[pick_epoch(history)]['epoch'];head.load_state_dict(best)
                cal=score(head,parts['calibration'],paths,torch);ev=score(head,parts['evaluation'],paths,torch)
                checkpoint.parent.mkdir(parents=True,exist_ok=True);np.savez(checkpoint,**{k:v.numpy() for k,v in best.items()})
                save(dest,{'protocol_sha256':sha(OUT/'locked_protocol.json'),'fold':fid,'arm':arm,'seed':seed,
                     'selected_epoch':best_epoch,'history':history,'fit_frame_counts':denominators,
                     'checkpoint_sha256':sha(checkpoint),'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),
                     'calibration':cal,'evaluation':ev,'seconds':time.perf_counter()-tick,
                     'layer_weights':best['mix'].softmax(0).tolist() if arm=='mixed' else None})
                completed.append(dest)
                print({'completed_fits':len(completed),'expected':55,'fold':fid,'seed':seed,'arm':arm},flush=True)
                del head,optimizer,best,cal,ev
    assert len(completed)==55
    save(OUT/'fits_locked.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'files_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in completed},
         'neural_fits':50,'deterministic_prior_fits':5,'seconds_this_invocation':time.perf_counter()-started,'resources_end':s.resources()})
    print('All55development fits complete.',flush=True)


def attach_thresholds(cal,held):
    cal=[r for r in cal if r['y']>=0]
    global_t=threshold([r['y'] for r in cal],[r['score'] for r in cal]);choices={}
    for phone in range(39):
        rr=[r for r in cal if r['phone']==phone]
        both={r['y'] for r in rr}=={0,1}
        choices[phone]={'threshold':threshold([r['y'] for r in rr],[r['score'] for r in rr]) if both else global_t,
                        'global_fallback':not both,'cal_positions':len(rr)}
    return [r|{'flag':bool(r['score']>choices[r['phone']]['threshold'])} for r in held if r['y']>=0],choices


def ensemble(records):
    table={}
    for rows in records:
        for r in rows:table.setdefault((r['row_id'],r['slot']),[]).append(r)
    result=[]
    for key,rr in sorted(table.items()):
        assert len(rr)==len(records)
        assert all((r['speaker'],r['phone'],r['y'])==(rr[0]['speaker'],rr[0]['phone'],rr[0]['y']) for r in rr)
        result.append(rr[0]|{'score':float(np.mean([r['score'] for r in rr]))})
    return result


def metrics(rows):
    from sklearn.metrics import roc_auc_score,average_precision_score
    people=[]
    for person in sorted({r['speaker'] for r in rows}):
        rr=[r for r in rows if r['speaker']==person];y=np.array([r['y'] for r in rr]);q=np.array([r['score'] for r in rr]);f=np.array([r['flag'] for r in rr])
        good=y==0;bad=y==1;fp=int((f&good).sum());tp=int((f&bad).sum())
        people.append({'speaker':person,'n':len(rr),'pauc':float(roc_auc_score(y,q,max_fpr=.1)),
            'auroc':float(roc_auc_score(y,q)),'ap':float(average_precision_score(y,q)),
            'conditional_fpr':fp/int(good.sum()),'recall':tp/int(bad.sum()),
            'precision':tp/int(f.sum()) if f.any() else None,'flags':int(f.sum())})
    phones={}
    for phone in range(39):
        rr=[r for r in rows if r['phone']==phone];ng=sum(r['y']==0 for r in rr);nb=sum(r['y']==1 for r in rr)
        fp=sum(r['flag'] and r['y']==0 for r in rr);fn=sum(not r['flag'] and r['y']==1 for r in rr)
        phones[str(phone)]={'good':ng,'bad':nb,'fp':fp,'fn':fn,'eligible':min(ng,nb)>=50,
                            'ActCost':fn/nb+2*fp/ng if min(ng,nb)>0 else None}
    means={k:float(np.mean([r[k] for r in people if r[k] is not None])) for k in ('pauc','auroc','ap','conditional_fpr','recall','precision','flags')}
    eligible=[r['ActCost'] for r in phones.values() if r['eligible']]
    return {'people':people,'means':means,'phones':phones,'eligible_phones':len(eligible),'phone_macro_ActCost':float(np.mean(eligible)) if eligible else None}


def evaluate():
    p=protocol();assert not (OUT/'summary.json').exists();fits=read(OUT/'fits_locked.json')
    assert fits['neural_fits']==50 and fits['deterministic_prior_fits']==5
    for rel,h in fits['files_sha256'].items():assert sha(ROOT/rel)==h,rel
    held={a:[] for a in ('prior','last','mixed')};choices={};seed_rows={a:{str(seed):[] for seed in p['seeds']} for a in ('last','mixed')}
    for fold in p['folds']:
        fid=fold['fold']
        for arm in held:
            rr=[read(PRIVATE/'fits'/f'fold{fid}_prior.json')] if arm=='prior' else [read(PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}.json') for seed in p['seeds']]
            for r in rr:
                assert r['protocol_sha256']==sha(OUT/'locked_protocol.json')
                if arm!='prior':
                    assert r['selected_epoch']==r['history'][pick_epoch(r['history'])]['epoch']
                    seed_rows[arm][str(r['seed'])]+=r['evaluation']
            cal=ensemble([r['calibration'] for r in rr]);ev=ensemble([r['evaluation'] for r in rr])
            flags,choice=attach_thresholds(cal,ev);held[arm]+=flags;choices[f'{fid}__{arm}']=choice
    old=read(ROOT/'data/derived/epadb_matched_head_v1/held_predictions.json')
    for arm in ('raw','source79','pooled','phone_only','conditional79'):
        held['old_'+arm]=[{'row_id':r['row_id'],'speaker':r['speaker'],'slot':r['slot'],'phone':r['canonical']-1,'y':r['y'],'score':r['score'],'flag':r['flag']} for r in old if r['arm']==arm]
    primary_keys={(r['row_id'],r['slot']):(r['phone'],r['y']) for r in held['mixed']}
    assert len(primary_keys)==10013
    for arm,rows in held.items():assert {(r['row_id'],r['slot']):(r['phone'],r['y']) for r in rows}==primary_keys,arm
    allmetrics={arm:metrics(rr) for arm,rr in held.items()}
    rng=np.random.default_rng(3191);draws=rng.integers(0,10,(5000,10));contrasts={}
    for arm in held:
        if arm=='mixed':continue
        contrasts['mixed_minus_'+arm]={}
        for metric in ('pauc','conditional_fpr','recall'):
            delta=np.array([r[metric] for r in allmetrics['mixed']['people']])-np.array([r[metric] for r in allmetrics[arm]['people']])
            contrasts['mixed_minus_'+arm][metric]={'delta':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),'positive_people':int((delta>0).sum())}
    perseed={a:{seed:macro_pauc(rr) for seed,rr in seeds.items()} for a,seeds in seed_rows.items()}
    save(PRIVATE/'held_predictions.json',held);save(OUT/'thresholds.json',choices)
    save(OUT/'summary.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'metrics':allmetrics,'contrasts':contrasts,
         'seed_pauc':perseed,'known_held_positions':10013,'unknown_positions':453,
         'neural_fits':50,'deterministic_prior_fits':5,'new_v2_feature_rows':380,'reused_feature_rows':254,
         'scope':p['scope'],'fresh_validation_people':0,'new_labels':0,
         'limitations':['Retrospective10speaker development, shared-fold fittedmodels and empirical cal selection; no fresh method confirmation',
                       'Fixed-model speaker bootstrap does not include training/calibration uncertainty',
                       'Operational family-certified subset, not full-corpus physical pronunciation truth',
                       'Matched COML/automatic-posterior adaptation, not faithful published PR/MD/GOP-FT recipe',
                       'Old controls use different feature/objective weights; last/prior are current matched controls']})
    print({arm:m['means']['pauc'] for arm,m in allmetrics.items()},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['lock','extract','fit','evaluate']);args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);globals()[args.phase]()
