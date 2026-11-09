"""SSL-MD v3: 60-epoch convergence study and uniform-layer control on the sealed v2 caches. Development only."""
import argparse
from datetime import datetime, timezone
import math
import time
import numpy as np
from . import sslmd_smoke as s
from . import sslmd_training_v2 as v2

ROOT=s.ROOT;OUT=ROOT/'06_experiment_results/sslmd_convergence_v3'
PRIVATE=ROOT/'data/derived/sslmd_convergence_v3'
CONTRACT=ROOT/'05_methodology/sslmd_convergence_v3_contract_2026-11-08.md'
ARMS=('last','mixed','uniform');CAPS=(20,60)
read,save,sha=s.read,s.save,s.sha


def make_head(arm,dimension=768):
    """Same linear head for every arm; only the layer reduction differs."""
    import torch
    if arm in ('last','mixed'):return s.make_head(arm=='mixed',dimension)
    if arm!='uniform':raise ValueError(arm)
    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(dimension,39)
        def forward(self,layers):
            if layers.ndim!=3 or layers.shape[0]!=12 or layers.shape[2]!=dimension:
                raise ValueError('12 by frames by dimension required')
            return self.linear(layers.mean(0))
    return Uniform()


def pick_capped(history,cap):
    """Best calibration pAUC among epochs 1..cap, earliest on ties (v2 rule)."""
    return v2.pick_epoch(history[:cap])


def protocol():
    p=read(OUT/'locked_protocol.json')
    for rel,h in p['inputs_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p


def lock():
    assert not (OUT/'locked_protocol.json').exists()
    old=v2.protocol();features=read(v2.OUT/'features_locked.json');fits=read(v2.OUT/'fits_locked.json')
    files=[ROOT/'code/vpa/sslmd_convergence_v3.py',ROOT/'code/tests/test_sslmd_convergence_v3.py',CONTRACT,
           ROOT/'code/vpa/sslmd_training_v2.py',ROOT/'code/vpa/sslmd_smoke.py',
           v2.OUT/'locked_protocol.json',v2.OUT/'features_locked.json',v2.OUT/'fits_locked.json',
           v2.OUT/'summary.json',s.PRIVATE/'frame_mapping.json',OUT/'software_tests.json']
    files+=[ROOT/r['path'] for r in features['rows']]+[ROOT/rel for rel in fits['files_sha256']]
    save(OUT/'locked_protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'inputs_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))},
        'folds':old['folds'],'seeds':old['seeds'],'arms':list(ARMS),'max_epochs':60,'caps':list(CAPS),
        'inherited':'v2 caches, targets, folds, seeds, optimizer, weighting, ordering, selection, thresholds, ensemble, bootstrap; v2 prior fits reused by hash',
        'changes':'uniform 12-layer mean arm; 60-epoch cap with cap20 and cap60 checkpoints per fit',
        'endpoints':{'E1':'mixed60-uniform60 pAUC','E2':'uniform60-last60 pAUC','E3':'cap60-cap20 per arm; share of fits selecting epoch>=55',
                     'secondary':'mixed60-last60, mixed60-prior; conditional FPR, recall, ActCost at empirical thresholds'},
        'reproduction_check':'last/mixed cap20 histories and selections compared to v2; deviations reported, not used to select',
        'statistics':old['statistics'],'training_wall_budget_seconds':16200,'fits_expected':75,
        'scope':'Retrospective10speaker development; not fresh confirmation','no_test_or_reserved_values':True})
    print('v3 protocol locked.',flush=True)


def fit():
    p=protocol();assert not (OUT/'fits_locked.json').exists();v2.guard()
    torch=v2.configure_torch();features=read(v2.OUT/'features_locked.json')
    paths={r['row_id']:r['path'] for r in features['rows']}
    rows=read(s.PRIVATE/'frame_mapping.json');started=time.perf_counter();completed=[]
    def check_budget():
        v2.guard()
        if time.perf_counter()-started>p['training_wall_budget_seconds']:raise RuntimeError('Predeclared training wall budget exhausted; completed fits retained')
    for fold in p['folds']:
        fid=fold['fold'];parts={k:[r for r in rows if r['speaker'] in fold[k]] for k in ('fit','calibration','evaluation')}
        denominators=v2.fit_counts(rows,fold['fit']);n=len(parts['fit'])
        for seed in p['seeds']:
            for arm in ARMS:
                dest=PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}.json'
                if dest.exists():
                    old=read(dest)
                    for cap in CAPS:assert sha(ROOT/old['caps'][str(cap)]['checkpoint_path'])==old['caps'][str(cap)]['checkpoint_sha256']
                    completed.append(dest);continue
                check_budget();torch.manual_seed(seed);head=make_head(arm);head.train()
                optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
                rng=np.random.default_rng(seed+10000*fid);history=[];best={c:(-math.inf,None,None) for c in CAPS}
                tick=time.perf_counter()
                for epoch in range(1,p['max_epochs']+1):
                    order=rng.permutation(n);epoch_loss=0.;head.train()
                    for offset in range(0,n,4):
                        check_budget();indices=order[offset:offset+4];optimizer.zero_grad(set_to_none=True)
                        for index in indices:
                            r=parts['fit'][int(index)];h=v2.tensors(r,paths,torch)
                            m=s.frame_map(r['phones'],r['labels'],r['ranges'],r['frames'])
                            valid=np.flatnonzero(m['valid']);out=head(h)
                            loss=torch.nn.functional.binary_cross_entropy_with_logits(out[valid,m['phone'][valid]],torch.as_tensor(m['target'][valid],dtype=out.dtype),reduction='sum')
                            weighted=loss/(len(denominators)*denominators[r['speaker']])
                            epoch_loss+=float(weighted.detach());(weighted*(n/len(indices))).backward()
                            del h,out,loss,weighted
                        optimizer.step()
                    value=v2.macro_pauc(v2.score(head,parts['calibration'],paths,torch))
                    history.append({'epoch':epoch,'fit_equal_speaker_frame_BCE':epoch_loss,'calibration_pauc':value})
                    for cap in CAPS:
                        if epoch<=cap and value>best[cap][0]:
                            best[cap]=(value,epoch,{k:v.detach().clone() for k,v in head.state_dict().items()})
                    if epoch%10==0:print({'fold':fid,'seed':seed,'arm':arm,'epoch':epoch,'seconds':round(time.perf_counter()-tick,1)},flush=True)
                record={'protocol_sha256':sha(OUT/'locked_protocol.json'),'fold':fid,'arm':arm,'seed':seed,
                        'history':history,'fit_frame_counts':denominators,'caps':{}}
                for cap in CAPS:
                    _,epoch,state=best[cap];assert epoch==history[pick_capped(history,cap)]['epoch']
                    head.load_state_dict(state);checkpoint=PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}_cap{cap}.npz'
                    checkpoint.parent.mkdir(parents=True,exist_ok=True);np.savez(checkpoint,**{k:v.numpy() for k,v in state.items()})
                    record['caps'][str(cap)]={'selected_epoch':epoch,'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),
                        'checkpoint_sha256':sha(checkpoint),'calibration':v2.score(head,parts['calibration'],paths,torch),
                        'evaluation':v2.score(head,parts['evaluation'],paths,torch),
                        'layer_weights':state['mix'].softmax(0).tolist() if arm=='mixed' else None}
                record['seconds']=time.perf_counter()-tick;save(dest,record);completed.append(dest)
                print({'completed_fits':len(completed),'expected':75,'fold':fid,'seed':seed,'arm':arm,'cap60_epoch':record['caps']['60']['selected_epoch']},flush=True)
                del head,optimizer,best,record
    assert len(completed)==75
    save(OUT/'fits_locked.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'files_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in completed},
         'neural_fits':75,'seconds_this_invocation':time.perf_counter()-started,'resources_end':s.resources()})
    print('All75v3 fits complete.',flush=True)


def reproduction(p):
    """Compare cap20 last/mixed with the sealed v2 fits."""
    worst_pauc=0.;mismatched=[];checked=0
    for fold in p['folds']:
        for seed in p['seeds']:
            for arm in ('last','mixed'):
                new=read(PRIVATE/'fits'/f"fold{fold['fold']}_{arm}_{seed}.json");old=read(v2.PRIVATE/'fits'/f"fold{fold['fold']}_{arm}_{seed}.json")
                a=np.array([h['calibration_pauc'] for h in new['history'][:20]]);b=np.array([h['calibration_pauc'] for h in old['history']])
                worst_pauc=max(worst_pauc,float(np.abs(a-b).max()));checked+=1
                if new['caps']['20']['selected_epoch']!=old['selected_epoch']:mismatched.append(f"fold{fold['fold']}_{arm}_{seed}")
    return {'fits_compared':checked,'max_abs_calibration_pauc_difference_epochs1_20':worst_pauc,'selected_epoch_mismatches':mismatched}


def evaluate():
    p=protocol();assert not (OUT/'summary.json').exists();fits=read(OUT/'fits_locked.json')
    assert fits['neural_fits']==75
    for rel,h in fits['files_sha256'].items():assert sha(ROOT/rel)==h,rel
    held={};choices={};selected={}
    for fold in p['folds']:
        fid=fold['fold'];prior=read(v2.PRIVATE/'fits'/f'fold{fid}_prior.json')
        flags,choice=v2.attach_thresholds(v2.ensemble([prior['calibration']]),v2.ensemble([prior['evaluation']]));held.setdefault('prior',[]).extend(flags);choices[f'{fid}__prior']=choice
        for arm in ARMS:
            rr=[read(PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}.json') for seed in p['seeds']]
            for cap in CAPS:
                key=f'{arm}{cap}';c=[r['caps'][str(cap)] for r in rr]
                for r,x in zip(rr,c):
                    assert r['protocol_sha256']==sha(OUT/'locked_protocol.json') and x['selected_epoch']==r['history'][pick_capped(r['history'],cap)]['epoch']
                selected.setdefault(key,[]).extend(x['selected_epoch'] for x in c)
                flags,choice=v2.attach_thresholds(v2.ensemble([x['calibration'] for x in c]),v2.ensemble([x['evaluation'] for x in c]))
                held.setdefault(key,[]).extend(flags);choices[f'{fid}__{key}']=choice
    keys={(r['row_id'],r['slot']):(r['phone'],r['y']) for r in held['mixed60']};assert len(keys)==10013
    for arm,rr in held.items():assert {(r['row_id'],r['slot']):(r['phone'],r['y']) for r in rr}==keys,arm
    allmetrics={arm:v2.metrics(rr) for arm,rr in held.items()}
    rng=np.random.default_rng(p['statistics']['bootstrap_seed']);draws=rng.integers(0,10,(p['statistics']['draws'],10))
    def contrast(a,b):
        out={}
        for metric in ('pauc','conditional_fpr','recall'):
            d=np.array([r[metric] for r in allmetrics[a]['people']])-np.array([r[metric] for r in allmetrics[b]['people']])
            out[metric]={'delta':float(d.mean()),'ci95':np.quantile(d[draws].mean(1),[.025,.975]).tolist(),'positive_people':int((d>0).sum())}
        return out
    pairs={'E1_mixed60_minus_uniform60':('mixed60','uniform60'),'E2_uniform60_minus_last60':('uniform60','last60'),
           'E3_last60_minus_last20':('last60','last20'),'E3_mixed60_minus_mixed20':('mixed60','mixed20'),
           'E3_uniform60_minus_uniform20':('uniform60','uniform20'),'mixed60_minus_last60':('mixed60','last60'),
           'mixed60_minus_prior':('mixed60','prior'),'uniform20_minus_last20':('uniform20','last20'),'mixed20_minus_uniform20':('mixed20','uniform20')}
    curves={arm:np.mean([[h['calibration_pauc'] for h in read(PRIVATE/'fits'/f"fold{f['fold']}_{arm}_{seed}.json")['history']]
                         for f in p['folds'] for seed in p['seeds']],0).tolist() for arm in ARMS}
    v2summary=read(v2.OUT/'summary.json')
    save(PRIVATE/'held_predictions.json',held);save(OUT/'thresholds.json',choices)
    save(OUT/'summary.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'metrics':allmetrics,
         'contrasts':{k:contrast(a,b) for k,(a,b) in pairs.items()},
         'selected_epochs':{k:{'values':v,'median':float(np.median(v)),'at_or_after_55':int(sum(e>=55 for e in v)),'at_cap20':int(sum(e==20 for e in v))} for k,v in selected.items()},
         'mean_calibration_pauc_curve':curves,'reproduction_vs_v2':reproduction(p),
         'v2_reference_pauc':{k:v2summary['metrics'][k]['means']['pauc'] for k in ('last','mixed','prior')},
         'known_held_positions':10013,'unknown_positions':453,'neural_fits':75,'fresh_validation_people':0,'new_labels':0,
         'limitations':['Retrospective10speaker development reused from v2; no fresh confirmation',
                        'Calibration speakers both select epochs and thresholds, as in v2',
                        'Fixed-model speaker bootstrap excludes training/calibration uncertainty; no multiplicity adjustment',
                        'Empirical ActCost thresholds, not CRC']})
    print({arm:round(m['means']['pauc'],6) for arm,m in allmetrics.items()},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['lock','fit','evaluate']);args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);globals()[args.phase]()
