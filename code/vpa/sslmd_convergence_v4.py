"""SSL-MD v4: patience-based training to convergence (mixed, uniform) on the sealed v2 caches. Development only."""
import argparse
from datetime import datetime, timezone
import math
import time
import numpy as np
from . import sslmd_smoke as s
from . import sslmd_training_v2 as v2
from . import sslmd_convergence_v3 as v3

ROOT=s.ROOT;OUT=ROOT/'06_experiment_results/sslmd_convergence_v4'
PRIVATE=ROOT/'data/derived/sslmd_convergence_v4'
CONTRACT=ROOT/'05_methodology/sslmd_convergence_v4_contract_2026-11-08.md'
ARMS=('mixed','uniform');MAX_EPOCHS=200;PATIENCE=20;REF_CAP=60
read,save,sha=s.read,s.save,s.sha


def should_stop(history,patience=PATIENCE):
    """Stop once `patience` epochs have passed without a strictly better calibration pAUC."""
    if not history:return False
    best=v2.pick_epoch(history)
    return len(history)-1-best>=patience


def holm(pvalues):
    """Holm step-down adjusted p-values, same order as input."""
    p=np.asarray(pvalues,float);m=len(p);order=np.argsort(p,kind='stable');adj=np.empty(m);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(m-rank)*p[i]));adj[i]=running
    return adj.tolist()


def bootstrap_p(deltas_draws):
    """Two-sided bootstrap p for mean delta = 0, with the (count+1)/(B+1) convention."""
    d=np.asarray(deltas_draws);b=len(d)
    tail=min((d<=0).sum(),(d>=0).sum())
    return float(min(1.,2*(tail+1)/(b+1)))


def protocol():
    p=read(OUT/'locked_protocol.json')
    for rel,h in p['inputs_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p


def lock():
    assert not (OUT/'locked_protocol.json').exists()
    old=v2.protocol();features=read(v2.OUT/'features_locked.json');v3fits=read(v3.OUT/'fits_locked.json')
    files=[ROOT/'code/vpa/sslmd_convergence_v4.py',ROOT/'code/tests/test_sslmd_convergence_v4.py',CONTRACT,
           ROOT/'code/vpa/sslmd_convergence_v3.py',ROOT/'code/vpa/sslmd_training_v2.py',ROOT/'code/vpa/sslmd_smoke.py',
           v2.OUT/'locked_protocol.json',v2.OUT/'features_locked.json',v2.OUT/'fits_locked.json',
           v3.OUT/'locked_protocol.json',v3.OUT/'fits_locked.json',v3.OUT/'summary.json',
           s.PRIVATE/'frame_mapping.json',OUT/'software_tests.json']
    files+=[ROOT/r['path'] for r in features['rows']]+[ROOT/rel for rel in v3fits['files_sha256']]
    files+=[v2.PRIVATE/'fits'/f"fold{f['fold']}_prior.json" for f in old['folds']]
    save(OUT/'locked_protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'inputs_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))},
        'folds':old['folds'],'seeds':old['seeds'],'arms':list(ARMS),'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'reference_cap':REF_CAP,
        'inherited':'v2/v3 caches, targets, folds, seeds, optimizer, weighting, ordering, selection rule, thresholds, ensemble, bootstrap; v2 prior fits by hash',
        'changes':'arms mixed+uniform only; patience-20 stopping on strictly better calibration pAUC, cap 200; cap60 checkpoint kept for v3 comparison',
        'endpoints':{'E1':'mixedConv-uniformConv pAUC (primary)','E3a':'mixedConv-mixed60','E3b':'uniformConv-uniform60',
                     'multiplicity':'Holm over E1,E3a,E3b with two-sided bootstrap p','E2':'descriptive convergence: patience stops, cap hits, median selected/stop epochs',
                     'secondary':'mixedConv-prior; conditional FPR, recall, ActCost at empirical thresholds',
                     'baseline_rule':'gate-B baseline = mixed if E1 positive and Holm-significant, else uniform'},
        'reproduction_check':'epochs1-60 calibration histories compared with v3; deviations reported, not used to select',
        'statistics':old['statistics'],'training_wall_budget_seconds':28800,'fits_expected':50,
        'scope':'Retrospective10speaker development; not fresh confirmation','no_test_or_reserved_values':True})
    print('v4 protocol locked.',flush=True)


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
                    for key in ('conv','60'):assert sha(ROOT/old['caps'][key]['checkpoint_path'])==old['caps'][key]['checkpoint_sha256']
                    completed.append(dest);continue
                check_budget();torch.manual_seed(seed);head=v3.make_head(arm);head.train()
                optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
                rng=np.random.default_rng(seed+10000*fid);history=[];best={'conv':(-math.inf,None,None),'60':(-math.inf,None,None)}
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
                    for key,limit in (('conv',p['max_epochs']),('60',p['reference_cap'])):
                        if epoch<=limit and value>best[key][0]:
                            best[key]=(value,epoch,{k:v.detach().clone() for k,v in head.state_dict().items()})
                    if epoch%20==0:print({'fold':fid,'seed':seed,'arm':arm,'epoch':epoch,'seconds':round(time.perf_counter()-tick,1)},flush=True)
                    if epoch>=p['reference_cap'] and should_stop(history,p['patience']):break
                record={'protocol_sha256':sha(OUT/'locked_protocol.json'),'fold':fid,'arm':arm,'seed':seed,
                        'history':history,'stopped_epoch':len(history),'stopped_by_patience':should_stop(history,p['patience']),
                        'fit_frame_counts':denominators,'caps':{}}
                for key in ('conv','60'):
                    _,epoch,state=best[key]
                    limit=len(history) if key=='conv' else p['reference_cap']
                    assert epoch==history[v2.pick_epoch(history[:limit])]['epoch']
                    head.load_state_dict(state);checkpoint=PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}_{key}.npz'
                    checkpoint.parent.mkdir(parents=True,exist_ok=True);np.savez(checkpoint,**{k:v.numpy() for k,v in state.items()})
                    record['caps'][key]={'selected_epoch':epoch,'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),
                        'checkpoint_sha256':sha(checkpoint),'calibration':v2.score(head,parts['calibration'],paths,torch),
                        'evaluation':v2.score(head,parts['evaluation'],paths,torch),
                        'layer_weights':state['mix'].softmax(0).tolist() if arm=='mixed' else None}
                record['seconds']=time.perf_counter()-tick;save(dest,record);completed.append(dest)
                print({'completed_fits':len(completed),'expected':50,'fold':fid,'seed':seed,'arm':arm,
                       'conv_epoch':record['caps']['conv']['selected_epoch'],'stopped':record['stopped_epoch'],'patience':record['stopped_by_patience']},flush=True)
                del head,optimizer,best,record
    assert len(completed)==50
    save(OUT/'fits_locked.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'files_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in completed},
         'neural_fits':50,'seconds_this_invocation':time.perf_counter()-started,'resources_end':s.resources()})
    print('All50v4 fits complete.',flush=True)


def reproduction(p):
    worst=0.;mismatched=[];checked=0
    for fold in p['folds']:
        for seed in p['seeds']:
            for arm in ARMS:
                new=read(PRIVATE/'fits'/f"fold{fold['fold']}_{arm}_{seed}.json");old=read(v3.PRIVATE/'fits'/f"fold{fold['fold']}_{arm}_{seed}.json")
                a=np.array([h['calibration_pauc'] for h in new['history'][:60]]);b=np.array([h['calibration_pauc'] for h in old['history'][:60]])
                worst=max(worst,float(np.abs(a-b).max()));checked+=1
                if new['caps']['60']['selected_epoch']!=old['caps']['60']['selected_epoch']:mismatched.append(f"fold{fold['fold']}_{arm}_{seed}")
    return {'fits_compared':checked,'max_abs_calibration_pauc_difference_epochs1_60':worst,'cap60_selected_epoch_mismatches':mismatched}


def evaluate():
    p=protocol();assert not (OUT/'summary.json').exists();fits=read(OUT/'fits_locked.json')
    assert fits['neural_fits']==50
    for rel,h in fits['files_sha256'].items():assert sha(ROOT/rel)==h,rel
    held={};choices={};convergence={}
    for fold in p['folds']:
        fid=fold['fold'];prior=read(v2.PRIVATE/'fits'/f'fold{fid}_prior.json')
        flags,choice=v2.attach_thresholds(v2.ensemble([prior['calibration']]),v2.ensemble([prior['evaluation']]));held.setdefault('prior',[]).extend(flags);choices[f'{fid}__prior']=choice
        for arm in ARMS:
            rr=[read(PRIVATE/'fits'/f'fold{fid}_{arm}_{seed}.json') for seed in p['seeds']]
            for r in rr:
                assert r['protocol_sha256']==sha(OUT/'locked_protocol.json')
                c=convergence.setdefault(arm,{'selected':[],'stopped':[],'patience_stops':0,'cap_hits':0})
                c['selected'].append(r['caps']['conv']['selected_epoch']);c['stopped'].append(r['stopped_epoch'])
                c['patience_stops']+=int(r['stopped_by_patience']);c['cap_hits']+=int(not r['stopped_by_patience'])
            for key,name in (('conv','Conv'),('60','60')):
                label=f'{arm}{name}'
                flags,choice=v2.attach_thresholds(v2.ensemble([r['caps'][key]['calibration'] for r in rr]),v2.ensemble([r['caps'][key]['evaluation'] for r in rr]))
                held.setdefault(label,[]).extend(flags);choices[f'{fid}__{label}']=choice
    keys={(r['row_id'],r['slot']):(r['phone'],r['y']) for r in held['mixedConv']};assert len(keys)==10013
    for arm,rr in held.items():assert {(r['row_id'],r['slot']):(r['phone'],r['y']) for r in rr}==keys,arm
    allmetrics={arm:v2.metrics(rr) for arm,rr in held.items()}
    rng=np.random.default_rng(p['statistics']['bootstrap_seed']);draws=rng.integers(0,10,(p['statistics']['draws'],10))
    def contrast(a,b):
        out={}
        for metric in ('pauc','conditional_fpr','recall'):
            d=np.array([r[metric] for r in allmetrics[a]['people']])-np.array([r[metric] for r in allmetrics[b]['people']]);bd=d[draws].mean(1)
            out[metric]={'delta':float(d.mean()),'ci95':np.quantile(bd,[.025,.975]).tolist(),'positive_people':int((d>0).sum()),'bootstrap_p':bootstrap_p(bd)}
        return out
    family={'E1_mixedConv_minus_uniformConv':('mixedConv','uniformConv'),'E3a_mixedConv_minus_mixed60':('mixedConv','mixed60'),
            'E3b_uniformConv_minus_uniform60':('uniformConv','uniform60')}
    contrasts={k:contrast(a,b) for k,(a,b) in family.items()}
    adjusted=holm([contrasts[k]['pauc']['bootstrap_p'] for k in family])
    for k,q in zip(family,adjusted):contrasts[k]['pauc']['holm_p']=q
    contrasts['secondary_mixedConv_minus_prior']=contrast('mixedConv','prior')
    contrasts['secondary_uniformConv_minus_prior']=contrast('uniformConv','prior')
    e1=contrasts['E1_mixedConv_minus_uniformConv']['pauc']
    baseline='mixed' if e1['delta']>0 and e1['holm_p']<.05 else 'uniform'
    conv={arm:{'median_selected_epoch':float(np.median(c['selected'])),'median_stopped_epoch':float(np.median(c['stopped'])),
               'patience_stops':c['patience_stops'],'cap200_hits':c['cap_hits'],'selected':c['selected'],'stopped':c['stopped']} for arm,c in convergence.items()}
    save(PRIVATE/'held_predictions.json',held);save(OUT/'thresholds.json',choices)
    save(OUT/'summary.json',{'protocol_sha256':sha(OUT/'locked_protocol.json'),'metrics':allmetrics,'contrasts':contrasts,
         'convergence':conv,'gate_B_baseline_by_locked_rule':baseline,'reproduction_vs_v3':reproduction(p),
         'known_held_positions':10013,'unknown_positions':453,'neural_fits':50,'fresh_validation_people':0,'new_labels':0,
         'limitations':['Retrospective10speaker development reused from v2/v3; no fresh confirmation',
                        'Calibration speakers select epochs, stopping and thresholds',
                        'Fixed-model speaker bootstrap excludes training/calibration uncertainty',
                        'Empirical ActCost thresholds, not CRC']})
    print({arm:round(m['means']['pauc'],6) for arm,m in allmetrics.items()},{'baseline':baseline},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['lock','fit','evaluate']);args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);globals()[args.phase]()
