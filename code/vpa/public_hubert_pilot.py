"""Locked external HuBERT pilot with oracle phone boundaries; not full S/D/I MDD."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score

from .external_hubert import (MODEL_ID, MODEL_REVISION, PHONE_TO_ID, PHONEMES,
                              conv_output_length, frame_center_samples, load_model)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def select_pilot(rows: list[dict], n_per_speaker: int = 12, seed: int = 1729) -> list[dict]:
    if len({(r['speaker_id'], r['utterance_id']) for r in rows}) != len(rows):
        raise ValueError('duplicate utterance identity')
    if n_per_speaker < 1:
        raise ValueError('positive pilot size required')
    selected = []
    rng = np.random.default_rng(seed)
    for l1 in sorted({r['l1'] for r in rows}):
        speaker = min(r['speaker_id'] for r in rows if r['l1'] == l1)
        group = sorted((r for r in rows if r['speaker_id'] == speaker), key=lambda r:r['utterance_id'])
        if len(group) < n_per_speaker:
            raise ValueError('insufficient speaker utterances')
        selected.extend(group[i] for i in sorted(rng.choice(len(group), n_per_speaker, replace=False).tolist()))
    return selected


def exclusion(event: dict) -> str | None:
    if event['kind'] not in ('acceptable', 'substitution'):
        return event['kind']
    if event.get('observed_deviation_mark') or '*' in event.get('raw_label', ''):
        return 'deviation_mark'
    if event.get('observed_unresolved'):
        return 'unresolved_observed'
    if event.get('canonical_phone') not in PHONE_TO_ID or event.get('observed_phone') not in PHONE_TO_ID:
        return 'out_of_inventory'
    if event['end_seconds'] <= event['start_seconds']:
        return 'zero_duration'
    return None


def pool_interval(probabilities: np.ndarray, start: float, end: float, coordinate: str) -> np.ndarray | None:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(PHONEMES):
        raise ValueError('expected frames by40 posterior array')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('invalid frame posteriors')
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError('invalid interval')
    if coordinate == 'author_label_grid':
        times = np.arange(len(probabilities)) * .020
    elif coordinate == 'geometric_center':
        times = frame_center_samples(len(probabilities)) / 16000
    else:
        raise ValueError('unknown coordinate')
    mask = (times >= start) & (times < end)
    return probabilities[mask].mean(0) if mask.any() else None


def waveform(path: Path) -> np.ndarray:
    sr, pcm = wavfile.read(path)
    if pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError('pilot expects release mono PCM16 WAV')
    audio = pcm.astype(np.float32) / 32768.
    if sr != 16000:
        divisor = math.gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sr) // divisor).astype(np.float32)
    if not np.isfinite(audio).all() or len(audio) < 400:
        raise ValueError('invalid decoded audio')
    return np.ascontiguousarray(audio)


def binary_counts(rows: list[dict], threshold: float | None = None) -> dict:
    y = np.asarray([r['y_true'] for r in rows], dtype=int)
    flagged = (np.asarray([r['priority'] for r in rows]) > threshold if threshold is not None
               else np.asarray([r['top1_phone'] != r['canonical_phone'] for r in rows]))
    tp, fp = int(np.sum(flagged & (y == 1))), int(np.sum(flagged & (y == 0)))
    positives, negatives, n_flags = int(y.sum()), int((y == 0).sum()), int(flagged.sum())
    ratio = lambda a,b: a/b if b else None
    return {'n':len(rows), 'n_substitution':positives, 'n_acceptable':negatives,
            'flags':n_flags, 'tp':tp, 'fp':fp, 'fn':positives-tp, 'tn':negatives-fp,
            'flag_coverage':ratio(n_flags,len(rows)), 'false_positive_rate_all':ratio(fp,negatives),
            'full_substitution_recall':ratio(tp,positives), 'precision':ratio(tp,n_flags),
            'false_fraction_among_flags':ratio(fp,n_flags), 'f1':ratio(2*tp,2*tp+fp+positives-tp),
            'nonflagged_action':'abstain for threshold policy; no correctness claim'}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('no eligible pilot events')
    y = np.asarray([r['y_true'] for r in rows])
    priority = np.asarray([r['priority'] for r in rows])
    return {'n_events':len(rows), 'n_speakers':len({r['speaker_id'] for r in rows}),
            'substitution_prevalence':float(y.mean()),
            'priority_auroc':float(roc_auc_score(y,priority)) if len(set(y)) == 2 else None,
            'priority_average_precision':float(average_precision_score(y,priority)) if y.sum() else None,
            'observed_phone_top1_accuracy':float(np.mean([r['top1_phone']==r['observed_phone'] for r in rows])),
            'top1_disagrees_with_canonical':binary_counts(rows),
            'predeclared_warning_curve':[{'threshold':float(t),**binary_counts(rows,float(t))} for t in np.linspace(0,1,21)],
            'by_speaker':{s:binary_counts([r for r in rows if r['speaker_id']==s]) for s in sorted({r['speaker_id'] for r in rows})},
            'by_l1':{s:binary_counts([r for r in rows if r['l1']==s]) for s in sorted({r['l1'] for r in rows})}}


def lock(args) -> None:
    if args.protocol.exists():
        raise ValueError('protocol already exists; never overwrite after seeing performance')
    rows = read_rows(args.manifest)
    selected = select_pilot(rows)
    if len(selected) != 72 or len({r['speaker_id'] for r in selected}) != 6:
        raise ValueError('expected 72 utterances from six L1 representatives')
    protocol = {'locked_at_utc':datetime.now(timezone.utc).isoformat(),
                'scope':'Development oracle-boundary acceptable versus substitution pilot; not full S/D/I or end-to-end MDD; no target training or calibration',
                'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'model_provenance_sha256':sha256(args.assets/'provenance.json'),
                'manifest_sha256':sha256(args.manifest),'implementation_sha256':sha256(Path(__file__)),
                'loader_sha256':sha256(Path(__file__).with_name('external_hubert.py')),
                'selection':'one alphabetically first speaker per L1; sorted L1 order; numpy default_rng1729 choose12 sorted utterances without replacement for each speaker',
                'seed':1729,'n_per_speaker':12,
                'selected':[{'speaker_id':r['speaker_id'],'l1':r['l1'],'utterance_id':r['utterance_id'],'wav_sha256':r['sha256']['wav']} for r in selected],
                'preprocessing':'mono PCM16 /32768; scipy resample_poly to16k; no gain or mean/variance normalization; entire utterance, batch1, CPU4threads',
                'event_scope':'release manual acceptable/substitution only; exclude silence/insertion/deletion/unparsed/unscorable, all*, unresolved, inventory mismatch including AX, zero duration, bounds beyond audio, intervals with no assigned frame; report exclusions',
                'primary_pooling':'mean frame softmax within[start,end); author label index grid t=.020*j from100Hz tokens[::2]',
                'secondary_pooling':'same operation on convolution geometric centers t=(199.5+320*j)/16000; report without selecting better performance',
                'score':'priority=1-meanP(canonical); not calibrated error probability',
                'metrics':['observed_phone_top1_accuracy','top1 != canonical binary counts','AUROC','average precision','fixed21 threshold warning curve','per speaker andL1'],
                'uncertainty':'2000 development-speaker bootstrap draws seed1729; descriptive only, six speakers inadequate for broad population claims',
                'model_selection_on_pilot':False,'fine_tuning':False,'crc':False,
                'future_exposure':'selected six speakers are development; other18 L2 speakers receive no model inference in this run; future hold-L1 study must disclose pilot exposure',
                'limits':['gold manual boundaries supplied; no localization assessment','deletions and insertions excluded and counted; not total-error recall','no across-paper F1 ranking','native aligned-label training is not independent human phonetic gold','pilot does not establish novelty or fulfill whole headroom gate']}
    save(args.protocol,protocol)
    print(json.dumps({'locked_protocol_sha256':sha256(args.protocol),'selected_speakers':sorted({r['speaker_id'] for r in selected}),'n_utterances':len(selected)}),flush=True)


def run(args) -> None:
    import torch
    from .external_hubert import verify_assets
    if (args.output/'summary.json').exists():
        raise ValueError('pilot performance exists; inspect stored predictions instead of rerunning')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    if (sha256(args.manifest)!=protocol['manifest_sha256'] or sha256(Path(__file__))!=protocol['implementation_sha256']
            or sha256(args.assets/'provenance.json')!=protocol['model_provenance_sha256']
            or sha256(Path(__file__).with_name('external_hubert.py'))!=protocol['loader_sha256']):
        raise ValueError('locked source changed')
    verify_assets(args.assets)
    source=read_rows(args.manifest)
    lookup={(r['speaker_id'],r['utterance_id']):r for r in source}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    model=load_model(args.assets)
    args.private.mkdir(parents=True,exist_ok=True)
    all_events={c:[] for c in ('author_label_grid','geometric_center')}
    excluded={c:Counter() for c in all_events}
    elapsed=0.;duration=0.;n_inferred=0
    for k,item in enumerate(protocol['selected']):
        row=lookup[(item['speaker_id'],item['utterance_id'])]
        path=args.corpus/row['paths_relative_to_corpus']['wav']
        if sha256(path)!=item['wav_sha256']:
            raise ValueError('pilot source WAV changed')
        audio=waveform(path);duration+=len(audio)/16000
        cache=args.private/f"{row['speaker_id']}_{row['utterance_id']}.npz"
        if cache.exists():
            z=np.load(cache,allow_pickle=False)
            if z['protocol_sha256'].item()!=sha256(args.protocol):
                raise ValueError('cache protocol mismatch')
            probability=z['probabilities']
        else:
            start=time.perf_counter()
            with torch.inference_mode():
                logits=model(torch.from_numpy(audio)[None,:])
                probability=torch.softmax(logits,dim=-1)[0].numpy()
            elapsed+=time.perf_counter()-start;n_inferred+=1
            np.savez_compressed(cache,probabilities=probability,protocol_sha256=sha256(args.protocol))
        if probability.shape!=(conv_output_length(len(audio)),len(PHONEMES)):
            raise ValueError('unexpected frame count')
        for event in row['phone_intervals']:
            reason=exclusion(event)
            if event['end_seconds']>len(audio)/16000+1/16000:
                reason='boundary_beyond_wav'
            for coordinate in all_events:
                if reason:
                    excluded[coordinate][reason]+=1;continue
                pooled=pool_interval(probability,event['start_seconds'],event['end_seconds'],coordinate)
                if pooled is None:
                    excluded[coordinate]['no_assigned_frame']+=1;continue
                top=PHONEMES[int(pooled.argmax())]
                all_events[coordinate].append({'speaker_id':row['speaker_id'],'l1':row['l1'],'utterance_id':row['utterance_id'],
                    'event_id':f"{row['speaker_id']}/{row['utterance_id']}:{event['interval_index']}",
                    'canonical_phone':event['canonical_phone'],'observed_phone':event['observed_phone'],
                    'y_true':int(event['kind']=='substitution'),'top1_phone':top,
                    'priority':float(1-pooled[PHONE_TO_ID[event['canonical_phone']]]),
                    'start_seconds':event['start_seconds'],'end_seconds':event['end_seconds']})
        if k==0 or (k+1)%12==0:
            print(json.dumps({'utterances_complete':k+1,'of':72,'inference_seconds':elapsed}),flush=True)
    summary={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'scope':protocol['scope'],
             'protocol_sha256':sha256(args.protocol),'implementation_sha256':sha256(Path(__file__)),
             'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'n_utterances':72,
             'runtime':{'torch':torch.__version__,'numpy':np.__version__,'scipy':__import__('scipy').__version__,
                        'sklearn':__import__('sklearn').__version__,'cpu_threads':torch.get_num_threads()},
             'audio_seconds':duration,'inference_seconds_this_invocation':elapsed,'new_inferred_utterances':n_inferred,
             'fine_tuning':False,'crc_applied':False,'unseen_l1_guarantee':False,
             'results':{c:{'excluded_intervals':dict(excluded[c]),**summarize(events)} for c,events in all_events.items()},
             'limits':protocol['limits'],'training_corpus':'publisher: LibriSpeech train-clean-100; no target training in this run'}
    primary=all_events['author_label_grid'];speakers=sorted({r['speaker_id'] for r in primary})
    draws=np.random.default_rng(1729).integers(0,len(speakers),size=(2000,len(speakers)))
    grouped={s:[r for r in primary if r['speaker_id']==s] for s in speakers}
    boot=[]
    for ix in draws:
        chosen=[r for i in ix for r in grouped[speakers[i]]]
        yy=np.asarray([r['y_true'] for r in chosen]);pp=np.asarray([r['priority'] for r in chosen])
        if len(set(yy))==2:boot.append(roc_auc_score(yy,pp))
    summary['primary_auroc_speaker_bootstrap']={'n_draws':2000,'n_defined':len(boot),'seed':1729,
       'ci95':np.quantile(boot,[.025,.975]).tolist() if boot else None,'scope':'descriptive six development speakers; excludes model and boundary uncertainty'}
    for coordinate,events in all_events.items():
        save(args.private/f'{coordinate}_events.json',events)
    save(args.output/'summary.json',summary)
    print(json.dumps({'complete':True,'primary':{k:v for k,v in summary['results']['author_label_grid'].items() if k not in ('predeclared_warning_curve','by_speaker','by_l1')}}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['lock','run'])
    for name in ('manifest','corpus','assets','protocol','private','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    lock(args) if args.phase=='lock' else run(args)


if __name__=='__main__':
    main()
