"""Matched SSL-MD smoke: development-only, no fitted model or performance search."""
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '06_experiment_results/sslmd_smoke_v1'
PRIVATE = ROOT / 'data/derived/sslmd_smoke_v1'
REFS = ROOT / 'data/derived/epadb_reference_prep_v1/input_references.json'
LABELS = ROOT / 'data/derived/epadb_development_v1/certified_labels.json'
PREDS = ROOT / 'data/derived/epadb_development_v1/hubert_predictions.json'
DEV = ROOT / 'data/derived/epadb_development_v1'
SPEC = ROOT / '05_methodology/sslmd_matched_baseline_spec_2026-11-07.md'
RESOURCE_AMENDMENT = ROOT / '05_methodology/sslmd_smoke_resource_amendment_2026-11-07.md'
MODEL = ROOT / 'data/resources/external_hubert_coml/9251d0fdc406d973ce1dbb7738ac6f7b124fbc65'
PARQUET = ROOT / 'data/raw/epadb_koellabs/52404544712cdd5f45a020a1a86dfe27d5c9cbd8/data/train-00000-of-00001.parquet'
GIB = 1024 ** 3


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def resources():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong) for k in ('total_phys', 'avail_phys', 'total_page',
                                             'avail_page', 'total_virtual', 'avail_virtual', 'extended')]
    m = Memory(); m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('GlobalMemoryStatusEx failed')
    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (k, ctypes.c_size_t) for k in ('peak_working', 'working', 'peak_paged_pool',
                                          'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool',
                                          'pagefile', 'peak_pagefile')]
    p = ProcessMemory(); p.cb = ctypes.sizeof(p)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(p), ctypes.sizeof(p))
    if not ok:
        raise OSError('GetProcessMemoryInfo failed')
    return {'utc': datetime.now(timezone.utc).isoformat(), 'available_ram_bytes': m.avail_phys,
            'total_ram_bytes': m.total_phys, 'free_disk_bytes': shutil.disk_usage(ROOT).free,
            'working_set_bytes': p.working, 'peak_working_set_bytes': p.peak_working}


def frame_map(phones, labels, ranges, frames):
    """Disjoint half-open phone ranges; silence/unknown never become negatives."""
    if len(phones) != len(labels) or len(phones) != len(ranges) or frames <= 0:
        raise ValueError('position count/frame count mismatch')
    owner = np.full(frames, -1, dtype=np.int32)
    target = np.full(frames, -1, dtype=np.int8)
    canonical = np.full(frames, -1, dtype=np.int16)
    last_end = 0
    for j, (phone, label, span) in enumerate(zip(phones, labels, ranges, strict=True)):
        if phone not in range(39) or label not in (-1, 0, 1) or len(span) != 2:
            raise ValueError('invalid phone/label/span')
        lo, hi = span
        if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int) or not isinstance(hi, int):
            raise ValueError('integer frame boundaries required')
        if not 0 <= last_end <= lo <= hi <= frames:
            raise ValueError('out-of-order, overlapping or out-of-bounds ranges')
        owner[lo:hi] = j
        canonical[lo:hi] = phone
        target[lo:hi] = label
        last_end = hi
    valid = (owner >= 0) & (target >= 0)
    return {'owner': owner, 'phone': canonical, 'target': target, 'valid': valid}


def pool_logits(logits, phones, ranges):
    if logits.ndim != 2 or logits.shape[1] != 39 or len(phones) != len(ranges):
        raise ValueError('logit dimensions mismatch')
    result = []
    for phone, (lo, hi) in zip(phones, ranges, strict=True):
        if not 0 <= phone < 39 or not 0 <= lo <= hi <= len(logits):
            raise ValueError('invalid pooling position')
        result.append(None if hi == lo else float(logits[lo:hi, phone].mean()))
    return result


def make_head(mixed, dimension=768):
    import torch
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(dimension, 39)
            self.mix = torch.nn.Parameter(torch.zeros(12)) if mixed else None

        def forward(self, layers):
            if layers.ndim != 3 or layers.shape[0] != 12 or layers.shape[2] != dimension:
                raise ValueError('12 by frames by dimension required')
            features = layers[-1] if self.mix is None else torch.einsum('l,ltd->td', self.mix.softmax(0), layers)
            return self.linear(features)
    return Head()


def masked_loss(logits, mapping, weights=None):
    import torch
    valid = torch.as_tensor(mapping['valid'], dtype=torch.bool)
    indices = valid.nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        return None
    phones = torch.as_tensor(mapping['phone'][mapping['valid']], dtype=torch.long)
    targets = torch.as_tensor(mapping['target'][mapping['valid']], dtype=logits.dtype)
    selected = logits[indices, phones]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(selected, targets, reduction='none')
    if weights is None:
        return loss.mean()
    w = torch.as_tensor(weights, dtype=loss.dtype)
    if w.shape != loss.shape or not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
        raise ValueError('finite positive eligible-frame weights required')
    return (loss * w).sum() / w.sum()


def gradient_checks(encoder):
    import torch
    mapping = frame_map([0, 2], [0, 1], [[1, 3], [4, 6]], 8)
    torch.manual_seed(1729)
    layers = torch.randn(12, 8, 8)
    torch.manual_seed(1729); last = make_head(False, 8)
    torch.manual_seed(1729); mixed = make_head(True, 8)
    assert torch.equal(last.linear.weight, mixed.linear.weight)
    assert torch.equal(last.linear.bias, mixed.linear.bias)
    logits = mixed(layers); logits.retain_grad()
    loss = masked_loss(logits, mapping); loss.backward()
    expected = torch.zeros_like(logits, dtype=torch.bool)
    for frame in np.flatnonzero(mapping['valid']):
        expected[frame, int(mapping['phone'][frame])] = True
    assert bool((logits.grad[~expected] == 0).all())
    assert bool((logits.grad[expected] != 0).all())
    assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in mixed.parameters())
    assert float(mixed.mix.grad.abs().sum()) > 0
    assert all(not t.requires_grad and t.grad is None for t in encoder.parameters())
    reference = torch.nn.functional.binary_cross_entropy_with_logits
    assert reference(torch.tensor([2.]), torch.tensor([1.])) < reference(torch.tensor([-2.]), torch.tensor([1.]))
    # The finite synthetic test never takes an optimizer step.
    return {'all_passed': True, 'paired_head_initialization_exact': True,
            'mix_and_head_gradients_finite': True, 'mix_gradient_nonzero': True,
            'padding_silence_and_nonselected_output_gradients_zero': True,
            'error_label_positive_logit_polarity_verified': True,
            'actual_frozen_encoder_gradients_absent': True, 'optimizer_steps': 0,
            'synthetic_frames': 8, 'eligible_synthetic_frames': 4}


def protocol():
    p = read(OUT / 'locked_smoke_protocol.json')
    for rel, h in p['inputs_sha256'].items():
        assert sha(ROOT / rel) == h, rel
    return p


def lock():
    assert not (OUT / 'locked_smoke_protocol.json').exists()
    refs = read(REFS)
    assert len(refs) == 634 and len({r['speaker'] for r in refs}) == 10
    selected = sorted(refs, key=lambda r: hashlib.sha256(('sslmd-smoke-v1|' + r['row_id']).encode()).hexdigest())[:20]
    paths = [REFS, LABELS, PREDS, SPEC, RESOURCE_AMENDMENT, PARQUET, Path(__file__),
             ROOT / 'code/tests/test_sslmd_smoke.py', OUT / 'software_tests.json',
             ROOT / 'code/vpa/external_hubert.py', ROOT / 'code/vpa/phone_alignment.py',
             ROOT / '06_experiment_results/epadb_matched_head_v1/locked_protocol.json',
             ROOT / '06_experiment_results/epadb_label_version_forensics_v1/delivery_manifest_sha256.json']
    provenance = read(MODEL / 'provenance.json')
    paths += [MODEL / 'provenance.json'] + [MODEL / name for name in provenance['files']]
    paths += [DEV / 'cache/hubert' / (r['row_id'].replace(':', '_') + '.npz') for r in refs]
    save(OUT / 'locked_smoke_protocol.json', {
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Retrospective development-only implementation/resource smoke; no performance claims or fits',
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'all_eligible_row_ids': [r['row_id'] for r in refs],
        'selected_rows': [{k: r[k] for k in ('row_id', 'speaker', 'utterance')} for r in selected],
        'selection': '20 smallest SHA256(sslmd-smoke-v1|row_id) across the predeclared634 rows',
        'minimum_available_ram_bytes': 2 * GIB, 'preload_reserve_bytes': int(1.5 * GIB),
        'resource_policy': 'Pre-inference smoke-only amendment: batch1, keep2GiB available; full fitting still requires original4GiB reserve.',
        'minimum_disk_after_full_cache_bytes': 10 * GIB,
        'preprocessing': 'PCM16/32768, mono16kHz, no padding in physical batch1',
        'layers': '12transformer outputs captured from unchanged native model forward; no CNN layer',
        'threads': 4, 'interop': 1, 'seed': 1729,
        'probability_comparison_max_absolute_tolerance': 5e-6,
        'new_fits': 0, 'no_threshold_or_performance_selection': True})
    print('Locked smoke20 IDs and source/input hashes.', flush=True)


def mapping_audit():
    p = protocol()
    assert not (OUT / 'mapping_audit.json').exists()
    from .external_hubert import PHONE_TO_ID
    from .phone_alignment import posterior_align
    refs = read(REFS)
    gold = {r['row_id']: r for r in read(LABELS) if r['arm'] == 'author_first__hubert_mean'}
    predictions = {r['row_id']: r for r in read(PREDS)}
    selected = {r['row_id'] for r in p['selected_rows']}
    ledger, totals = [], Counter()
    for r in refs:
        key = r['row_id']
        phones = r['arms']['hubert__author_first']['phones']
        codes = [PHONE_TO_ID[x] for x in phones]
        old = predictions[key]
        assert old['arms']['author_first__hubert_mean']['reference'] == phones
        ranges = old['author_first_details']['ranges']
        with np.load(DEV / 'cache/hubert' / (key.replace(':', '_') + '.npz'), allow_pickle=False) as z:
            prob = z['probabilities']
        assert posterior_align(prob, codes)['phone_frame_ranges'] == ranges
        labels = gold[key]['labels']
        m = frame_map([x - 1 for x in codes], labels, ranges, len(prob))
        one = {'row_id': key, 'speaker': r['speaker'], 'smoke_selected': key in selected,
               'positions': len(phones), 'known_positions': sum(y >= 0 for y in labels),
               'unknown_positions': sum(y < 0 for y in labels),
               'zero_frame_positions': sum(hi == lo for lo, hi in ranges),
               'frames': len(prob), 'silence_frames': int((m['owner'] < 0).sum()),
               'eligible_frames': int(m['valid'].sum()),
               'unknown_phone_frames': int(((m['owner'] >= 0) & (m['target'] < 0)).sum()),
               'error_frames': int((m['target'] == 1).sum()),
               'ranges': ranges, 'phones': [x - 1 for x in codes], 'labels': labels}
        totals.update({k: one[k] for k in ('positions', 'known_positions', 'unknown_positions',
                                          'zero_frame_positions', 'frames', 'silence_frames',
                                          'eligible_frames', 'unknown_phone_frames', 'error_frames')})
        ledger.append(one)
    assert len(ledger) == 634 and sum(r['smoke_selected'] for r in ledger) == 20
    assert totals['frames'] == totals['silence_frames'] + totals['eligible_frames'] + totals['unknown_phone_frames']
    save(PRIVATE / 'frame_mapping.json', ledger)
    save(OUT / 'mapping_audit.json', {'all_passed': True, 'rows': 634,
         'selected_rows': 20, 'people': 10, 'totals': dict(totals),
         'mapping_sha256': sha(PRIVATE / 'frame_mapping.json'),
         'exact_float32_full_cache_payload_bytes': totals['frames'] * 12 * 768 * 4,
         'new_encoder_runs': 0, 'input_gold_used_to_align': False,
         'target_use': 'certified labels only mask existing automatic phone-frame ranges; no manual times or observed phones'})
    print('All634 cached alignments replayed; mapping invariant checks passed.', flush=True)


def extract():
    p = protocol()
    assert not (OUT / 'extraction_complete.json').exists()
    audit = read(OUT / 'mapping_audit.json')
    assert audit['all_passed'] and sha(PRIVATE / 'frame_mapping.json') == audit['mapping_sha256']
    initial = resources(); needed = audit['exact_float32_full_cache_payload_bytes']
    allowed = (initial['available_ram_bytes'] >= p['minimum_available_ram_bytes'] + p['preload_reserve_bytes']
               and initial['free_disk_bytes'] - needed >= p['minimum_disk_after_full_cache_bytes'])
    gate = {'resources_before_torch': initial, 'full_cache_payload_estimate_bytes': needed,
            'memory_gate_passed': initial['available_ram_bytes'] >= p['minimum_available_ram_bytes'] + p['preload_reserve_bytes'],
            'disk_gate_passed': initial['free_disk_bytes'] - needed >= p['minimum_disk_after_full_cache_bytes'],
            'can_load_model': allowed}
    save(OUT / ('preflight_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'), gate)
    if not allowed:
        print(json.dumps({'resource_gate_pending': gate}), flush=True)
        return
    import torch
    import pyarrow.parquet as pq
    from scipy.io import wavfile
    from .external_hubert import load_model, conv_output_length
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter(); model = load_model(MODEL)
    assert all(not x.requires_grad and x.grad is None for x in model.parameters())
    transformer = model.model.wav2vec2.encoder.transformer
    assert len(transformer.layers) == 12 and transformer.layer_norm_first
    rows = read(REFS); roles = read(ROOT / 'data/derived/epadb_admission_v1/speaker_roles.json')
    by_global_index = {int(r['row_id'].split(':')[1]): r for r in rows}
    assert set(by_global_index) == set(roles['mapping_train_row_indices'])
    table = pq.read_table(PARQUET, columns=['audio', 'speaker_code'],
                          filters=[('speaker_code', 'in', sorted({r['speaker'] for r in rows}))]).to_pylist()
    assert len(table) == 634
    audio = {f'train:{i}': a for i, a in zip(roles['mapping_train_row_indices'], table, strict=True)}
    maps = {r['row_id']: r for r in read(PRIVATE / 'frame_mapping.json')}
    cache = PRIVATE / 'hidden'; cache.mkdir(parents=True, exist_ok=True)
    results = []
    for choice in p['selected_rows']:
        key = choice['row_id']; stem = key.replace(':', '_')
        receipt = cache / (stem + '.json'); target = cache / (stem + '.npy')
        if receipt.exists():
            old = read(receipt); assert sha(target) == old['sha256']; results.append(old); continue
        assert not target.exists(), 'unreceipted feature cache requires review'
        before = resources()
        if before['available_ram_bytes'] < p['minimum_available_ram_bytes']:
            save(OUT / ('runtime_resource_stop_' + stem + '.json'), {'resources': before, 'completed_rows': len(results)})
            print('Resource guard stopped before next utterance; caches retained.', flush=True)
            return
        a = audio[key]; assert a['speaker_code'] == choice['speaker']
        rate, pcm = wavfile.read(io.BytesIO(a['audio']['bytes']))
        assert rate == 16000 and pcm.dtype == np.int16 and pcm.ndim == 1
        wave = torch.from_numpy(np.ascontiguousarray(pcm.astype(np.float32) / 32768.))[None]
        captured, hooks = [], []
        def hook(_module, _input, output):
            captured.append(output[0].detach())
        for layer in transformer.layers:
            hooks.append(layer.register_forward_hook(hook))
        tick = time.perf_counter()
        try:
            with torch.inference_mode():
                logits = model(wave)
                assert len(captured) == 12
                assert all(tuple(t.shape) == (1, conv_output_length(len(pcm)), 768) for t in captured)
                assert torch.equal(model.aux(captured[-1]), logits)
                features = torch.stack([t[0] for t in captured]).numpy()
                prob = logits.softmax(-1)[0].numpy()
        finally:
            for h in hooks:
                h.remove()
        assert np.isfinite(features).all() and features.dtype == np.float32
        with np.load(DEV / 'cache/hubert' / (stem + '.npz'), allow_pickle=False) as z:
            old_prob = z['probabilities']
        assert prob.shape == old_prob.shape
        difference = float(np.abs(prob - old_prob).max())
        assert difference <= p['probability_comparison_max_absolute_tolerance']
        assert features.shape[1] == maps[key]['frames']
        np.save(target, features, allow_pickle=False)
        item = {**choice, 'shape': list(features.shape), 'dtype': 'float32',
                'sha256': sha(target), 'cache_bytes': target.stat().st_size,
                'encoder_seconds': time.perf_counter() - tick,
                'max_posterior_difference_from_frozen_cache': difference,
                'native_last_layer_classifier_exact': True,
                'resources_after': resources()}
        save(receipt, item); results.append(item)
        del captured, features, logits, prob, wave
        print(json.dumps({'completed': len(results), 'of': 20, 'row_id': key,
                          'seconds': round(time.perf_counter() - started, 2)}), flush=True)
    gradient = gradient_checks(model)
    save(OUT / 'gradient_checks.json', gradient)
    save(OUT / 'extraction_complete.json', {'all_passed': True, 'rows': results,
        'unique_encoder_utterances': 20, 'new_model_fits': 0, 'seconds_in_this_invocation': time.perf_counter() - started,
        'total_encoder_seconds': sum(r['encoder_seconds'] for r in results),
        'actual_cache_bytes': sum(r['cache_bytes'] for r in results),
        'resources_at_end': resources(), 'all_encoder_parameters_frozen': True,
        'gradient_checks_sha256': sha(OUT / 'gradient_checks.json')})


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('phase', choices=['lock', 'mapping_audit', 'extract'])
    args = a.parse_args(); OUT.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    globals()[args.phase]()
