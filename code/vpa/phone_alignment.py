"""Maximum-posterior monotone phone alignment without supplied boundaries.

This diagnostic aligns a known phone sequence to frame-classifier posteriors.
It is not a CTC decoder, a likelihood/prior-corrected GOP implementation, or a
trained HMM. Every reference phone consumes at least one frame; optional SIL
states may consume zero or more frames. Gold labels and times are not inputs.
"""
from __future__ import annotations

import numpy as np


PROBABILITY_FLOOR = 1e-12


def posterior_align(probabilities: np.ndarray, phone_ids: list[int] | tuple[int, ...],
                    silence_id: int = 0) -> dict:
    """Return the legal path maximizing the sum of frame log posteriors.

    State order is SIL, phone0, SIL, phone1, ..., SIL. Allowed transitions
    are self, advance one, or advance two between adjacent phone states to
    skip the intervening SIL. No phone can be skipped. Leading/trailing SIL
    are optional. Probabilities are floored at1e-12 for numerical log safety.

    Equal scores prefer self, then advance one, then skip SIL. Equal terminal
    scores prefer the final phone over trailing SIL. Returned frame ranges are
    half-open [start,end), indexed by the original canonical phone positions.
    This function receives neither annotation boundaries nor observed phones.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not p.shape[0] or p.shape[1] < 2:
        raise ValueError('nonempty frames by classes probabilities required')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('probabilities must be finite and within [0,1]')
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError('frame probabilities must sum to one')
    if isinstance(silence_id, (bool, np.bool_)) or not isinstance(silence_id, (int, np.integer)):
        raise ValueError('silence ID must be an integer')
    if not 0 <= silence_id < p.shape[1]:
        raise ValueError('silence ID outside probability vocabulary')
    ids = list(phone_ids)
    if not ids:
        raise ValueError('at least one canonical phone required')
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or not 0 <= i < p.shape[1] or i == silence_id for i in ids):
        raise ValueError('canonical phone IDs must be valid non-silence integers')
    if p.shape[0] < len(ids):
        raise ValueError('fewer frames than mandatory canonical phones')

    n_states = 2 * len(ids) + 1
    state_ids = np.full(n_states, int(silence_id), dtype=np.int64)
    state_ids[1::2] = ids
    emissions = np.log(np.maximum(p[:, state_ids], PROBABILITY_FLOOR))
    previous = np.full(n_states, -np.inf)
    previous[:2] = emissions[0, :2]
    predecessors = np.zeros((p.shape[0], n_states), dtype=np.int32)
    states = np.arange(n_states)
    allow_skip = (states >= 3) & (states % 2 == 1)
    for frame in range(1, p.shape[0]):
        advance = np.r_[-np.inf, previous[:-1]]
        skip = np.r_[-np.inf, -np.inf, previous[:-2]]
        skip[~allow_skip] = -np.inf
        choices = np.stack((previous, advance, skip))
        transition = np.argmax(choices, axis=0)
        predecessors[frame] = states - transition
        previous = choices[transition, states] + emissions[frame]

    # Ordered terminal candidates explicitly define the tie convention.
    terminals = [n_states - 2, n_states - 1]
    terminal = terminals[int(np.argmax(previous[terminals]))]
    if not np.isfinite(previous[terminal]):
        raise ValueError('no valid complete alignment')
    score = float(previous[terminal])
    state_path = np.empty(p.shape[0], dtype=np.int32)
    state_path[-1] = terminal
    for frame in range(p.shape[0] - 1, 0, -1):
        state_path[frame - 1] = predecessors[frame, state_path[frame]]
    ranges = []
    for index in range(len(ids)):
        frames = np.flatnonzero(state_path == 2 * index + 1)
        if not len(frames) or frames[-1] - frames[0] + 1 != len(frames):
            raise AssertionError('mandatory phone missing or assigned disjoint frames')
        ranges.append([int(frames[0]), int(frames[-1] + 1)])
    return {'state_path': state_path, 'state_phone_ids': state_ids,
            'phone_frame_ranges': ranges, 'best_score': score,
            'n_silence_frames': int(np.sum(state_path % 2 == 0)),
            'probability_floor': PROBABILITY_FLOOR}
