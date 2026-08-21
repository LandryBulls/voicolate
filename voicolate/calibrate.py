"""Per-track calibration for cross-microphone isolation.

Everything here works on frame-rate envelopes rather than samples, so a whole
session's statistics cost one cheap pass before any STFT work happens.

The reason this module exists: the cross-microphone dominance test compares one
mic's level against another's, so any gain difference between the two tracks is
read as a difference in who is speaking. The legacy path normalised each track to
a common overall RMS, which equalises tracks by *how much that person talked* --
a participant who says little gets their whole track boosted, and their bleed then
looks louder in everyone else's dominance test.
"""

import numpy as np

EPS = 1e-12


def frame_levels(audio, hop=512):
    """RMS level of `audio` in consecutive non-overlapping frames of `hop` samples."""
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0)
    # reshape is a view; einsum accumulates in float64 without a full-length temporary
    frames = np.asarray(audio[:n * hop]).reshape(n, hop)
    return np.sqrt(np.einsum('ij,ij->i', frames, frames, dtype=np.float64) / hop) + EPS


def noise_floor(levels, percentile=5.0):
    """This track's noise floor.

    A low percentile of the frame levels, which is immune to the transients that
    make a peak reference unusable: one mic bump moves a track's peak by 20 dB and
    leaves this number untouched.
    """
    return float(np.percentile(levels, percentile))


def speech_level(levels, percentile=95.0):
    """This track's speech level.

    Unlike overall RMS this does not depend on how much the person talked, which
    is what makes it usable as a calibration reference.
    """
    return float(np.percentile(levels, percentile))


def bleed_matrix(levels, dominance_db=10.0, min_frames=50):
    """Measure how much of each speaker lands in every other microphone.

    ``B[i, j]`` is, in dB, how far track ``i`` sits below track ``j`` during frames
    where ``j`` is the uncontested dominant speaker -- i.e. the bleed of speaker
    ``j`` into mic ``i``. For an array whose microphones are gain-matched and whose
    participants are seated symmetrically, ``B`` is roughly symmetric. A systematic
    row/column offset is a microphone gain difference rather than a talker
    difference, and is what `calibration_gains(method='bleed_symmetry')` removes.

    Returns ``(B, n_frames)``; entries with too little uncontested speech are NaN.
    """
    n = len(levels)
    m = min(len(l) for l in levels)
    db = 20 * np.log10(np.array([l[:m] for l in levels]))

    B = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)
    for j in range(n):
        others = np.delete(db, j, axis=0).max(axis=0)
        # j clearly leads everyone else, and is actually speaking rather than
        # merely being the loudest thing in a silent stretch
        solo = ((db[j] - others) > dominance_db) & (db[j] > np.percentile(db[j], 75))
        if solo.sum() < min_frames:
            continue
        for i in range(n):
            B[i, j] = 0.0 if i == j else float(np.median(db[i][solo] - db[j][solo]))
            counts[i, j] = int(solo.sum())
    return B, counts


def _symmetrising_gains(B):
    """Least-squares per-track gains (dB) that make the bleed matrix symmetric.

    Applying gain ``a_i`` to track ``i`` turns ``B[i, j]`` into ``B[i, j] + a_i - a_j``,
    so symmetry asks for ``a_i - a_j = (B[j, i] - B[i, j]) / 2`` over every pair with
    a measurement. Solved as a weighted graph Laplacian problem with the gains
    constrained to sum to zero.

    Returns ``(gains_db, constrained)``, where ``constrained[i]`` is False for a track
    sharing no measured pair with any other -- a mic quiet enough that it never wins an
    uncontested stretch, which the caller falls back on speech level for.
    """
    n = B.shape[0]
    L = np.zeros((n, n))
    rhs = np.zeros(n)
    constrained = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if not (np.isfinite(B[i, j]) and np.isfinite(B[j, i])):
                continue
            d = (B[j, i] - B[i, j]) / 2.0
            L[i, i] += 1; L[j, j] += 1
            L[i, j] -= 1; L[j, i] -= 1
            rhs[i] += d; rhs[j] -= d
            constrained[i] = constrained[j] = True
    # sum-to-zero constraint keeps the solution unique
    L += np.ones((n, n)) / n
    return np.linalg.lstsq(L, rhs, rcond=None)[0], constrained


def calibration_gains(levels, method='speech_level', target_level_db=-20.0,
                      speech_level_percentile=95.0):
    """Linear per-track gains that put the tracks on a common footing.

    Returns ``(gains, info)``. ``info`` records what was measured so the choice can
    be audited from the sidecar written next to the audio.
    """
    n = len(levels)
    speech_db = np.array([20 * np.log10(speech_level(l, speech_level_percentile))
                          for l in levels])
    info = {'method': method, 'speech_level_db': speech_db.round(2).tolist()}

    if method == 'none':
        gains_db = np.zeros(n)

    elif method == 'rms':
        # legacy behaviour, kept so old outputs can be reproduced
        rms_db = np.array([20 * np.log10(np.sqrt(np.mean(l ** 2)) + EPS) for l in levels])
        gains_db = target_level_db - rms_db
        info['rms_db'] = rms_db.round(2).tolist()

    elif method == 'speech_level':
        gains_db = target_level_db - speech_db

    elif method == 'bleed_symmetry':
        B, counts = bleed_matrix(levels)
        info['bleed_matrix_db'] = np.where(np.isfinite(B), B, None).tolist()
        info['bleed_frames'] = counts.tolist()
        a, constrained = _symmetrising_gains(B)
        if not constrained.any():
            # no track ever holds the floor uncontested; speech level is the safe fallback
            info['fallback'] = 'speech_level (no uncontested speech)'
            gains_db = target_level_db - speech_db
        else:
            # anchor so the median speech level of the solved tracks lands on the target
            offset = target_level_db - np.median((speech_db + a)[constrained])
            gains_db = a + offset
            # tracks the bleed matrix could not constrain keep their speech-level gain
            gains_db[~constrained] = (target_level_db - speech_db)[~constrained]
            info['symmetrising_db'] = np.round(a, 2).tolist()
            info['constrained'] = constrained.tolist()
            if not constrained.all():
                info['fallback'] = ('speech_level for tracks '
                                    f'{np.flatnonzero(~constrained).tolist()}')
    else:
        raise ValueError(f"unknown calibration method: {method!r}")

    info['gains_db'] = np.round(gains_db, 2).tolist()
    return 10 ** (gains_db / 20), info
