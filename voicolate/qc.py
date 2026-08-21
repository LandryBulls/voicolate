"""Quality control for isolated audio.

The v2 outputs destroyed real speech on most sessions and nobody knew, because
the only way to find out was to listen. These metrics make the same judgement
numerically, against the calibrated raw audio, so a bad session raises instead of
passing silently.

The key measurement is deliberately conservative about what counts as speech: a
frame only qualifies if the target's own mic leads every other mic *and* the frame
sits well above that track's own noise floor. Anything cut under those conditions
is speech the isolation should have kept.
"""

import numpy as np

from .calibrate import frame_levels, noise_floor, speech_level

# a track failing any of these is worth looking at by ear
WARN_SPEECH_CUT_12 = 0.03      # fraction of clear speech frames attenuated >12 dB
WARN_SPEECH_SILENCED_S = 0.5   # seconds of clear speech driven to exact zero
# Bleed rejection is capped by design: `residual_floor_db` is the most the
# suppressor may ever attenuate, so the useful question is whether it reaches its
# own floor, not whether it beats some absolute number.
WARN_BLEED_MARGIN_DB = 2.0


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def longest_silence(audio, rate):
    """Longest run of exactly-zero samples, in seconds, and the overall zero fraction.

    Reported but never warned on by itself. Once suppression is floored, most zeros
    come from int16 quantisation of a residual already 100 dB below speech, which is
    harmless. What matters is whether zeros land on top of speech, which
    `speech_silenced_seconds` answers directly.
    """
    z = (audio == 0)
    if not z.any():
        return 0.0, 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], z.view(np.int8), [0]))))
    runs = edges[1::2] - edges[0::2]
    return float(runs.max()) / rate, float(z.mean())


def speech_silenced_seconds(out, speech_frames, rate, hop):
    """Seconds of clear-speech frames whose output is exactly zero.

    The precise version of "did we destroy speech". A long silent run over a
    stretch where nobody was talking is quantisation; the same run over frames
    where the target clearly held the floor is the v2 failure.
    """
    n = min(len(speech_frames), len(out) // hop)
    if n == 0:
        return 0.0
    frames = np.asarray(out[:n * hop]).reshape(n, hop)
    silent = ~frames.any(axis=1)
    return float((silent & speech_frames[:n]).sum()) * hop / rate


def track_qc(raw, out, others_levels, rate, hop=512, residual_floor_db=-20.0,
             dominance_db=6.0, above_floor_db=12.0, bleed_dominance_db=-6.0):
    """Compare one isolated track against the calibrated raw track it came from."""
    e_raw = frame_levels(raw, hop)
    e_out = frame_levels(out, hop)
    m = min(len(e_raw), len(e_out), min(len(o) for o in others_levels))
    e_raw, e_out = e_raw[:m], e_out[:m]
    others = np.max(np.array([o[:m] for o in others_levels]), axis=0)

    dom_db = _db(e_raw) - _db(others)
    lvl_db = _db(e_raw)
    floor_db = _db(noise_floor(e_raw))
    speech_db = _db(speech_level(e_raw))

    gain_db = _db(e_out) - lvl_db
    # the isolation applies a session-wide output gain; take it out so what is
    # left is attenuation the mask actually chose to apply
    ref = (e_raw > np.percentile(e_raw, 90)) & (dom_db > 15)
    if ref.sum() >= 50:
        gain_db = gain_db - np.percentile(gain_db[ref], 90)

    speech = (dom_db > dominance_db) & (lvl_db > floor_db + above_floor_db)
    bleed = (dom_db < bleed_dominance_db) & (lvl_db > floor_db + 6)

    longest, zero_frac = longest_silence(out, rate)

    q = {
        'speech_frames': int(speech.sum()),
        'speech_seconds': round(float(speech.sum()) * hop / rate, 1),
        'noise_floor_db': round(float(floor_db), 1),
        'speech_level_db': round(float(speech_db), 1),
        # how far the peak sits above speech: the v2 gate's threshold was set from
        # the peak, so this number predicted how aggressive that session would be
        'peak_over_speech_db': round(float(_db(np.abs(raw).max()) - speech_db), 1),
        'zero_fraction': round(float(zero_frac), 4),
        'longest_silence_s': round(longest, 2),
        'speech_silenced_s': round(speech_silenced_seconds(out, speech, rate, hop), 2),
    }
    if speech.sum() >= 100:
        q.update({
            'speech_gain_median_db': round(float(np.median(gain_db[speech])), 2),
            'speech_cut_6_frac': round(float(np.mean(gain_db[speech] < -6)), 4),
            'speech_cut_12_frac': round(float(np.mean(gain_db[speech] < -12)), 4),
            'speech_cut_20_frac': round(float(np.mean(gain_db[speech] < -20)), 4),
        })
    if bleed.sum() >= 100:
        q['bleed_gain_median_db'] = round(float(np.median(gain_db[bleed])), 2)

    warnings = []
    if q.get('speech_cut_12_frac', 0) > WARN_SPEECH_CUT_12:
        warnings.append(
            f"{100 * q['speech_cut_12_frac']:.1f}% of clear speech frames cut >12 dB "
            f"(limit {100 * WARN_SPEECH_CUT_12:.0f}%)")
    if q['speech_silenced_s'] > WARN_SPEECH_SILENCED_S:
        warnings.append(
            f"{q['speech_silenced_s']:.1f} s of clear speech driven to digital silence")
    bleed_limit = residual_floor_db + WARN_BLEED_MARGIN_DB
    if 'bleed_gain_median_db' in q and q['bleed_gain_median_db'] > bleed_limit:
        warnings.append(
            f"bleed only attenuated {q['bleed_gain_median_db']:.1f} dB; the "
            f"{residual_floor_db:.0f} dB residual floor should reach {bleed_limit:.0f} dB")
    q['warnings'] = warnings
    return q


def session_qc(raw_tracks, out_tracks, rate, hop=512, names=None,
               residual_floor_db=-20.0):
    """QC every track in a session. `raw_tracks` must be the calibrated raw audio."""
    levels = [frame_levels(t, hop) for t in raw_tracks]
    names = names or [f'track{i}' for i in range(len(raw_tracks))]
    tracks = {}
    for i, (raw, out) in enumerate(zip(raw_tracks, out_tracks)):
        others = [levels[j] for j in range(len(levels)) if j != i]
        tracks[names[i]] = track_qc(raw, out, others, rate, hop=hop,
                                    residual_floor_db=residual_floor_db)
    return {
        'tracks': tracks,
        'n_warnings': sum(len(t['warnings']) for t in tracks.values()),
    }


def format_report(qc, title=''):
    """One-screen summary for a terminal."""
    lines = [f"QC {title}".rstrip(), '-' * 78,
             f"{'track':16s} {'spk_s':>7s} {'floor':>7s} {'cut>6':>7s} {'cut>12':>7s} "
             f"{'cut>20':>7s} {'bleed':>7s} {'zero%':>6s} {'spk_sil':>8s}"]
    for name, t in qc['tracks'].items():
        lines.append(
            f"{name:16s} {t.get('speech_seconds', 0):7.0f} {t['noise_floor_db']:7.1f} "
            f"{t.get('speech_cut_6_frac', float('nan')):7.3f} "
            f"{t.get('speech_cut_12_frac', float('nan')):7.3f} "
            f"{t.get('speech_cut_20_frac', float('nan')):7.3f} "
            f"{t.get('bleed_gain_median_db', float('nan')):7.1f} "
            f"{100 * t['zero_fraction']:6.2f} {t['speech_silenced_s']:8.2f}")
    for name, t in qc['tracks'].items():
        for w in t['warnings']:
            lines.append(f"  WARNING [{name}] {w}")
    if not qc['n_warnings']:
        lines.append("  all tracks within thresholds")
    return '\n'.join(lines)
