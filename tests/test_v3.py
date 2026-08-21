"""Unit tests for the v3 isolation path.

Fast, synthetic, and focused on the properties that broke in v2: level references
that move with a transient, unbounded attenuation, and calibration that depends on
how much someone talked.
"""

import numpy as np
import pytest

from voicolate.config import IsolationConfig
from voicolate.calibrate import (frame_levels, noise_floor, speech_level,
                                 calibration_gains, bleed_matrix)
from voicolate.isolate import (_sigmoid, max_excluding, dominance_mask,
                               speech_presence, isolate_v3, load_tracks)
from voicolate.qc import speech_silenced_seconds, track_qc

RATE = 44100
HOP = 512


def synth_array(n_mics=4, seconds=20, rate=RATE, bleed_db=-25.0, mic_gains=None,
                talk_fraction=None, seed=0):
    """Each speaker talks in a distinct stretch; every other mic hears them attenuated."""
    rng = np.random.default_rng(seed)
    n = seconds * rate
    tracks = [rng.standard_normal(n).astype(np.float32) * 1e-4 for _ in range(n_mics)]
    talk_fraction = talk_fraction or [1.0 / n_mics] * n_mics
    bleed = 10 ** (bleed_db / 20)

    start = 0
    for j in range(n_mics):
        length = int(n * talk_fraction[j])
        sl = slice(start, start + length)
        start += length
        # a voiced-ish source: harmonic stack on a 120 Hz fundamental
        t = np.arange(length) / rate
        voice = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 12))
        voice = (voice / np.abs(voice).max() * 0.3).astype(np.float32)
        for i in range(n_mics):
            tracks[i][sl] += voice if i == j else voice * bleed

    if mic_gains:
        tracks = [t * g for t, g in zip(tracks, mic_gains)]
    return tracks


def test_frame_levels_matches_reference():
    x = (np.random.default_rng(0).standard_normal(HOP * 20) * 0.1).astype(np.float32)
    want = np.sqrt((x.reshape(20, HOP).astype(np.float64) ** 2).mean(axis=1))
    assert np.allclose(frame_levels(x, HOP) - 1e-12, want)


def test_noise_floor_survives_a_transient():
    """The property `soft_gate` lacked: one loud bump must not move the reference."""
    x = (np.random.default_rng(0).standard_normal(HOP * 2000) * 0.01).astype(np.float32)
    quiet = noise_floor(frame_levels(x, HOP))
    x[HOP * 500:HOP * 500 + 200] = 1.0          # a mic bump: +40 dB over everything
    assert np.isclose(noise_floor(frame_levels(x, HOP)), quiet, rtol=1e-6)
    # a peak reference, by contrast, moves by the full height of the transient
    assert np.abs(x).max() / quiet > 100


def test_speech_level_is_talk_time_invariant():
    """RMS calibration is biased by talk time; speech-level calibration is not."""
    chatty = synth_array(2, talk_fraction=[0.7, 0.2], seed=1)
    lev = [frame_levels(t, HOP) for t in chatty]
    speech = np.array([20 * np.log10(speech_level(l)) for l in lev])
    rms = np.array([20 * np.log10(np.sqrt(np.mean(l ** 2))) for l in lev])
    # both mics carry the same voice at the same gain, so speech levels must match
    assert abs(speech[0] - speech[1]) < 1.0
    # while overall RMS differs purely because one person talked more
    assert abs(rms[0] - rms[1]) > 3.0


@pytest.mark.parametrize('method', ['speech_level', 'bleed_symmetry'])
def test_calibration_recovers_mic_gains(method):
    gains = [1.0, 2.0, 0.5, 1.0]
    tracks = synth_array(4, mic_gains=gains, seed=2)
    lev = [frame_levels(t, HOP) for t in tracks]
    got, info = calibration_gains(lev, method=method)
    corrected = 20 * np.log10(np.array(gains) * got)
    assert np.ptp(corrected) < 1.0, info


def test_bleed_matrix_is_symmetric_for_a_matched_array():
    tracks = synth_array(4, bleed_db=-25.0, seed=3)
    B, counts = bleed_matrix([frame_levels(t, HOP) for t in tracks])
    off = B[~np.eye(4, dtype=bool)]
    assert np.isfinite(off).all()
    assert np.allclose(off, -25.0, atol=2.0)
    assert np.allclose(B, B.T, atol=1.0)


def test_max_excluding_matches_the_naive_reduction_and_restores_input():
    m = np.abs(np.random.default_rng(0).standard_normal((4, 40, 60))).astype(np.float32)
    original = m.copy()
    top1, top2, idx = max_excluding(m)
    assert np.array_equal(m, original), "input must be left untouched"
    for i in range(4):
        want = np.max(np.delete(original, i, axis=0), axis=0)
        assert np.allclose(np.where(idx == i, top2, top1), want)


def test_dominance_mask_is_floored_and_monotone():
    # smoothing off, so this measures the mask curve rather than the neighbourhood filter
    cfg = IsolationConfig(smoothing_freq_bins=1, smoothing_time_frames=1)
    target = np.ones((8, 3), dtype=np.float32)
    other = np.tile(np.array([100.0, 1.0, 0.01], dtype=np.float32), (8, 1))
    m = dominance_mask(target, other, cfg)                     # -40, 0, +40 dB
    assert m.min() >= cfg.mask_floor - 1e-6, "a bin must never close completely"
    assert (m[:, 0] < m[:, 1]).all() and (m[:, 1] < m[:, 2]).all()
    assert m[:, 2].min() > 0.95, "a clearly dominant bin must pass essentially intact"
    assert m[:, 0].max() < cfg.mask_floor + 0.02, "a clearly lost bin must sit at the floor"


def test_mask_smoothing_preserves_the_floor():
    cfg = IsolationConfig()
    rng = np.random.default_rng(0)
    target = np.abs(rng.standard_normal((64, 64))).astype(np.float32)
    other = np.abs(rng.standard_normal((64, 64))).astype(np.float32) * 100
    m = dominance_mask(target, other, cfg)
    assert m.min() >= cfg.mask_floor - 1e-6
    assert m.max() <= 1.0 + 1e-9


def test_speech_presence_needs_both_dominance_and_level():
    cfg = IsolationConfig()
    n = 400
    quiet, loud = 1e-5, 1e-1
    lev = np.full((2, n), quiet)
    lev[0, 100:200] = loud                 # track 0 speaks: dominant and well above floor
    lev[1, 250:350] = loud                 # track 1 speaks
    p = speech_presence(lev, 0, quiet, cfg)
    assert p[150] > 0.9, "dominant and loud must pass"
    assert p[300] < 0.1, "loud on another mic must not pass"
    assert p[50] < 0.1, "quiet and undominant must not pass"


def test_hangover_is_symmetric():
    """Protects onsets as well as unvoiced offsets; v2's release-only gate clipped both."""
    cfg = IsolationConfig(hangover_ms=100.0)
    n = 400
    lev = np.full((2, n), 1e-5)
    lev[0, 200:210] = 1e-1
    p = speech_presence(lev, 0, 1e-5, cfg)
    hang = int(round(cfg.hangover_ms / 1000 * cfg.frame_rate))
    assert p[200 - hang + 1] > 0.9, "gate must be open before the onset"
    assert p[210 + hang - 1] > 0.9, "gate must stay open past the offset"


def test_isolate_v3_never_destroys_speech(tmp_path):
    from scipy.io import wavfile
    tracks = synth_array(3, seconds=12, seed=5)
    files = []
    for i, t in enumerate(tracks):
        p = tmp_path / f'TRACK0{i + 1}.wav'
        wavfile.write(str(p), RATE, (t * 32767).astype(np.int16))
        files.append(str(p))

    cfg = IsolationConfig(block_seconds=5.0, block_pad_seconds=0.5)
    res = isolate_v3(files, config=cfg, verbose=False)

    assert len(res['audio']) == 3
    _, raw = load_tracks(files)
    gains = 10 ** (np.array(res['calibration']['gains_db']) / 20)
    raw = [r * g for r, g in zip(raw, gains)]
    lev = [frame_levels(r, HOP) for r in raw]

    for i in range(3):
        others = [lev[j] for j in range(3) if j != i]
        q = track_qc(raw[i], res['audio'][i], others, RATE, hop=HOP)
        assert q['speech_silenced_s'] == 0.0
        assert q.get('speech_cut_12_frac', 0.0) <= 0.05, q


def test_isolate_v3_respects_the_residual_floor():
    """Total attenuation is bounded, which is what v2's unbounded expander was not."""
    cfg = IsolationConfig(residual_floor_db=-20.0)
    assert np.isclose(cfg.residual_floor, 0.1)
    n = 400
    lev = np.full((2, n), 1e-5)
    lev[1, 100:200] = 1e-1                 # the *other* mic is speaking
    p = speech_presence(lev, 0, 1e-5, cfg)
    gain = cfg.residual_floor + (1 - cfg.residual_floor) * p
    assert gain.min() >= cfg.residual_floor - 1e-6
    assert 20 * np.log10(gain.min()) >= cfg.residual_floor_db - 1e-6


def test_blocking_does_not_change_the_result(tmp_path):
    from scipy.io import wavfile
    tracks = synth_array(2, seconds=12, seed=7)
    files = []
    for i, t in enumerate(tracks):
        p = tmp_path / f'TRACK0{i + 1}.wav'
        wavfile.write(str(p), RATE, (t * 32767).astype(np.int16))
        files.append(str(p))

    one = isolate_v3(files, config=IsolationConfig(block_seconds=600.0), verbose=False)
    many = isolate_v3(files, config=IsolationConfig(block_seconds=3.0), verbose=False)
    for a, b in zip(one['audio'], many['audio']):
        assert np.abs(a - b).max() < 1e-4


def test_speech_silenced_seconds_only_counts_zeros_over_speech():
    out = np.ones(HOP * 10, dtype=np.float32)
    out[HOP * 2:HOP * 4] = 0.0
    speech = np.zeros(10, dtype=bool)
    assert speech_silenced_seconds(out, speech, RATE, HOP) == 0.0
    speech[2:4] = True
    assert np.isclose(speech_silenced_seconds(out, speech, RATE, HOP), 2 * HOP / RATE)


def _two_speaker_files(tmp_path, seconds=12, bleed_db=-25.0, seed=11):
    """Speaker A talks in the first half, B in the second; each bleeds into the other mic."""
    from scipy.io import wavfile
    rng = np.random.default_rng(seed)
    n = seconds * RATE
    half = n // 2
    t = np.arange(half) / RATE
    voices = []
    for f0 in (120.0, 175.0):
        v = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        # syllable-rate modulation, which is what an ASR front end actually reads
        v *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
        voices.append((v / np.abs(v).max() * 0.3).astype(np.float32))

    bleed = 10 ** (bleed_db / 20)
    tracks = [rng.standard_normal(n).astype(np.float32) * 1e-4 for _ in range(2)]
    tracks[0][:half] += voices[0]
    tracks[1][:half] += voices[0] * bleed
    tracks[1][half:] += voices[1]
    tracks[0][half:] += voices[1] * bleed

    files = []
    for i, tr in enumerate(tracks):
        p = tmp_path / f'TRACK0{i + 1}.wav'
        wavfile.write(str(p), RATE, (tr * 32767).astype(np.int16))
        files.append(str(p))
    return files, voices, half


def _leak_correlation(out, interferer, start, stop):
    """How much of the interfering voice is still linearly present in the output."""
    a = out[start:stop].astype(np.float64)
    b = interferer.astype(np.float64)[:stop - start]
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return 0.0 if denom == 0 else float(abs((a * b).sum()) / denom)


def test_comfort_noise_fills_at_the_configured_level(tmp_path):
    """In frames where the target is confidently absent, the fill must raise the
    residual by the configured amount; where they are speaking it must add nothing."""
    files, voices, half = _two_speaker_files(tmp_path)
    over_db = 6.0
    off = isolate_v3(files, config=IsolationConfig(residual_noise_over_db=None),
                     verbose=False)
    on = isolate_v3(files, config=IsolationConfig(residual_noise_over_db=over_db),
                    verbose=False)

    a = frame_levels(off['audio'][0], HOP)
    b = frame_levels(on['audio'][0], HOP)
    p = on['speech_probability'][0]
    n = min(len(a), len(b), len(p))
    a, b, p = a[:n], b[:n], p[:n]
    ratio_db = 20 * np.log10(b / a)

    absent = p < 0.05
    assert absent.sum() > 50
    # total = residual + noise at `over_db` above it -> 10*log10(1 + 10**(over/10))
    expected = 10 * np.log10(1 + 10 ** (over_db / 10))
    assert abs(np.median(ratio_db[absent]) - expected) < 1.5, np.median(ratio_db[absent])

    present = p > 0.95
    assert present.sum() > 50
    assert np.median(ratio_db[present]) < 0.5, "fill leaked into the target's own speech"


def test_comfort_noise_scales_with_its_setting(tmp_path):
    files, _, _ = _two_speaker_files(tmp_path)
    levels = {}
    for over_db in (3.0, 12.0):
        res = isolate_v3(files, config=IsolationConfig(residual_noise_over_db=over_db),
                         verbose=False)
        lv = frame_levels(res['audio'][0], HOP)
        p = res['speech_probability'][0]
        n = min(len(lv), len(p))
        levels[over_db] = np.median(lv[:n][p[:n] < 0.05])
    gap = 20 * np.log10(levels[12.0] / levels[3.0])
    expected = (10 * np.log10(1 + 10 ** 1.2)) - (10 * np.log10(1 + 10 ** 0.3))
    assert abs(gap - expected) < 1.5, gap


def test_comfort_noise_leaves_no_silent_holes(tmp_path):
    files, _, _ = _two_speaker_files(tmp_path)
    res = isolate_v3(files, config=IsolationConfig(residual_noise_over_db=6.0,
                                                   block_seconds=5.0,
                                                   block_pad_seconds=0.5), verbose=False)
    for audio in res['audio']:
        # no run of exact zeros survives a noise fill
        assert not np.any(np.convolve((audio == 0).astype(np.int32),
                                      np.ones(RATE // 10, dtype=np.int32),
                                      mode='valid') == RATE // 10)
