#!/usr/bin/env python
"""Isolate every microphone in a session with the v3 cross-microphone path.

The single entry point for isolation. It replaces three scripts that called the
library with divergent defaults -- `multidata/run_isolation.py` used the library
defaults, while the two copies of `isolate_all_group_audio.py` overrode them with
`--dominance 3.0 --gate-threshold -40` and `--dominance 1.0 --gate-threshold -75`
respectively -- and nothing recorded which had produced a given output file.

Every run writes, next to the audio:

    TRACK0N_trimmed_isolated_v3.wav   isolated audio
    TRACK0N_speaking_v3.npy           per-frame speech probability (the VAD that
                                      drove the suppression), with rate metadata
    {session}_isolation_params.json   the exact config and voicolate git SHA
    {session}_isolation_qc.json       measured speech retention and bleed rejection

Usage:
    python scripts/isolate_session.py /path/to/session
    python scripts/isolate_session.py /path/to/sessions --all
    python scripts/isolate_session.py /path/to/session --force --calibration bleed_symmetry
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voicolate.config import IsolationConfig
from voicolate.isolate import isolate_v3, load_tracks
from voicolate.qc import session_qc, format_report

SUFFIX = '_isolated_v3'


def voicolate_sha():
    try:
        return subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def audio_files(session):
    return sorted((session / 'derivatives').glob('TRACK*_trimmed.wav'))


def find_sessions(base):
    if audio_files(base):
        return [base]
    return [d for d in sorted(base.iterdir())
            if d.is_dir() and not d.name.startswith('.') and audio_files(d)]


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def isolate_session(session, cfg, force=False, run_qc=True):
    files = audio_files(session)
    if not files:
        print(f"  no TRACK*_trimmed.wav in {session / 'derivatives'}")
        return False

    processed = session / 'processed'
    stems = [f.stem for f in files]
    outputs = [processed / f'{s}{SUFFIX}.wav' for s in stems]
    if all(o.exists() for o in outputs) and not force:
        print(f"  already isolated, skipping (use --force)")
        return True

    print(f"  {len(files)} tracks: {', '.join(stems)}")
    processed.mkdir(exist_ok=True)

    result = isolate_v3([str(f) for f in files], config=cfg)
    rate = result['rate']

    for stem, audio, prob in zip(stems, result['audio'], result['speech_probability']):
        wavfile.write(str(processed / f'{stem}{SUFFIX}.wav'), rate, to_int16(audio))
        np.savez(str(processed / f'{stem}_speaking_v3.npz'),
                 speech_probability=prob, frame_rate=result['frame_rate'],
                 hop=result['hop'], rate=rate)
    print(f"  wrote {len(stems)} isolated tracks + speech probabilities")

    params = {k: v for k, v in result.items()
              if k not in ('audio', 'speech_probability')}
    params['voicolate_sha'] = voicolate_sha()
    (processed / f'{session.name}_isolation_params.json').write_text(
        json.dumps(params, indent=2))

    if run_qc:
        # QC compares against the *calibrated* raw audio, so reapply the gains
        _, raw = load_tracks([str(f) for f in files])
        gains = 10 ** (np.array(result['calibration']['gains_db']) / 20)
        raw = [r * g for r, g in zip(raw, gains)]
        # QC the audio that was actually written, not the float array it came from:
        # int16 quantisation is part of the artefact downstream tools will read
        written = [to_int16(a).astype(np.float32) / 32768.0 for a in result['audio']]
        qc = session_qc(raw, written, rate, hop=result['hop'], names=stems,
                        residual_floor_db=cfg.residual_floor_db)
        (processed / f'{session.name}_isolation_qc.json').write_text(
            json.dumps(qc, indent=2))
        print(format_report(qc, session.name))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', type=Path, help='session directory, or a directory of them')
    p.add_argument('--all', action='store_true', help='process every session under path')
    p.add_argument('--force', action='store_true', help='reprocess existing outputs')
    p.add_argument('--no-qc', action='store_true', help='skip the QC pass')
    p.add_argument('--calibration', default=None,
                   choices=['speech_level', 'bleed_symmetry', 'rms', 'none'])
    p.add_argument('--dominance-margin-db', type=float, default=None)
    p.add_argument('--dominance-width-db', type=float, default=None)
    p.add_argument('--residual-floor-db', type=float, default=None,
                   help='hard limit on total attenuation (default -20)')
    p.add_argument('--presence-snr-db', type=float, default=None,
                   help='dB above a track\'s own noise floor for speech presence')
    p.add_argument('--residual-noise-over-db', type=float, default=None,
                   help='fill suppressed regions with ambience-shaped noise this many dB '
                        'over the residual (off by default; removes stray cross-speaker '
                        'words but costs ~4%% of the target\'s own words -- see config.py)')
    p.add_argument('--hangover-ms', type=float, default=None)
    p.add_argument('--block-seconds', type=float, default=None)
    args = p.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k in IsolationConfig.__dataclass_fields__}
    cfg = IsolationConfig(**overrides)

    if not args.path.exists():
        print(f"path does not exist: {args.path}")
        return 1
    sessions = find_sessions(args.path) if args.all else [args.path]
    if not sessions:
        print(f"no sessions found under {args.path}")
        return 1

    failed = []
    for i, s in enumerate(sessions, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(sessions)}] {s.name}\n{'=' * 78}")
        try:
            if not isolate_session(s, cfg, force=args.force, run_qc=not args.no_qc):
                failed.append(s.name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            failed.append(s.name)

    print(f"\n{len(sessions) - len(failed)}/{len(sessions)} sessions isolated")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
