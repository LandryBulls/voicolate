#!/usr/bin/env python
"""Transcribe every session's v3-isolated tracks with WhisperX.

Third step after `stage_raw_audio.py` and `isolate_all_sessions.py`. Runs in the
``annotate`` conda env, the only one with whisperx and CUDA:

    /safestore/users/landry/miniconda3/envs/annotate/bin/python scripts/transcribe_all_sessions.py

Reuses `groupconv.extract.transcription.TranscriptionExtractor` -- the same class,
model (WhisperX large-v2 + alignment) and JSON schema the extraction pipeline uses
for the v2 transcripts -- so the output is a drop-in replacement.

Tracks map to person letters the way the pipeline does it, by sorted track order:
TRACK01 -> A, TRACK02 -> B, TRACK03 -> C, TRACK04 -> D. By default each becomes
``{letter}_transcript_v3.json``, leaving the pipeline's existing v2-derived
``{letter}_transcript_with_nonwords.json`` untouched. Pass ``--pipeline-names`` to
write to that filename instead, which is what the downstream pipeline reads --
that replaces the v2 transcripts, so use it when you are ready to switch.

The model is loaded once for the whole run. Sessions already transcribed are
skipped unless ``--force``; a failure in one session is logged and the run
carries on.

Usage:
    python scripts/transcribe_all_sessions.py
    python scripts/transcribe_all_sessions.py --sessions 2024-10-14_001
    python scripts/transcribe_all_sessions.py --force
    python scripts/transcribe_all_sessions.py --pipeline-names   # overwrite v2 transcripts
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/safescratch/groups/session_data')
GROUPCONV_SRC = Path('/safestore/users/landry/SCRAP/analyses/group_conversation_multimodal/src')
ANNOTATE_PY = '/safestore/users/landry/miniconda3/envs/annotate/bin/python'
AUDIO_SUFFIX = '_trimmed_isolated_v3.wav'
LETTERS = 'ABCD'


def log(fh, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    fh.write(line + '\n'); fh.flush()


def v3_tracks(session):
    return sorted((session / 'processed').glob(f'TRACK0*{AUDIO_SUFFIX}'))


def transcript_path(audio, letter, pipeline_names):
    name = (f'{letter}_transcript_with_nonwords.json' if pipeline_names
            else f'{letter}_transcript_v3.json')
    return audio.parent / name


def letter_for(tracks):
    """Person letter per track, by sorted order -- the pipeline's own mapping."""
    if len(tracks) > len(LETTERS):
        raise ValueError(f"{len(tracks)} tracks but only letters {LETTERS}: {[t.name for t in tracks]}")
    return {t: LETTERS[i] for i, t in enumerate(tracks)}


def load_extractor(device):
    try:
        import whisperx  # noqa: F401
    except ImportError:
        sys.exit(f"whisperx is not importable from {sys.executable}\n"
                 f"run this script with the annotate env:\n  {ANNOTATE_PY} {' '.join(sys.argv)}")
    sys.path.insert(0, str(GROUPCONV_SRC))
    from groupconv.extract.transcription import TranscriptionExtractor
    return TranscriptionExtractor({'method': 'whisperx', 'model': 'large-v2',
                                   'device': device, 'output_format': 'json'})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT, help='directory of session directories')
    p.add_argument('--sessions', nargs='+', default=None,
                   help='only these session ids (default: every session with v3 audio)')
    p.add_argument('--include-removed', action='store_true',
                   help='also transcribe sessions whose name ends in _remove')
    p.add_argument('--force', action='store_true', help='re-transcribe existing outputs')
    p.add_argument('--pipeline-names', action='store_true',
                   help='write {letter}_transcript_with_nonwords.json, the file the extraction '
                        'pipeline reads -- this REPLACES the v2-derived transcripts')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--log', type=Path, default=None,
                   help='log file (default: <root>/transcribe_all_sessions_<date>.log)')
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"root not found: {args.root}"); return 1

    if args.sessions:
        sessions = [args.root / s for s in args.sessions]
    else:
        sessions = sorted(d for d in args.root.iterdir() if d.is_dir())
        if not args.include_removed:
            sessions = [d for d in sessions if not d.name.endswith('_remove')]

    # (session, [tracks needing transcription])
    work = []
    for sess in sessions:
        tracks = v3_tracks(sess)
        if not tracks:
            continue
        letters = letter_for(tracks)
        todo = [t for t in tracks
                if args.force or not transcript_path(t, letters[t], args.pipeline_names).exists()]
        work.append((sess, tracks, letters, todo))
    n_todo = sum(len(todo) for *_, todo in work)

    log_path = args.log or args.root / f"transcribe_all_sessions_{time.strftime('%Y%m%d')}.log"
    with open(log_path, 'a') as fh:
        log(fh, f"{len(work)} sessions with v3 audio under {args.root}; "
                f"{n_todo} tracks to transcribe"
                + ("  [--pipeline-names: replacing v2-derived transcripts]" if args.pipeline_names else ""))
        if not n_todo:
            log(fh, "nothing to do"); return 0

        extractor = load_extractor(args.device)
        log(fh, f"loaded WhisperX large-v2 on {args.device}")

        done, failed = 0, []
        t0 = time.time()
        for i, (sess, tracks, letters, todo) in enumerate(work, 1):
            if not todo:
                log(fh, f"[{i}/{len(work)}] {sess.name}: all {len(tracks)} transcripts exist, skipping")
                continue
            for audio in todo:
                letter = letters[audio]
                out = transcript_path(audio, letter, args.pipeline_names)
                t = time.time()
                try:
                    extractor.extract(audio, out, person_letter=letter, overwrite=args.force)
                    done += 1
                    log(fh, f"[{i}/{len(work)}] {sess.name}: {audio.name} -> {out.name} "
                            f"in {time.time() - t:.0f} s")
                except Exception as e:
                    failed.append(f"{sess.name}/{audio.name}")
                    log(fh, f"[{i}/{len(work)}] {sess.name}: FAILED {audio.name}: {e}")
                    fh.write(traceback.format_exc()); fh.flush()

        log(fh, f"done in {(time.time() - t0) / 60:.1f} min: {done} transcribed, {len(failed)} failed"
                + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
