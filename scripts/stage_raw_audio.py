#!/usr/bin/env python
"""Copy each session's raw trimmed tracks from andromeda to fast scratch.

Andromeda is a RAID array; isolation reads every track several times, so the
tracks are staged once here and processed from scratch. For every session
directory under ``--dst`` that also exists under ``--src``, copies
``derivatives/TRACK0*_trimmed.wav`` into the same relative place.

Safe to re-run: a file already present with the right size is skipped, and each
copy goes through a temporary name and rename so an interrupted run never leaves
a truncated file that looks complete.

Usage:
    python scripts/stage_raw_audio.py                    # every session under --dst
    python scripts/stage_raw_audio.py --sessions 2024-10-14_001 2024-10-15_000
    python scripts/stage_raw_audio.py --dry-run
    python scripts/stage_raw_audio.py --include-removed  # also *_remove sessions
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

SRC = Path('/safestore/users/landry/SCRAP/data/andromeda_storage/conversations_unconstrained')
DST = Path('/safescratch/groups/session_data')
PATTERN = 'TRACK0*_trimmed.wav'


def log(fh, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    fh.write(line + '\n'); fh.flush()


def copy_atomic(src: Path, dst: Path):
    tmp = dst.with_name(dst.name + '.part')
    shutil.copyfile(src, tmp)
    os.chmod(tmp, src.stat().st_mode & 0o777)
    os.replace(tmp, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=SRC, help='andromeda sessions root')
    p.add_argument('--dst', type=Path, default=DST, help='scratch sessions root')
    p.add_argument('--sessions', nargs='+', default=None,
                   help='only these session ids (default: every directory under --dst)')
    p.add_argument('--include-removed', action='store_true',
                   help='also stage sessions whose name ends in _remove')
    p.add_argument('--dry-run', action='store_true', help='report what would be copied')
    p.add_argument('--log', type=Path, default=None,
                   help='log file (default: <dst>/stage_raw_audio_<date>.log)')
    args = p.parse_args()

    if not args.src.is_dir():
        print(f"source root not found: {args.src}"); return 1
    if not args.dst.is_dir():
        print(f"destination root not found: {args.dst}"); return 1

    if args.sessions:
        sessions = [args.dst / s for s in args.sessions]
    else:
        sessions = sorted(d for d in args.dst.iterdir() if d.is_dir())
        if not args.include_removed:
            sessions = [d for d in sessions if not d.name.endswith('_remove')]

    log_path = args.log or args.dst / f"stage_raw_audio_{time.strftime('%Y%m%d')}.log"
    counts = dict(copied=0, skipped=0, missing=0, failed=0)
    bytes_copied = 0
    t0 = time.time()

    with open(log_path, 'a') as fh:
        log(fh, f"staging {len(sessions)} sessions  {args.src} -> {args.dst}"
                + ("  [DRY RUN]" if args.dry_run else ""))
        for i, sess in enumerate(sessions, 1):
            src_dir = args.src / sess.name / 'derivatives'
            files = sorted(src_dir.glob(PATTERN)) if src_dir.is_dir() else []
            if not files:
                log(fh, f"[{i}/{len(sessions)}] {sess.name}: no {PATTERN} on andromeda, skipping")
                counts['missing'] += 1
                continue

            dst_dir = sess / 'derivatives'
            for f in files:
                dst = dst_dir / f.name
                if dst.exists() and dst.stat().st_size == f.stat().st_size:
                    counts['skipped'] += 1
                    continue
                size_mb = f.stat().st_size / 1e6
                if args.dry_run:
                    log(fh, f"[{i}/{len(sessions)}] {sess.name}: would copy {f.name} ({size_mb:.0f} MB)")
                    counts['copied'] += 1
                    continue
                try:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    t = time.time()
                    copy_atomic(f, dst)
                    dt = time.time() - t
                    log(fh, f"[{i}/{len(sessions)}] {sess.name}: {f.name} "
                            f"({size_mb:.0f} MB in {dt:.0f} s, {size_mb / max(dt, 1e-3):.0f} MB/s)")
                    counts['copied'] += 1
                    bytes_copied += f.stat().st_size
                except Exception as e:
                    log(fh, f"[{i}/{len(sessions)}] {sess.name}: FAILED {f.name}: {e}")
                    counts['failed'] += 1

        log(fh, f"done in {(time.time() - t0) / 60:.1f} min: {counts['copied']} copied "
                f"({bytes_copied / 1e9:.1f} GB), {counts['skipped']} already present, "
                f"{counts['missing']} sessions without tracks on andromeda, {counts['failed']} failed")
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
