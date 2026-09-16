#!/usr/bin/env python
"""Run v3 isolation over every staged session and summarise the QC across them.

Wraps `isolate_session.py` for the batch case. For each session directory under
``--root`` that holds ``derivatives/TRACK0*_trimmed.wav`` (stage them first with
`stage_raw_audio.py`), writes into that session's ``processed/``:

    TRACK0N_trimmed_isolated_v3.wav
    TRACK0N_trimmed_speaking_v3.npz
    {session}_isolation_params.json
    {session}_isolation_qc.json

then collects every session's QC into one CSV and prints the sessions worth
listening to, ranked by how much clear speech was cut. Sessions already isolated
are skipped unless ``--force``; a failure in one session is logged and the run
carries on.

Usage:
    python scripts/isolate_all_sessions.py
    python scripts/isolate_all_sessions.py --sessions 2024-10-14_001 2024-10-15_000
    python scripts/isolate_all_sessions.py --force --residual-noise-over-db 3
    python scripts/isolate_all_sessions.py --summary-only   # just rebuild the CSV
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isolate_session import isolate_session, audio_files, add_config_args, config_from_args

ROOT = Path('/safescratch/groups/session_data')
QC_FIELDS = ['speech_seconds', 'noise_floor_db', 'speech_level_db', 'peak_over_speech_db',
             'speech_gain_median_db', 'speech_cut_6_frac', 'speech_cut_12_frac',
             'speech_cut_20_frac', 'bleed_gain_median_db', 'zero_fraction',
             'longest_silence_s', 'speech_silenced_s']


def log(fh, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    fh.write(line + '\n'); fh.flush()


def summarise(sessions, out_csv):
    """One row per track across every session that has a QC file."""
    rows = []
    for sess in sessions:
        qc_path = sess / 'processed' / f'{sess.name}_isolation_qc.json'
        if not qc_path.exists():
            continue
        qc = json.loads(qc_path.read_text())
        for track, t in qc['tracks'].items():
            row = {'session': sess.name, 'track': track, 'n_tracks': len(qc['tracks'])}
            row.update({k: t.get(k) for k in QC_FIELDS})
            row['warnings'] = ' | '.join(t.get('warnings', []))
            rows.append(row)
    if not rows:
        return rows
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def print_tail(rows, n=10):
    scored = [r for r in rows if r.get('speech_cut_12_frac') is not None]
    print(f"\n{len(rows)} tracks across {len({r['session'] for r in rows})} sessions")
    warned = [r for r in rows if r['warnings']]
    print(f"{len(warned)} tracks with warnings")
    print(f"\nmost clear speech cut >12 dB (top {n}):")
    print(f"  {'session':22s} {'track':16s} {'cut>12':>7s} {'bleed':>7s} {'spk_sil':>8s}  warnings")
    for r in sorted(scored, key=lambda r: -r['speech_cut_12_frac'])[:n]:
        print(f"  {r['session']:22s} {r['track']:16s} {r['speech_cut_12_frac']:7.3f} "
              f"{(r['bleed_gain_median_db'] if r['bleed_gain_median_db'] is not None else float('nan')):7.1f} "
              f"{r['speech_silenced_s']:8.2f}  {r['warnings']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT, help='directory of session directories')
    p.add_argument('--sessions', nargs='+', default=None,
                   help='only these session ids (default: every staged session under --root)')
    p.add_argument('--include-removed', action='store_true',
                   help='also process sessions whose name ends in _remove')
    p.add_argument('--force', action='store_true', help='reprocess sessions already isolated')
    p.add_argument('--no-qc', action='store_true', help='skip the per-session QC pass')
    p.add_argument('--summary-only', action='store_true',
                   help='do not isolate anything; rebuild the CSV from existing QC files')
    p.add_argument('--summary', type=Path, default=None,
                   help='CSV path (default: <root>/isolation_v3_qc_summary.csv)')
    p.add_argument('--log', type=Path, default=None,
                   help='log file (default: <root>/isolate_all_sessions_<date>.log)')
    add_config_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)

    if not args.root.is_dir():
        print(f"root not found: {args.root}"); return 1

    if args.sessions:
        sessions = [args.root / s for s in args.sessions]
    else:
        sessions = sorted(d for d in args.root.iterdir() if d.is_dir())
        if not args.include_removed:
            sessions = [d for d in sessions if not d.name.endswith('_remove')]

    summary_csv = args.summary or args.root / 'isolation_v3_qc_summary.csv'

    if not args.summary_only:
        staged = [s for s in sessions if audio_files(s)]
        unstaged = [s.name for s in sessions if s not in staged]
        log_path = args.log or args.root / f"isolate_all_sessions_{time.strftime('%Y%m%d')}.log"
        done, failed = [], []
        t0 = time.time()
        with open(log_path, 'a') as fh:
            log(fh, f"{len(staged)} staged sessions under {args.root}"
                    + (f"; {len(unstaged)} without raw tracks skipped: {', '.join(unstaged)}"
                       if unstaged else ""))
            log(fh, f"config: {json.dumps(cfg.to_dict())}")
            for i, sess in enumerate(staged, 1):
                log(fh, f"[{i}/{len(staged)}] {sess.name}")
                t = time.time()
                try:
                    ok = isolate_session(sess, cfg, force=args.force, run_qc=not args.no_qc)
                    (done if ok else failed).append(sess.name)
                    log(fh, f"[{i}/{len(staged)}] {sess.name}: "
                            f"{'ok' if ok else 'FAILED'} in {time.time() - t:.0f} s")
                except Exception as e:
                    failed.append(sess.name)
                    log(fh, f"[{i}/{len(staged)}] {sess.name}: FAILED in {time.time() - t:.0f} s: {e}")
                    fh.write(traceback.format_exc()); fh.flush()
            log(fh, f"done in {(time.time() - t0) / 60:.1f} min: {len(done)} ok, {len(failed)} failed"
                    + (f": {', '.join(failed)}" if failed else ""))
        sessions = staged

    rows = summarise(sessions, summary_csv)
    if rows:
        print_tail(rows)
        print(f"\nsummary written to {summary_csv}")
    else:
        print("no QC files found to summarise")
    return 0


if __name__ == '__main__':
    sys.exit(main())
