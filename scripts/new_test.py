"""
Grid search for voicolate isolate_spectral parameters.

Tests combinations of dominance_margin_db and harmonicity_weight
to find optimal settings for the 2023-09-18_000 session.

Uses only the middle 5 minutes of audio for faster iteration.
"""

from voicolate.isolate import isolate_spectral
from glob import glob
import os
import json
import itertools
from datetime import datetime
from scipy.io import wavfile
import numpy as np
import shutil


# Input data
ddir = '/fastscratch/2023-09-18_000_audio/raw'
files = sorted(glob(f'{ddir}/*.wav'))

# Output base directory
output_base = '/fastscratch/2023-09-18_000_audio/grid_search'
os.makedirs(output_base, exist_ok=True)

# Extract middle 5 minutes for faster testing
EXTRACT_DURATION_MIN = 5
temp_dir = os.path.join(output_base, '_temp_trimmed')
os.makedirs(temp_dir, exist_ok=True)

print(f"Extracting middle {EXTRACT_DURATION_MIN} minutes from each file...\n")
trimmed_files = []
for f in files:
    rate, audio = wavfile.read(f)
    total_samples = len(audio)
    total_duration_sec = total_samples / rate
    
    # Calculate middle portion
    extract_samples = int(EXTRACT_DURATION_MIN * 60 * rate)
    start_sample = (total_samples - extract_samples) // 2
    end_sample = start_sample + extract_samples
    
    # Extract
    trimmed_audio = audio[start_sample:end_sample]
    
    # Save temp file
    basename = os.path.basename(f).replace('.wav', '_trimmed.wav')
    temp_path = os.path.join(temp_dir, basename)
    wavfile.write(temp_path, rate, trimmed_audio)
    trimmed_files.append(temp_path)
    
    print(f"  {os.path.basename(f)}: {total_duration_sec/60:.1f} min -> middle {EXTRACT_DURATION_MIN} min")

# Use trimmed files for grid search
files = trimmed_files
print(f"\nTrimmed files saved to: {temp_dir}\n")

# Parameter grid
param_grid = {
    'dominance_margin_db': [1.0, 3.0],      # Lower = more aggressive
    'harmonicity_weight': [0.5, 0.7],  # Higher = more breath suppression
    'smoothing_time_frames': [3],                 # Fixed for now
    'smoothing_freq_bins': [5],                   # Fixed for now
    'nperseg': [2048],                            # Fixed for now
    'apply_gate': [True],                         # Soft gate for residual bleedthrough
    'gate_threshold_db': [-75],              # dB below peak (-35 = more aggressive)
    'gate_ratio': [4.0],                          # Expansion ratio
}

# Generate all combinations
param_names = list(param_grid.keys())
param_values = list(param_grid.values())
combinations = list(itertools.product(*param_values))

print(f"Running grid search with {len(combinations)} parameter combinations\n")
print(f"Input files: {[os.path.basename(f) for f in files]}\n")

# Run grid search
for i, combo in enumerate(combinations):
    params = dict(zip(param_names, combo))
    
    # Create descriptive folder name
    gate_str = f"_gate{abs(params['gate_threshold_db'])}" if params.get('apply_gate', True) else "_nogate"
    folder_name = f"dom{params['dominance_margin_db']}_harm{params['harmonicity_weight']}{gate_str}"
    output_dir = os.path.join(output_base, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(combinations)}] {folder_name}")
    print(f"{'='*60}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_files': [os.path.basename(f) for f in files],
        'source_files': [os.path.basename(f) for f in sorted(glob(f'{ddir}/*.wav'))],
        'audio_extraction': {
            'duration_minutes': EXTRACT_DURATION_MIN,
            'position': 'middle',
        },
        'parameters': params,
        'description': {
            'dominance_margin_db': 'dB advantage required to be considered dominant (lower = more aggressive)',
            'harmonicity_weight': 'Weight for harmonicity penalty (higher = more breath/friction suppression)',
            'smoothing_time_frames': 'Temporal smoothing of mask',
            'smoothing_freq_bins': 'Frequency smoothing of mask',
            'nperseg': 'STFT window size',
            'apply_gate': 'Whether to apply soft noise gate for residual bleedthrough',
            'gate_threshold_db': 'Gate threshold in dB below peak (more negative = less aggressive)',
            'gate_ratio': 'Expansion ratio (higher = more attenuation below threshold)',
        }
    }
    
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run isolation
    try:
        isolated = isolate_spectral(
            files,
            dominance_margin_db=params['dominance_margin_db'],
            harmonicity_weight=params['harmonicity_weight'],
            smoothing_time_frames=params['smoothing_time_frames'],
            smoothing_freq_bins=params['smoothing_freq_bins'],
            nperseg=params['nperseg'],
            apply_gate=params.get('apply_gate', True),
            gate_threshold_db=params.get('gate_threshold_db', -40),
            gate_ratio=params.get('gate_ratio', 4.0),
            normalize_input=True,
            save_files=True,
            output_path=output_dir,
        )
        print(f"Saved to: {output_dir}")
    except Exception as e:
        print(f"ERROR: {e}")
        with open(os.path.join(output_dir, 'error.txt'), 'w') as f:
            f.write(str(e))

print(f"\n{'='*60}")
print(f"Grid search complete! Results in: {output_base}")
print(f"{'='*60}")

# Cleanup temp files
print(f"\nCleaning up temporary trimmed files...")
shutil.rmtree(temp_dir)
print("Done.")