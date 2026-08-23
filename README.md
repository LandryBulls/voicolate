# voicolate

Voice isolation for conversation studies with multiple microphones: one close-talk,
face-worn cardioid per participant, recorded in close quarters and sample-aligned.

## Which path to use

`isolate_v3` (v3) is the current path. `isolate` (v2) is kept unchanged so existing
`_isolated_v2.wav` files remain reproducible; it should not be used for new work.

```bash
python scripts/isolate_session.py /path/to/session          # one session
python scripts/isolate_session.py /path/to/sessions --all   # every session under a dir
python scripts/isolate_session.py /path/to/session --force  # reprocess
```

Each run writes, into the session's `processed/`:

| file | contents |
|---|---|
| `TRACK0N_trimmed_isolated_v3.wav` | isolated audio |
| `TRACK0N_trimmed_speaking_v3.npz` | per-frame speech probability (~86 Hz) + rate metadata |
| `{session}_isolation_params.json` | the exact config and the voicolate git SHA |
| `{session}_isolation_qc.json` | measured speech retention and bleed rejection |

```python
from voicolate import IsolationConfig, isolate_v3
result = isolate_v3(['TRACK01_trimmed.wav', 'TRACK02_trimmed.wav'],
                    config=IsolationConfig())
result['audio']               # one isolated array per microphone
result['speech_probability']  # the multichannel VAD that drove the suppression
```

## How it works

The physical fact the method rests on: during real speech, a talker's own microphone
leads every other microphone by a wide margin -- median 27 dB, measured. Deciding
who owns each time-frequency bin is therefore mostly easy, and the whole problem is
in handling the frames where it is not.

1. **Calibration.** Per-track gains that put the microphones on a common footing,
   from the 95th-percentile frame level (`calibration='speech_level'`, the default)
   or by symmetrising the measured bleed matrix (`'bleed_symmetry'`). Both are
   invariant to how much each person talked.
2. **Time-frequency mask.** A soft, floored sigmoid on cross-microphone dominance.
3. **Speech presence.** A per-frame probability combining dominance across the array
   with level above *that track's own noise floor*, with a symmetric hangover.
4. **Suppression.** Bounded by `residual_floor_db`, so nothing is ever destroyed.

Processing is blocked, so peak memory does not depend on session length or on the
number of microphones; calibration and speech presence are decided globally
beforehand on cheap frame envelopes, so blocking cannot change the result.

## Why v3 replaces v2

v2 suppressed residual bleed with `soft_gate`, which measured level against the
**file's global peak** and expanded below a threshold with **no floor**.

* The peak of a 30-minute track is set by its worst transient. On `2024-05-22_000`
  the peak sits 17.7-19.0 dB above the actual speech level, so a "-40 dB below peak"
  threshold lands only ~22 dB under speech and removes the bottom half of the speech
  dynamic range. A session with a louder bump gets a proportionally more aggressive
  gate, which is why the damage varied session to session.
* `gain_db = (rms_db - threshold_db) * (ratio - 1)` is unbounded, so quiet frames
  collapsed to digital silence. Across 82 sessions the median track has **70% of its
  samples at exactly zero**, with silent runs up to 219 s.

Measured on the same harness -- fraction of *clear* speech frames (target dominant by
>6 dB and >12 dB above its own noise floor) attenuated by more than 12 dB:

| session | v2 | v3 |
|---|---|---|
| `2024-05-22_000` (2 mics) | 17.7% | **0.0%** |
| `2024-09-27_000` (4 mics) | 16.3% | **0.0%** |

with no loss of bleed rejection, no digital silence over speech, and a 2.5-5x
speedup (185 s -> 37 s for 2 mics; 73 s for 4 mics).

## QC

Every run measures itself, so a bad session raises instead of passing silently:

```
track              spk_s   floor   cut>6  cut>12  cut>20   bleed  zero%  spk_sil
TRACK01_trimmed      402   -67.2   0.003   0.000   0.000   -23.8  26.11     0.00
TRACK02_trimmed     1132   -67.6   0.000   0.000   0.000   -25.0  14.01     0.00
```

`spk_sil` -- seconds of clear speech driven to exact zero -- is the metric that
distinguishes the v2 failure from harmless int16 quantisation of a residual already
100 dB below speech. Thresholds are in `voicolate/qc.py`.

## Residual bleed and ASR

After v3, what survives suppression is quiet but structurally intact, and audible on
close listening. Whether that matters was measured rather than assumed, with WhisperX
large-v2 (which repeats bit-identically on the same audio, so the comparison has no
run-to-run noise). On `2024-05-22_000` TRACK01, of 1486 transcribed words, 10 fall
where that speaker's own VAD says they were silent and 6 match a word TRACK02 said at
the same moment -- 0.4-0.7% either way.

Leakage does not worsen with more microphones. On the four-person `2024-09-27_000`
(6944 words across four tracks) the same two measures give 0.4% and 1.4% -- and
repeating the second test with timestamps shifted by a third of the session still
flags 0.6%, so barely half of those matches exceed chance, and 48% of the flagged
tokens are backchannels ("yeah" x17, "no" x10) that genuinely do overlap between
speakers.

`residual_noise_over_db` fills the suppressed regions with ambience-shaped noise and
removes all 10 of the first kind, but costs ~4.3% of the speaker's own words, at every
level tested (+3, +6, +12 dB -- the cost is a step, not a slope). It is therefore
**off by default**; enable it when a word attributed to the wrong speaker costs more
than a missing one. Note that the 6 text-matched words are not clean bleed: all have
the target's own VAD active and are backchannels during overlap ("nice", "no no no"),
so no audio-side treatment separates them from genuine simultaneous speech.

## Tests

```bash
python -m pytest tests/
```


---

## v2 pipeline (historical)

This document explains the audio isolation pipeline implemented in the `voicolate` package, with particular focus on the adaptive masking process used to separate audio sources.

## Overview

The pipeline consists of three key stages that work together to isolate individual speakers in group conversations:

1. **Wiener Filtering**: Performs initial source separation using statistical properties of the audio signals. This creates a "first pass" separation that helps identify where each person's speech is likely to occur.

2. **Adaptive Masking**: Creates dynamic time-varying masks that adapt to changing conversation dynamics. Using signal-to-interference ratios (SIR), it determines when to pass or block audio based on who is speaking and how strongly their voice is being picked up by each microphone.

3. **Post-processing and Normalization**: Refines the separated audio by smoothing mask transitions, removing artifacts, and ensuring consistent volume levels across all speakers. This makes the final output clean and natural-sounding.

## Practical Example: Group Conversation

Consider a scenario with four people sitting around a table, each wearing a face-worn microphone. Let's call them Alice, Bob, Charlie, and David. When we want to isolate Alice's speech:

1. **Target vs. Interference**
   - Target: Alice's speech captured by her microphone
   - Interference: Bob, Charlie, and David's voices bleeding into Alice's microphone

2. **Adaptive Behavior**
   - When Alice is speaking alone: The threshold remains low because the SIR is high (her voice is much stronger than background noise)
   - When others are speaking: The threshold automatically increases to reject their interference
   - During overlapping speech: The mask becomes more selective, only passing audio when Alice's voice is significantly stronger than the interference

For example, if Bob, Charlie, and David are having a side conversation while Alice is silent:
```python
# Typical values during others' speech
target_rms = 0.05      # Low level pickup of others on Alice's mic
interference_rms = 0.15 # Stronger interference signal
sir_db = -9.5          # Negative SIR indicates interference is stronger
scale = 2.1            # Higher scale due to negative SIR
threshold = 0.12       # Elevated threshold to reject interference
```

This results in the mask being "closed" (values near 0), effectively silencing the audio during this period. When Alice starts speaking:
```python
# Values when Alice speaks
target_rms = 0.8       # Strong direct speech
interference_rms = 0.15 # Same interference level
sir_db = 14.5          # Positive SIR indicates target is stronger
scale = 0.7            # Lower scale due to positive SIR
threshold = 0.08       # Relaxed threshold allows speech through
```

The mask now "opens" (values near 1) to let Alice's speech pass through while still rejecting interference.

## Detailed Process Flow

### 1. Wiener Filtering

The process begins with Wiener filtering, which provides an initial separation of audio sources. The implementation:
- Uses the `nussl` library's WienerFilter implementation
- Processes audio in batches to manage memory efficiently
- Caches results to avoid recomputation
- Serves as a foundation for the more sophisticated adaptive masking stage

### 2. Adaptive Masking Process

The adaptive masking stage is the core of the isolation process. It uses a sophisticated approach to create time-varying masks that separate audio sources while accounting for varying signal-to-interference ratios (SIR).

#### Key Components of Adaptive Masking

1. **RMS Energy Calculation**
   - Calculates root mean square (RMS) energy using an optimized rolling window approach
   - Uses cumulative sum for efficient computation
   - Window size is configurable (default 10ms)

2. **Threshold Determination**
   - Base threshold derived from target audio's RMS distribution
   - Interference threshold from 25th percentile of interference RMS
   - Minimum threshold ensures noise floor consideration
   
3. **Signal-to-Interference Ratio (SIR) Analysis**
   - Calculates instantaneous SIR in decibels
   - Bounded between -20dB and +20dB
   - Used to adapt masking threshold dynamically

4. **Adaptive Scaling**
   - Scale factor varies based on SIR
   - Exponential scaling: \( scale = e^{-SIR * adaptive\_scale} \)
   - Bounded between 0.5 and 3.0 for stability
   - Higher interference leads to stricter masking

5. **Mask Generation**
   ```python
   primary_mask = (target_rms >= adaptive_threshold)
   interference_mask = (target_rms >= interference_rms * 0.8)
   mask = primary_mask * interference_mask
   ```

6. **Mask Refinement**
   - Minimum duration constraint removes brief activations
   - Binary dilation widens mask regions
   - Gaussian smoothing prevents artifacts
   - Downsampling/upsampling for efficient processing

### 3. Post-processing

The final stage includes:
- Optional normalization (LUFS, RMS, or peak-based)
- Audio reconstruction using the refined masks
- Output file generation with configurable formats

## Configuration

The process is highly configurable through the `MaskConfig` class, which allows tuning of:
- Window sizes for RMS calculation
- Smoothing parameters
- Threshold levels
- Minimum duration constraints
- Speech-specific parameters for voice isolation

## Performance Considerations

The pipeline incorporates several optimizations:
- Efficient RMS calculation using cumulative sum
- Mask smoothing with downsampling for speed
- Caching of Wiener filter results
- Batch processing for memory efficiency

## Usage Example

```python
from voicolate import isolate_audio, MaskConfig

# Configure isolation for face-worn microphone setup
config = MaskConfig(
    # RMS calculation window (100ms captures speech dynamics well)
    window_ms=100,
    
    # Smoothing for mask transitions (100ms prevents choppy speech)
    base_sigma=100,
    
    # Widen speech regions slightly (200ms helps with word boundaries)
    base_widen_ms=200,
    
    # Conservative adaptive scaling for clear speech separation
    adaptive_scale=0.2,
    
    # Minimum valid speech duration (100ms removes brief artifacts)
    min_duration_ms=100,
    
    # Speech-specific frequency band for better voice isolation
    speech_band_hz=(550, 2205)
)

# Process a group conversation (one file per microphone)
isolated_voices = isolate_audio(
    file_list=[
        'alice_mic.wav',   # Alice's face-worn microphone
        'bob_mic.wav',     # Bob's face-worn microphone
        'charlie_mic.wav', # Charlie's face-worn microphone
        'david_mic.wav'    # David's face-worn microphone
    ],
    config=config,
    normalize_output=True,  # Ensure consistent volume across speakers
    normalization_params={
        'method': 'lufs',  # Industry standard loudness normalization
        'target_level': -23  # Standard broadcast loudness level
    }
)

# isolated_voices now contains separated audio for each speaker
# Each array in the list corresponds to one speaker's cleaned audio
alice_clean = isolated_voices[0]
bob_clean = isolated_voices[1]
charlie_clean = isolated_voices[2]
david_clean = isolated_voices[3]
```

## Mathematical Background

The adaptive threshold at time t is calculated as:

```math
threshold(t) = \max(base_{threshold} \cdot e^{-SIR(t) \cdot scale}, min_{threshold})
```

where SIR(t) is the Signal-to-Interference Ratio in decibels:

```math
SIR(t) = 20 \log_{10}\left(\frac{RMS_{target}(t)}{RMS_{interference}(t)}\right)
```

This adaptive thresholding ensures that:
- When SIR is high (strong target signal), the threshold decreases exponentially
- When SIR is low (strong interference), the threshold increases exponentially
- The minimum threshold prevents over-masking during quiet periods

The exponential relationship creates a smooth, natural-sounding transition between states while maintaining effective separation quality. 