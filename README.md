# Audio Isolation Pipeline

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