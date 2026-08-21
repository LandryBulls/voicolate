import numpy as np
from scipy.io import wavfile
from scipy.ndimage import (gaussian_filter, binary_dilation, percentile_filter,
                           uniform_filter, maximum_filter1d)
import os
from tqdm import tqdm
from scipy import signal
from scipy.signal import stft, istft, welch
from dataclasses import replace

from .config import IsolationConfig
from .calibrate import (frame_levels, noise_floor, speech_level,
                        calibration_gains, bleed_matrix)

# Lazy import for nussl (has scipy version compatibility issues)
_nussl = None
def _get_nussl():
    global _nussl
    if _nussl is None:
        import nussl
        _nussl = nussl
    return _nussl

def arr_to_batch(array, batch_size):
    shape = array.shape[1]
    n_batches = shape // batch_size
    leftover = shape - batch_size*n_batches
    batches = []
    b = 0
    for batch in range(n_batches):
        batches.append(array[:,b:b+batch_size])
        b+=batch_size
    if leftover:
        batches.append(array[:,b:b+leftover])
    return batches

def additive_mix(audio_iter):
    """
    Takes an iterable of audio files and returns their sum.
    Handles arrays of different lengths by padding shorter arrays with zeros.
    """
    # Find the maximum length
    max_len = max(len(audio) for audio in audio_iter)
    
    # Pad all arrays to the same length and sum
    mix = np.zeros(max_len)
    for audio in audio_iter:
        if len(audio) < max_len:
            padded_audio = np.pad(audio, (0, max_len - len(audio)), 'constant')
            mix += padded_audio
        else:
            mix += audio
    return mix

def normalize_audio_tracks(audio_list, method='rms', target_level=-23):
    """
    Normalize a list of audio tracks to have similar levels using various methods.
    
    Parameters:
        audio_list: List of numpy arrays containing audio data
        method: Normalization method ('peak', 'rms', or 'lufs')
        target_level: Target level for normalization
                     For peak: value between 0 and 1
                     For RMS: value in dB (e.g., -20)
                     For LUFS: value in LUFS (typically -23 for broadcast)
    
    Returns:
        List of normalized audio tracks
    """
    normalized_tracks = []
    
    if method == 'peak':
        # Peak normalization
        for audio in audio_list:
            peak = np.max(np.abs(audio))
            normalized_tracks.append(audio * (target_level / peak))
            
    elif method == 'rms':
        # RMS normalization
        target_rms = 10 ** (target_level / 20)  # Convert dB to linear
        
        for audio in audio_list:
            # Calculate current RMS
            current_rms = np.sqrt(np.mean(audio ** 2))
            # Calculate scaling factor
            scaling_factor = target_rms / (current_rms + 1e-8)  # Add small value to prevent division by zero
            normalized_tracks.append(audio * scaling_factor)
            
    elif method == 'lufs':
        # Simplified LUFS-like normalization using sliding window RMS
        window_size = 4410  # 100ms at 44.1kHz
        target_lufs_linear = 10 ** (target_level / 20)
        
        for audio in audio_list:
            # Calculate running RMS
            running_rms = np.sqrt(np.convolve(audio**2, 
                                            np.ones(window_size)/window_size, 
                                            mode='valid'))
            # Get the 95th percentile as a pseudo-loudness measure
            pseudo_loudness = np.percentile(running_rms, 95)
            # Calculate scaling factor
            scaling_factor = target_lufs_linear / (pseudo_loudness + 1e-8)
            normalized_tracks.append(audio * scaling_factor)
    
    else:
        raise ValueError("Method must be 'peak', 'rms', or 'lufs'")
    
    return normalized_tracks

def apply_wiener(file_list, iterations=10, save_to_file=False, output_path=None, return_outputs=True,
                 batch_size=441000, cache_dir=None, overwrite=False):
    """
    Takes list of .wav files and returns filtered audio.
    Assumes all audio files are mono and of the *exact* same length.
    
    Parameters:
        cache_dir: Directory to store/load cached Wiener-filtered files. If None, uses 'wiener_cache' in current directory.
        overwrite: If True, recompute and overwrite existing cached Wiener files. If False, use cached files when available.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), 'wiener_cache')
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Check if all Wiener-filtered files exist in cache
    cached_files = []
    all_cached = True
    for file in file_list:
        cache_file = os.path.join(cache_dir, os.path.basename(file)[:-4] + '_wiener.wav')
        cached_files.append(cache_file)
        if not os.path.exists(cache_file):
            all_cached = False
    
    # If all files are cached and we're not overwriting, load them and return
    if all_cached and not overwrite:
        print("Loading cached Wiener-filtered files...\n")
        outs = []
        for cache_file in cached_files:
            rate, data = wavfile.read(cache_file)
            outs.append(data)
        outs = np.array(outs)
        if return_outputs:
            return outs
        return None
    
    # If not cached or overwriting, proceed with Wiener filtering
    nussl = _get_nussl()
    naud = len(file_list)
    estimates = [nussl.AudioSignal(i) for i in file_list]
    rate = estimates[0].sample_rate

    if not all([estimates[i].audio_data.shape for i in range(naud)]):
        raise Exception("Audio files are of different lengths!")
    
    # Extract raw audio data without normalization
    audio_data = [est.audio_data[0] for est in estimates]
    
    # Update the estimates with raw audio
    for i, est in enumerate(estimates):
        est.audio_data = audio_data[i][np.newaxis, :]

    batches = arr_to_batch(np.array([estimates[i].audio_data[0] for i in range(naud)]), batch_size=batch_size)

    outs = []

    print("Applying Wiener...\n")
    for batch in tqdm(batches):
        mix = additive_mix(batch)
        mix = nussl.core.AudioSignal(audio_data_array=mix, sample_rate=rate)
        wiener = nussl.separation.benchmark.WienerFilter(mix,
                                                         [nussl.core.AudioSignal(audio_data_array=i, sample_rate=rate)
                                                          for i in batch], iterations=iterations)
        wout = wiener()
        outs.append(np.array([i.audio_data[0] for i in wout]))

    outs = np.concatenate(outs, axis=1)

    # Always save to cache (this will overwrite existing cache files if overwrite=True)
    for f, file in enumerate(file_list):
        cache_file = os.path.join(cache_dir, os.path.basename(file)[:-4] + '_wiener.wav')
        wavfile.write(cache_file, estimates[0].sample_rate, outs[f])

    # Save to output path if requested
    if save_to_file:
        if output_path == None:
            output_path = os.getcwd()
        for f, file in enumerate(file_list):
            out = os.path.join(output_path, os.path.basename(file)[:-4] + '_wiener.wav')
            wavfile.write(out, estimates[0].sample_rate, outs[f])
            
    if return_outputs:
        return outs
    return None

def window_rms(a, rate=44100, window_ms=10):
    """
    Optimized rolling-window RMS calculation using cumsum, maintaining input length
    """
    window_size = int(round((rate/1000)*window_ms))
    # Pad the input array
    pad_width = window_size - 1
    padded = np.pad(np.power(a,2), (pad_width//2, pad_width//2), mode='edge')
    # Use cumsum for faster rolling window
    cumsum = np.cumsum(padded)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    rms = np.sqrt(cumsum[window_size-1:] / window_size)
    # Ensure output length matches input length by padding
    if len(rms) < len(a):
        rms = np.pad(rms, (0, len(a) - len(rms)), mode='edge')
    elif len(rms) > len(a):
        rms = rms[:len(a)]
    return rms


def get_band_rms(audio, band_hz, rate=44100, window_ms=10):
    """
    Calculate RMS energy in a specific frequency band
    
    Parameters:
        audio: Input audio signal
        band_hz: Tuple of (low_freq, high_freq) in Hz
        rate: Sample rate in Hz
        window_ms: Window size for RMS calculation in milliseconds
    """
    nyquist = rate / 2
    low_freq, high_freq = band_hz
    
    # Design bandpass filter
    b, a = signal.butter(4, [min(low_freq/nyquist, 0.99), min(high_freq/nyquist, 0.99)], btype='band')
    
    # Apply filter
    filtered = signal.filtfilt(b, a, audio)
    
    # Calculate RMS
    return window_rms(filtered, rate=rate, window_ms=window_ms)

def smooth_mask(mask, sigma, downsample_factor, rate):
    """
    Smooth a binary mask by downsampling, applying Gaussian filter, and upsampling.
    
    Parameters:
        mask: Binary mask array
        sigma: Standard deviation for Gaussian filter
        downsample_factor: Number of samples to combine for downsampling
        rate: Original sample rate
    
    Returns:
        Smoothed mask at original resolution
    """
    mask_len = len(mask)
    downsampled_len = mask_len // downsample_factor
    
    # Downsample the mask (taking max value in each window)
    downsampled_mask = np.array([np.max(mask[i:i+downsample_factor]) 
                                for i in range(0, mask_len, downsample_factor)])
    
    # Apply Gaussian filter to downsampled mask
    downsampled_sigma = sigma / downsample_factor  # Scale sigma for downsampled signal
    smoothed_downsampled = gaussian_filter(downsampled_mask.astype(float), 
                                         sigma=downsampled_sigma, 
                                         mode='reflect')
    
    # Upsample back to original resolution using cubic interpolation
    time_points = np.linspace(0, mask_len, len(smoothed_downsampled))
    full_time = np.arange(mask_len)
    smoothed_mask = np.interp(full_time, time_points, smoothed_downsampled)
    
    # Rescale the smoothed mask to [0,1] range
    smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min() + 1e-10)
    
    return smoothed_mask


def compute_spectral_dominance_mask(all_audio, target_idx, rate=44100, nperseg=2048, 
                                    noverlap=1536, dominance_margin_db=3.0,
                                    harmonicity_weight=0.5):
    """
    Compute time-frequency mask based on cross-microphone dominance.
    
    Exploits two properties:
    1. Perfect alignment - compare energy at exact same time-frequency bin across mics
    2. Cardioid attenuation - target speaker is always loudest in their own mic
    
    Parameters:
        all_audio: List/array of audio from ALL microphones (not just interference)
        target_idx: Index of the target microphone in all_audio
        rate: Sample rate
        nperseg: STFT window size (larger = better frequency resolution)
        noverlap: STFT overlap
        dominance_margin_db: How much louder target must be to be considered dominant (dB)
        harmonicity_weight: Weight for harmonicity in final mask (0-1)
    
    Returns:
        mask: Time-frequency mask
        time_mask: Collapsed time-domain mask
    """
    n_mics = len(all_audio)
    
    # Compute STFTs for all microphones
    specs = []
    for audio in all_audio:
        f, t, spec = stft(audio, fs=rate, nperseg=nperseg, noverlap=noverlap)
        specs.append(spec)
    
    # Stack magnitudes: shape (n_mics, n_freqs, n_times)
    mags = np.array([np.abs(spec) for spec in specs])
    
    target_mag = mags[target_idx]
    
    # Find the maximum magnitude across all OTHER microphones at each T-F bin
    other_indices = [i for i in range(n_mics) if i != target_idx]
    other_mags = mags[other_indices]
    max_other_mag = np.max(other_mags, axis=0)
    
    # Dominance ratio: how much louder is target mic vs loudest other mic?
    eps = 1e-10
    dominance_db = 20 * np.log10((target_mag + eps) / (max_other_mag + eps))
    
    # Soft mask based on dominance
    # When dominance_db > margin: mask ≈ 1.0  (target is clearly dominant)
    # When dominance_db < -margin: mask ≈ 0.0 (other mic is dominant)
    # In between: smooth transition
    margin = dominance_margin_db
    dominance_mask = np.clip((dominance_db + margin) / (2 * margin), 0, 1)
    
    # Harmonicity check: speech has harmonic structure, breath/friction doesn't
    # Compute spectral flatness per frame (low = harmonic, high = noise-like)
    geometric_mean = np.exp(np.mean(np.log(target_mag + eps), axis=0))
    arithmetic_mean = np.mean(target_mag, axis=0)
    spectral_flatness = geometric_mean / (arithmetic_mean + eps)
    
    # Convert to harmonicity score (inverse of flatness)
    harmonicity = 1 - spectral_flatness
    harmonicity = np.clip(harmonicity, 0, 1)
    
    # Expand harmonicity to full T-F shape (same value across frequencies)
    harmonicity_mask = np.broadcast_to(harmonicity, target_mag.shape)
    
    # Combine: must be both dominant AND harmonic (for speech)
    # This handles breath breakthrough - breath is dominant in target mic but not harmonic
    combined_mask = dominance_mask * (1 - harmonicity_weight + harmonicity_weight * harmonicity_mask)
    
    # Collapse to time-domain mask (energy-weighted mean across frequency)
    freq_weights = target_mag / (np.sum(target_mag, axis=0, keepdims=True) + eps)
    time_mask = np.sum(combined_mask * freq_weights, axis=0)
    
    return combined_mask, time_mask, (f, t, specs[target_idx])


def apply_spectral_mask(audio, mask, rate=44100, nperseg=2048, noverlap=1536):
    """
    Apply a time-frequency mask to audio and reconstruct.
    
    Parameters:
        audio: Input audio signal
        mask: Time-frequency mask (same shape as STFT output)
        rate: Sample rate
        nperseg: STFT window size (must match mask computation)
        noverlap: STFT overlap (must match mask computation)
    
    Returns:
        Reconstructed audio with mask applied
    """
    f, t, spec = stft(audio, fs=rate, nperseg=nperseg, noverlap=noverlap)
    
    # Handle shape mismatch
    min_f = min(mask.shape[0], spec.shape[0])
    min_t = min(mask.shape[1], spec.shape[1])
    
    masked_spec = spec.copy()
    masked_spec[:min_f, :min_t] = spec[:min_f, :min_t] * mask[:min_f, :min_t]
    
    _, reconstructed = istft(masked_spec, fs=rate, nperseg=nperseg, noverlap=noverlap)
    
    return reconstructed


def soft_gate(audio, rate=44100, threshold_db=-75, ratio=4.0, attack_ms=5, release_ms=50,
              window_ms=20, knee_db=6, lookahead_ms=30, hold_ms=50):
    """
    Apply a soft noise gate/expander to attenuate quiet sections.
    
    Unlike a hard gate, this uses a smooth expansion curve that gradually
    attenuates signals below the threshold, avoiding clicks and artifacts.
    
    Parameters:
        audio: Input audio signal
        rate: Sample rate
        threshold_db: Level below which expansion starts (dB, relative to peak)
        ratio: Expansion ratio (4.0 means 4dB reduction for every 1dB below threshold)
        attack_ms: Attack time in milliseconds (how fast gate opens)
        release_ms: Release time in milliseconds (how fast gate closes)
        window_ms: RMS window for level detection
        knee_db: Soft knee width in dB (0 = hard knee)
        lookahead_ms: Open gate this many ms BEFORE speech detected (preserves onsets)
        hold_ms: Keep gate open this many ms AFTER speech ends (preserves offsets)
    
    Returns:
        Gated audio signal
    """
    # Calculate RMS envelope
    rms = window_rms(audio, rate=rate, window_ms=window_ms)
    
    # Convert to dB (relative to signal peak)
    peak = np.max(np.abs(audio)) + 1e-10
    rms_db = 20 * np.log10(rms / peak + 1e-10)
    
    # Calculate gain reduction using soft knee expansion
    # Below threshold: gain decreases as signal gets quieter
    gain_db = np.zeros_like(rms_db)
    
    if knee_db > 0:
        # Soft knee - gradual transition
        knee_start = threshold_db - knee_db / 2
        knee_end = threshold_db + knee_db / 2
        
        # Below knee: full expansion
        below_knee = rms_db < knee_start
        gain_db[below_knee] = (rms_db[below_knee] - threshold_db) * (ratio - 1)
        
        # In knee: gradual transition
        in_knee = (rms_db >= knee_start) & (rms_db < knee_end)
        knee_factor = (rms_db[in_knee] - knee_start) / knee_db
        expansion = (rms_db[in_knee] - threshold_db) * (ratio - 1)
        gain_db[in_knee] = expansion * (1 - knee_factor)
        
        # Above knee: no gain change (gain_db remains 0)
    else:
        # Hard knee
        below_thresh = rms_db < threshold_db
        gain_db[below_thresh] = (rms_db[below_thresh] - threshold_db) * (ratio - 1)
    
    # Convert gain to linear
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply hold (dilation) - keep gate open after signal drops
    # This is like binary dilation: expand the "open" regions
    hold_samples = int(hold_ms * rate / 1000)
    if hold_samples > 1:
        # Use maximum filter to dilate high-gain regions forward in time
        from scipy.ndimage import maximum_filter1d
        gain_linear = maximum_filter1d(gain_linear, size=hold_samples, mode='nearest', origin=-(hold_samples//2))
    
    # Apply lookahead - shift gain envelope backwards so gate opens before speech
    lookahead_samples = int(lookahead_ms * rate / 1000)
    if lookahead_samples > 0:
        # Shift the gain curve backwards (so gate opens earlier)
        gain_linear = np.roll(gain_linear, -lookahead_samples)
        # Fill the end with the last valid value
        gain_linear[-lookahead_samples:] = gain_linear[-lookahead_samples-1]
    
    # Smooth the gain envelope with attack/release
    attack_samples = int(attack_ms * rate / 1000)
    release_samples = int(release_ms * rate / 1000)
    
    # Apply attack/release smoothing
    smoothed_gain = np.zeros_like(gain_linear)
    smoothed_gain[0] = gain_linear[0]
    
    attack_coeff = 1 - np.exp(-2.2 / attack_samples) if attack_samples > 0 else 1.0
    release_coeff = 1 - np.exp(-2.2 / release_samples) if release_samples > 0 else 1.0
    
    for i in range(1, len(gain_linear)):
        if gain_linear[i] > smoothed_gain[i-1]:
            # Attack (gate opening)
            smoothed_gain[i] = smoothed_gain[i-1] + attack_coeff * (gain_linear[i] - smoothed_gain[i-1])
        else:
            # Release (gate closing)
            smoothed_gain[i] = smoothed_gain[i-1] + release_coeff * (gain_linear[i] - smoothed_gain[i-1])
    
    # Ensure gain matches audio length
    if len(smoothed_gain) != len(audio):
        smoothed_gain = np.interp(
            np.linspace(0, 1, len(audio)),
            np.linspace(0, 1, len(smoothed_gain)),
            smoothed_gain
        )
    
    return audio * smoothed_gain


def cross_mic_spectral_filter(all_audio, rate=44100, nperseg=2048, noverlap=1536,
                              dominance_margin_db=3.0, harmonicity_weight=0.5,
                              smoothing_time_frames=3, smoothing_freq_bins=5,
                              apply_gate=True, gate_threshold_db=-75, gate_ratio=4.0,
                              gate_attack_ms=5, gate_release_ms=50,
                              gate_lookahead_ms=50, gate_hold_ms=75):
    """
    Main spectral filtering function using cross-microphone dominance.
    
    This is the recommended approach for close-proximity cardioid mic arrays.
    
    Parameters:
        all_audio: List of audio arrays, one per microphone (must be aligned!)
        rate: Sample rate
        nperseg: STFT window size
        noverlap: STFT overlap
        dominance_margin_db: Required dB advantage for dominance (lower = more aggressive)
        harmonicity_weight: How much to weight harmonicity (0 = ignore, 1 = fully weight)
        smoothing_time_frames: Temporal smoothing of mask
        smoothing_freq_bins: Frequency smoothing of mask
        apply_gate: Whether to apply soft noise gate to remove residual bleedthrough
        gate_threshold_db: Gate threshold in dB below peak (lower = more aggressive)
        gate_ratio: Expansion ratio (higher = more attenuation below threshold)
        gate_attack_ms: Gate attack time in ms
        gate_release_ms: Gate release time in ms
        gate_lookahead_ms: Open gate this many ms before speech (preserves onsets)
        gate_hold_ms: Keep gate open this many ms after speech (preserves offsets, like dilation)
    
    Returns:
        List of filtered audio, one per microphone
    """
    n_mics = len(all_audio)
    filtered_audio = []
    
    # Convert to numpy array for easier indexing
    all_audio = np.array([np.asarray(a) for a in all_audio])
    
    print("Computing cross-microphone spectral dominance masks...\n")
    
    for target_idx in tqdm(range(n_mics), desc="Filtering"):
        # Compute dominance mask for this microphone
        tf_mask, time_mask, (f, t, target_spec) = compute_spectral_dominance_mask(
            all_audio, target_idx, rate=rate, nperseg=nperseg, noverlap=noverlap,
            dominance_margin_db=dominance_margin_db, harmonicity_weight=harmonicity_weight
        )
        
        # Smooth the mask (reduce musical noise artifacts)
        if smoothing_time_frames > 1 or smoothing_freq_bins > 1:
            tf_mask = uniform_filter(tf_mask.astype(np.float64), 
                                    size=(smoothing_freq_bins, smoothing_time_frames),
                                    mode='nearest')
        
        # Apply mask
        filtered = apply_spectral_mask(all_audio[target_idx], tf_mask, 
                                       rate=rate, nperseg=nperseg, noverlap=noverlap)
        
        # Match original length
        if len(filtered) < len(all_audio[target_idx]):
            filtered = np.pad(filtered, (0, len(all_audio[target_idx]) - len(filtered)))
        else:
            filtered = filtered[:len(all_audio[target_idx])]
        
        # Apply soft gate to remove residual bleedthrough in quiet sections
        if apply_gate:
            filtered = soft_gate(filtered, rate=rate, threshold_db=gate_threshold_db,
                                ratio=gate_ratio, attack_ms=gate_attack_ms, 
                                release_ms=gate_release_ms, lookahead_ms=gate_lookahead_ms,
                                hold_ms=gate_hold_ms)
        
        filtered_audio.append(filtered)
    
    return filtered_audio


class MaskConfig:
    def __init__(self, rate=44100, window_ms=100, iterations=20, smoothing_sigma=100, 
                 attack_release_ms=200, min_duration_ms=100, threshold_scale=0.6,
                 lookahead_ms=50, gate_range_db=-20, adaptive_window_sec=3.0,
                 interference_threshold_db=-6):
        # Basic audio parameters
        self.rate = rate                    # Sample rate of audio (Hz)
        self.window_ms = window_ms          # Window size for RMS calculation (milliseconds)

        # Wiener filter parameters
        self.iterations = iterations
        
        # Gate parameters
        self.smoothing_sigma = smoothing_sigma        # Gaussian smoothing width for the mask
        self.attack_release_ms = attack_release_ms    # Attack/release time for gate (milliseconds)
        self.threshold_scale = threshold_scale        # Scale factor for adaptive threshold (0-1)
        self.lookahead_ms = lookahead_ms              # Lookahead time to anticipate speech (milliseconds)
        self.gate_range_db = gate_range_db            # Maximum attenuation when gate is closed (dB)
        self.adaptive_window_sec = adaptive_window_sec  # Window for calculating adaptive threshold (seconds)
        self.interference_threshold_db = interference_threshold_db  # SIR threshold for interference reduction (dB)
        
        # Noise removal parameters
        self.min_duration_ms = min_duration_ms  # Minimum duration of valid segments (milliseconds)
        
        # Pre-calculate commonly used values (in samples)
        self.attack_release_samples = int((attack_release_ms / 1000) * rate)
        self.lookahead_samples = int((lookahead_ms / 1000) * rate)
        self.min_duration_samples = int((min_duration_ms / 1000) * rate)
        self.adaptive_window_samples = int(adaptive_window_sec * rate)
        self.downsample_factor = int(rate * 0.1)  # 100ms worth of samples for mask smoothing
        self.gate_range_linear = 10 ** (gate_range_db / 20)  # Convert dB to linear scale

def adaptive_mask(target_audio, interference_audio, config):
    """
    Gentle gate-like mask with adaptive threshold based on target signal statistics.
    Uses proportional gain reduction instead of complete silencing.
    """
    # Calculate RMS values
    target_rms = window_rms(target_audio, rate=config.rate, window_ms=config.window_ms)
    interference_mix = additive_mix(interference_audio)
    interference_rms = window_rms(interference_mix, rate=config.rate, window_ms=config.window_ms)
    
    # Handle length mismatch
    if len(target_rms) != len(interference_rms):
        min_len = min(len(target_rms), len(interference_rms))
        target_rms = target_rms[:min_len]
        interference_rms = interference_rms[:min_len]
    
    # Calculate adaptive threshold based on target signal's running median
    # Use pandas rolling median for efficiency (much faster than loop)
    window_samples = config.adaptive_window_samples
    
    # Use uniform filter as it's fast and robust, then scale appropriately
    # This gives us a local average which is a good proxy for adaptive thresholding
    adaptive_threshold = uniform_filter(target_rms.astype(np.float64), 
                                       size=window_samples, mode='nearest')
    
    adaptive_threshold *= config.threshold_scale
    
    # Ensure minimum threshold to avoid division issues
    adaptive_threshold = np.maximum(adaptive_threshold, 1e-8)
    
    # Create soft gate: proportional reduction based on how far below threshold
    # When target_rms >= threshold: gain = 1.0
    # When target_rms < threshold: gain scales from gate_range_linear to 1.0
    ratio = target_rms / adaptive_threshold
    gate_gain = np.clip(ratio, config.gate_range_linear, 1.0)
    
    # Additional gentle reduction when interference is much louder
    sir = 20 * np.log10((target_rms + 1e-8) / (interference_rms + 1e-8))
    interference_reduction = np.ones_like(sir)
    
    # Only reduce when SIR is below threshold (interference >> target)
    below_threshold = sir < config.interference_threshold_db
    interference_reduction[below_threshold] = np.clip(
        (sir[below_threshold] - config.interference_threshold_db) / 10.0 + 1.0,
        0.5,  # Never reduce more than 50% from interference alone
        1.0
    )
    
    # Combine gate gain with interference reduction
    mask = gate_gain * interference_reduction
    
    # Remove very short segments
    if config.min_duration_samples > 1:
        # Convert to binary for morphological operations
        binary_mask = (mask > 0.5).astype(float)
        structure = np.ones(config.min_duration_samples)
        eroded = binary_dilation(1 - binary_mask, structure=structure)
        mask *= (1 - eroded)
    
    # Apply attack/release (widening)
    if config.attack_release_samples > 1:
        binary_mask = (mask > 0.3).astype(float)  # Lower threshold for attack/release
        structure = np.ones(config.attack_release_samples)
        widened = binary_dilation(binary_mask, structure=structure)
        # Use widened mask to ensure gate opens/closes smoothly
        mask = np.maximum(mask, widened * 0.3)
    
    # Apply lookahead by shifting mask forward
    if config.lookahead_samples > 0:
        mask = np.roll(mask, config.lookahead_samples)
        # Fill the beginning with zeros to avoid wraparound
        mask[:config.lookahead_samples] = 0
    
    # Smooth the mask
    smoothed_mask = smooth_mask(mask, config.smoothing_sigma, 
                              config.downsample_factor, config.rate)
    
    return smoothed_mask

def mask_audio(wiener_outputs, raw_audio, config):
    """
    Optimized version of audio masking without noise addition
    """
    masked_audio = []
    
    for i, (wout, raw) in enumerate(zip(wiener_outputs, raw_audio)):
        try:
            # Get interference tracks efficiently
            interference = np.delete(raw_audio, i, axis=0)
            
            # Create and apply mask
            mask_values = adaptive_mask(wout, interference, config)
            
            # Handle length mismatch between mask and raw audio
            if len(mask_values) != len(raw):
                min_len = min(len(mask_values), len(raw))
                # Truncate both to the minimum length
                mask_values = mask_values[:min_len]
                raw = raw[:min_len]
            
            # Simple masking operation
            masked = raw * mask_values
            masked_audio.append(masked)
        except Exception as e:
            print(f"\nError masking audio track {i}: {e}")
            import traceback
            traceback.print_exc()
            raise

    return masked_audio

def save_isolated_audio(array_list, rate=44100, output_path = None, output_name=None):
    if not output_name:
        outnames = [str(i)+'_isolated.wav' for i in range(len(array_list))]
    else:
        outnames = [output_name+'_'+str(i)+'.wav' for i in range(len(array_list))]
    if not output_path:
        output_path = os.getcwd()

    filenames = [os.path.join(output_path, outnames[i]) for i in range(len(array_list))]

    for f, array in enumerate(array_list):
        wavfile.write(filenames[f], rate, array)

    return filenames

def isolate(file_list, dominance_margin_db=1.0, harmonicity_weight=0.5,
                     nperseg=2048, noverlap=1536, smoothing_time_frames=3, 
                     smoothing_freq_bins=5, normalize_input=True, 
                     save_files=False, output_path=None, peak_limit_db=-1.0,
                     apply_gate=True, gate_threshold_db=-75, gate_ratio=4.0,
                     gate_attack_ms=5, gate_release_ms=50,
                     gate_lookahead_ms=50, gate_hold_ms=75):
    """
    Isolate audio using cross-microphone spectral dominance.
    
    This approach directly exploits:
    1. Perfect temporal alignment across microphones
    2. Cardioid pattern attenuation (target is always loudest in their own mic)
    
    At each time-frequency bin, we ask: "Is this mic the dominant source?"
    Combined with harmonicity check to filter breath/friction sounds.
    
    Parameters:
        file_list: List of aligned audio files (one per microphone)
        dominance_margin_db: Required dB advantage to be considered dominant.
                            Lower = more aggressive filtering (try 1-6 dB)
        harmonicity_weight: How much to penalize non-harmonic sounds (0-1).
                           Higher = more breath/friction suppression
        nperseg: STFT window size (2048 = ~46ms at 44.1kHz, good balance)
        noverlap: STFT overlap (higher = smoother but slower)
        smoothing_time_frames: Temporal smoothing to reduce artifacts
        smoothing_freq_bins: Frequency smoothing to reduce artifacts
        normalize_input: Normalize input levels before processing
        save_files: Whether to save output files
        output_path: Where to save output files
        peak_limit_db: Peak limiting threshold in dB
        apply_gate: Apply soft noise gate to remove residual bleedthrough (default: True)
        gate_threshold_db: Gate threshold in dB below peak (default: -75, lower = more aggressive)
        gate_ratio: Expansion ratio (default: 4.0, higher = more attenuation)
        gate_attack_ms: Gate attack time (default: 5ms)
        gate_release_ms: Gate release time (default: 50ms)
        gate_lookahead_ms: Open gate before speech onset (default: 30ms, preserves attack)
        gate_hold_ms: Keep gate open after speech ends (default: 50ms, like dilation)
    
    Returns:
        List of isolated audio arrays (or file paths if save_files=True)
    """
    # Load audio using scipy (no nussl dependency)
    print("Loading audio files...\n")
    all_audio = []
    rate = None
    for f in file_list:
        r, data = wavfile.read(f)
        if rate is None:
            rate = r
        elif r != rate:
            raise ValueError(f"Sample rate mismatch: {f} has rate {r}, expected {rate}")
        # Convert to float and normalize if needed
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        # Handle stereo by taking first channel
        if len(data.shape) > 1:
            data = data[:, 0]
        all_audio.append(data)
    
    # Normalize if requested
    if normalize_input:
        print("Normalizing input levels...\n")
        all_audio = normalize_audio_tracks(all_audio, method='rms', target_level=-20)
    
    # Apply spectral dominance filtering
    filtered_audio = cross_mic_spectral_filter(
        all_audio, rate=rate, nperseg=nperseg, noverlap=noverlap,
        dominance_margin_db=dominance_margin_db, harmonicity_weight=harmonicity_weight,
        smoothing_time_frames=smoothing_time_frames, smoothing_freq_bins=smoothing_freq_bins,
        apply_gate=apply_gate, gate_threshold_db=gate_threshold_db, gate_ratio=gate_ratio,
        gate_attack_ms=gate_attack_ms, gate_release_ms=gate_release_ms,
        gate_lookahead_ms=gate_lookahead_ms, gate_hold_ms=gate_hold_ms
    )
    
    # Peak limiting
    print("Applying peak limiting...\n")
    peak_limit_linear = 10 ** (peak_limit_db / 20)
    for i in range(len(filtered_audio)):
        peak = np.max(np.abs(filtered_audio[i]))
        if peak > peak_limit_linear:
            filtered_audio[i] = filtered_audio[i] * (peak_limit_linear / peak)
    
    if save_files:
        if output_path is None:
            output_path = os.getcwd()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return save_isolated_audio(filtered_audio, rate, output_path)
    
    return filtered_audio

# =============================================================================
# v3 isolation path
#
# The v2 path above suppresses residual bleed with `soft_gate`, which measures
# level against the file's *global peak* and expands below a threshold with no
# floor. Both properties are load-bearing failures:
#
#   * The peak of a 30-minute track is set by its worst transient. Measured on
#     2024-05-22_000, the peak sits 17.7-19.0 dB above the actual speech level,
#     so a "-40 dB below peak" threshold lands only ~22 dB under speech and
#     removes the bottom half of the speech dynamic range. A session with a
#     louder bump gets a proportionally more aggressive gate, which is why the
#     damage varies session to session rather than uniformly.
#   * `gain_db = (rms_db - threshold_db) * (ratio - 1)` is unbounded, so quiet
#     frames collapse to digital silence. Across the 82 sessions on safescratch
#     the median track has 70% of its samples at exactly zero, with silent runs
#     up to 219 s.
#
# v3 keeps the cross-microphone dominance idea, which is sound -- during real
# speech the target's own mic leads by ~27 dB median -- and replaces the gate
# with a decision that uses the array instead of an absolute level, referenced
# to each track's own noise floor and floored so nothing is ever destroyed.
# =============================================================================


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_tracks(file_list):
    """Load aligned mono tracks as float32 in [-1, 1], zero-padded to one length."""
    rate = None
    tracks = []
    for f in file_list:
        r, data = wavfile.read(f)
        if rate is None:
            rate = r
        elif r != rate:
            raise ValueError(f"Sample rate mismatch: {f} has rate {r}, expected {rate}")
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data[:, 0]
        tracks.append(data)

    n = max(len(t) for t in tracks)
    tracks = [t if len(t) == n else np.pad(t, (0, n - len(t))) for t in tracks]
    return rate, tracks


def _bandpass(audio, rate, band_hz):
    """Zero-phase bandpass, used to keep rumble and hiss out of the level decision."""
    nyq = rate / 2.0
    low = max(band_hz[0] / nyq, 1e-4)
    high = min(band_hz[1] / nyq, 0.99)
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, audio.astype(np.float64))


def max_excluding(mags):
    """For every mic, the loudest of all the *other* mics at each T-F bin.

    Derived from the array-wide largest and second-largest value rather than by
    re-reducing N-1 mics once per target, so the work does not grow with the
    number of microphones. `compute_spectral_dominance_mask` above recomputes
    every mic's STFT for every target, which costs N^2 transforms and roughly
    2.6 GB per full-length spectrum at N=4.

    Returns ``(top1, top2, argmax)``; the answer for target ``i`` is
    ``np.where(argmax == i, top2, top1)``.
    """
    idx = np.argmax(mags, axis=0)
    top1 = np.take_along_axis(mags, idx[None], axis=0)[0]
    # blank the winner in place, take the next largest, then put the winner back
    np.put_along_axis(mags, idx[None], 0.0, axis=0)
    top2 = np.max(mags, axis=0)
    np.put_along_axis(mags, idx[None], top1[None], axis=0)
    return top1, top2, idx


def _harmonicity(target_mag, eps=1e-10):
    """Per-frame 1 - spectral flatness. Off by default; see `IsolationConfig`."""
    gm = np.exp(np.mean(np.log(target_mag + eps), axis=0))
    am = np.mean(target_mag, axis=0)
    return np.clip(1.0 - gm / (am + eps), 0.0, 1.0)


def dominance_mask(target_mag, other_mag, cfg):
    """Soft, floored time-frequency mask from cross-microphone dominance.

    The v2 mask ramps linearly from fully closed to fully open across +/- the
    margin and clips at zero. Real dominance during speech is 20+ dB, so the
    sharpness buys nothing where the answer is obvious and destroys the frames
    where it is genuinely ambiguous. This uses a wider sigmoid and never closes
    a bin completely, which also suppresses musical noise.
    """
    eps = 1e-10
    dominance_db = 20 * np.log10((target_mag + eps) / (other_mag + eps))
    mask = cfg.mask_floor + (1.0 - cfg.mask_floor) * _sigmoid(
        (dominance_db - cfg.dominance_margin_db) / (cfg.dominance_width_db / 4.0))

    if cfg.harmonicity_weight > 0:
        h = _harmonicity(target_mag)
        mask *= (1.0 - cfg.harmonicity_weight) + cfg.harmonicity_weight * h[None, :]

    if cfg.smoothing_freq_bins > 1 or cfg.smoothing_time_frames > 1:
        mask = uniform_filter(mask, mode='nearest',
                              size=(cfg.smoothing_freq_bins, cfg.smoothing_time_frames))
    return mask


def speech_presence(band_levels, target_idx, floor, cfg):
    """Per-frame probability that the target is the one speaking.

    Two independent pieces of evidence, both required:

      * the target's mic leads every other mic across the speech band, and
      * the target's mic sits meaningfully above *its own* noise floor.

    The second is what `soft_gate` was reaching for, but it measured level
    against the global peak. A noise floor is a property of the microphone; a
    peak is a property of the worst thing that happened during the session.

    The hangover is symmetric, so one dilation protects both onsets and the
    low-energy unvoiced tails that a release-only gate clips off.
    """
    L = np.asarray(band_levels)
    target = L[target_idx]
    others = np.max(np.delete(L, target_idx, axis=0), axis=0)

    dominance_db = 20 * np.log10(target / others)
    snr_db = 20 * np.log10(target / floor)

    p = (_sigmoid(dominance_db / cfg.presence_dominance_width_db)
         * _sigmoid((snr_db - cfg.presence_snr_db) / (cfg.presence_snr_width_db / 4.0)))

    hang = max(1, int(round(cfg.hangover_ms / 1000.0 * cfg.frame_rate)))
    return maximum_filter1d(p, size=2 * hang + 1, mode='nearest')


def ambience_spectrum(track, levels, cfg, seconds=20.0):
    """Median magnitude spectrum of one microphone while nobody is talking.

    Sampled from the quietest one-second windows of the track, so the substituted
    noise sounds like that room through that microphone rather than like white
    noise dropped into a recording.
    """
    win = max(1, int(round(cfg.frame_rate)))            # frames per second
    n_win = len(levels) // win
    if n_win < 4:
        return None
    per_window = levels[:n_win * win].reshape(n_win, win).mean(axis=1)
    n_take = int(np.clip(seconds, 4, max(4, n_win // 10)))
    take = np.argsort(per_window)[:n_take]
    chunk = win * cfg.hop
    quiet = np.concatenate([track[k * chunk:(k + 1) * chunk] for k in take])
    if len(quiet) < cfg.nperseg * 2:
        return None
    spec = stft(quiet, fs=cfg.rate, nperseg=cfg.nperseg, noverlap=cfg.noverlap)[2]
    return np.median(np.abs(spec), axis=1)


_NOISE_GRID = 1 << 20


def positional_noise(start, n_samples, track_idx, cfg):
    """White noise determined by absolute sample position, not by draw order.

    Drawing from a running generator would make the output depend on how the
    session happened to be split into blocks. Seeding on a fixed grid keeps
    `block_seconds` a pure implementation detail.
    """
    first, last = start // _NOISE_GRID, (start + n_samples - 1) // _NOISE_GRID
    buf = np.concatenate([
        np.random.default_rng([cfg.seed, track_idx, g]).standard_normal(
            _NOISE_GRID).astype(np.float32)
        for g in range(first, last + 1)])
    off = start - first * _NOISE_GRID
    return buf[off:off + n_samples]


def shaped_noise_spectrum(start, n_samples, ambience, cfg, track_idx):
    """STFT of time-domain noise shaped to `ambience`.

    Generated in the time domain and transformed, rather than synthesised from
    random phase, so the overlap-add in `istft` stays consistent and the noise
    comes out at the level it was asked for.
    """
    noise = positional_noise(start, n_samples, track_idx, cfg)
    spec = stft(noise, fs=cfg.rate, nperseg=cfg.nperseg, noverlap=cfg.noverlap)[2]
    white = np.median(np.abs(spec), axis=1, keepdims=True)
    return spec * (ambience[:, None] / np.maximum(white, 1e-20))


def isolate_v3(file_list, config=None, verbose=True):
    """Isolate each microphone's own speaker using the whole array.

    Returns a dict with the isolated audio, the per-frame speech probability that
    drove the suppression, and everything measured along the way.

    Processing is blocked so peak memory does not depend on session length or on
    the number of microphones; the calibration and speech-presence decisions are
    made once globally beforehand, on cheap frame envelopes, so blocking cannot
    change them.
    """
    cfg = config or IsolationConfig()

    if verbose:
        print("Loading audio files...")
    rate, tracks = load_tracks(file_list)
    if rate != cfg.rate:
        cfg = replace(cfg, rate=rate)
    n_mics = len(tracks)
    n_samples = len(tracks[0])
    hop = cfg.hop

    # --- pass 1: calibrate the tracks against each other -------------------
    if verbose:
        print(f"Calibrating {n_mics} tracks ({cfg.calibration})...")
    levels = [frame_levels(t, hop) for t in tracks]
    gains, cal_info = calibration_gains(
        levels, method=cfg.calibration, target_level_db=cfg.target_level_db,
        speech_level_percentile=cfg.speech_level_percentile)
    tracks = [t * g for t, g in zip(tracks, gains)]
    levels = [l * g for l, g in zip(levels, gains)]
    floors = [noise_floor(l, cfg.noise_floor_percentile) for l in levels]
    if verbose:
        print(f"  gains (dB):        {cal_info['gains_db']}")
        print(f"  noise floor (dBFS): {[round(20 * np.log10(f), 1) for f in floors]}")

    # --- pass 2: frame-rate speech presence, from band-limited envelopes ----
    if verbose:
        print("Computing cross-microphone speech presence...")
    band = []
    for t in tqdm(tracks, desc="Band levels", disable=not verbose):
        band.append(frame_levels(_bandpass(t, rate, cfg.speech_band_hz), hop))
    band_floors = [noise_floor(b, cfg.noise_floor_percentile) for b in band]
    presence = [speech_presence(band, i, band_floors[i], cfg) for i in range(n_mics)]
    del band

    ambience = [None] * n_mics
    if cfg.fills_residual:
        ambience = [ambience_spectrum(t, l, cfg, cfg.ambience_seconds)
                    for t, l in zip(tracks, levels)]
        if verbose:
            missing = [i for i, a in enumerate(ambience) if a is None]
            print(f"Comfort noise at {cfg.residual_noise_over_db:+.0f} dB over the residual"
                  + (f" (no ambience for tracks {missing})" if missing else ""))

    # --- pass 3: blocked time-frequency masking ----------------------------
    block = max(1, int(cfg.block_seconds * rate) // hop) * hop
    pad = max(1, int(cfg.block_pad_seconds * rate) // hop) * hop
    out = [np.zeros(n_samples, dtype=np.float32) for _ in range(n_mics)]
    starts = list(range(0, n_samples, block))

    for start in tqdm(starts, desc="Isolating", disable=not verbose):
        stop = min(start + block, n_samples)
        # pad the block so STFT edge effects fall outside the part we keep
        a = max(0, start - pad)
        b = min(n_samples, stop + pad)

        specs = np.array([stft(t[a:b], fs=rate, nperseg=cfg.nperseg,
                               noverlap=cfg.noverlap)[2] for t in tracks],
                         dtype=np.complex64)
        mags = np.abs(specs)
        top1, top2, argmax = max_excluding(mags)
        frame0 = a // hop

        for i in range(n_mics):
            other = np.where(argmax == i, top2, top1)
            mask = dominance_mask(mags[i], other, cfg)

            p = presence[i][frame0:frame0 + mask.shape[1]]
            if len(p) < mask.shape[1]:
                p = np.pad(p, (0, mask.shape[1] - len(p)), mode='edge')
            mask *= (cfg.residual_floor + (1.0 - cfg.residual_floor) * p)[None, :]
            masked = specs[i] * mask

            if ambience[i] is not None:
                # Bury whatever survived the floor under noise at the same spectral
                # shape, weighted by how sure we are the target is *not* speaking, so
                # nothing is added over their own voice.
                noise = shaped_noise_spectrum(a, b - a, ambience[i], cfg, i)
                resid = np.sqrt(np.mean(np.abs(masked) ** 2, axis=0)) + 1e-20
                nlevel = np.sqrt(np.mean(np.abs(noise) ** 2, axis=0)) + 1e-20
                k = min(len(resid), len(nlevel), len(p))
                scale = np.zeros(noise.shape[1], dtype=np.float32)
                scale[:k] = (resid[:k] / nlevel[:k]
                             * 10 ** (cfg.residual_noise_over_db / 20)
                             * (1.0 - p[:k]))
                masked = masked + noise * scale[None, :]

            rec = istft(masked, fs=rate, nperseg=cfg.nperseg,
                        noverlap=cfg.noverlap)[1]
            rec = rec[:b - a] if len(rec) >= b - a else np.pad(rec, (0, b - a - len(rec)))
            out[i][start:stop] = rec[start - a:stop - a]

        del specs, mags, top1, top2, argmax

    # --- one output gain for the whole session -----------------------------
    # Per-track peak normalisation, as the v2 entry points did, makes every
    # track's level depend on its own worst transient and destroys the level
    # relationships between people that downstream acoustic features rely on.
    out_levels = [frame_levels(o, hop) for o in out]
    ref = float(np.median([speech_level(l, cfg.speech_level_percentile)
                           for l in out_levels]))
    g = 10 ** (cfg.output_level_db / 20) / max(ref, 1e-12)
    peak = max(float(np.abs(o).max()) for o in out) * g
    limit = 10 ** (cfg.peak_limit_db / 20)
    if peak > limit:
        g *= limit / peak
    out = [(o * g).astype(np.float32) for o in out]

    return {
        'rate': rate,
        'audio': out,
        'speech_probability': [p.astype(np.float32) for p in presence],
        'frame_rate': cfg.frame_rate,
        'hop': hop,
        'calibration': cal_info,
        'noise_floor_db': [round(20 * np.log10(f), 2) for f in floors],
        'output_gain_db': round(20 * np.log10(g), 2),
        'config': cfg.to_dict(),
        'files': [os.path.basename(f) for f in file_list],
    }
