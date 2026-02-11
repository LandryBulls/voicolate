import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, binary_dilation, percentile_filter, uniform_filter
import os
from tqdm import tqdm
from scipy import signal
from scipy.signal import stft, istft, welch

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


def soft_gate(audio, rate=44100, threshold_db=-40, ratio=4.0, attack_ms=5, release_ms=50,
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
                              apply_gate=True, gate_threshold_db=-40, gate_ratio=4.0,
                              gate_attack_ms=5, gate_release_ms=50,
                              gate_lookahead_ms=30, gate_hold_ms=50):
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

def isolate_audio(file_list, config=None, save_files=False, output_path=None, cache_dir=None, overwrite=False,
                normalize_input=True, input_normalization_params=None, 
                peak_limit_output=True, peak_limit_db=-1.0):
    """
    Uses RMS values from Wiener-filtered audio to remove interference.
    
    Parameters:
        file_list: List of audio files to process
        config: MaskConfig object for masking parameters
        save_files: Whether to save the final isolated audio files
        output_path: Directory to save final isolated audio files
        cache_dir: Directory to store/load cached Wiener-filtered files
        overwrite: If True, recompute and overwrite existing cached Wiener files
        normalize_input: Whether to normalize input audio before processing (default: True)
                        This equalizes speaker levels for consistent gate behavior
        input_normalization_params: Dictionary with normalization parameters for input. 
                                   Defaults to {'method': 'rms', 'target_level': -20}
        peak_limit_output: Whether to apply peak limiting to prevent clipping (default: True)
        peak_limit_db: Peak limit in dB (default: -1.0 to leave 1dB headroom)
    """
    if config is None:
        config = MaskConfig()
    
    if normalize_input and input_normalization_params is None:
        input_normalization_params = {'method': 'rms', 'target_level': -20}
    
    # Load raw audio
    nussl = _get_nussl()
    raw_audio = [nussl.AudioSignal(f).audio_data[0] for f in file_list]
    
    # Normalize input audio to equalize speaker levels BEFORE processing
    if normalize_input:
        print('Normalizing input audio to equalize speaker levels...\n')
        raw_audio = normalize_audio_tracks(raw_audio, 
                                          method=input_normalization_params['method'],
                                          target_level=input_normalization_params['target_level'])
        
        # Create temporary normalized files for Wiener filtering
        temp_dir = os.path.join(os.getcwd(), 'temp_normalized')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_files = []
        rate = nussl.AudioSignal(file_list[0]).sample_rate
        for i, audio in enumerate(raw_audio):
            temp_file = os.path.join(temp_dir, f'temp_normalized_{i}.wav')
            wavfile.write(temp_file, rate, audio.astype(np.float32))
            temp_files.append(temp_file)
        
        file_list_for_wiener = temp_files
    else:
        file_list_for_wiener = file_list
        
    print('Applying Wiener Filter or loading from cache...\n')
    wiener_outputs = apply_wiener(file_list_for_wiener, cache_dir=cache_dir, 
                                 overwrite=overwrite, iterations=config.iterations)
    
    # Clean up temporary files if we created them
    if normalize_input:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    
    print('Masking...\n')
    masked_audio = mask_audio(wiener_outputs, raw_audio, config)
    
    # Apply conservative peak limiting to prevent clipping
    if peak_limit_output:
        print('Applying peak limiting...\n')
        peak_limit_linear = 10 ** (peak_limit_db / 20)
        for i in range(len(masked_audio)):
            peak = np.max(np.abs(masked_audio[i]))
            if peak > peak_limit_linear:
                masked_audio[i] = masked_audio[i] * (peak_limit_linear / peak)
    
    if save_files:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        saved_files = save_isolated_audio(masked_audio, config.rate, output_path)
        return saved_files
    else:
        return masked_audio


def isolate_spectral(file_list, dominance_margin_db=1.0, harmonicity_weight=0.5,
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