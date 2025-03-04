import numpy as np
import nussl
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from tqdm import tqdm
from scipy import signal
from scipy.signal import stft, welch

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
    """
    mix = np.zeros(len(audio_iter[0]))
    for audio in audio_iter:
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
    Takes a numpy array representing audio and returns rolling-window root-mean-squared value.
    """
    window_size = int(round((rate/1000)*window_ms))
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))


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

class MaskConfig:
    def __init__(self, rate=44100, window_ms=100, iterations=20, base_sigma=100, base_widen_ms=200, 
                 base_percentile=75, speech_band_hz=(550, 2205), speech_sigma=100, 
                 speech_widen_ms=100, speech_percentile=75, speech_filter=False,
                 min_duration_ms=100):
        # Basic audio parameters
        self.rate = rate                    # Sample rate of audio (Hz)
        self.window_ms = window_ms          # Window size for RMS calculation (milliseconds)

        # Wiener filter parameters
        self.iterations = iterations
        
        # Base mask parameters
        self.base_sigma = base_sigma        # Gaussian smoothing width for the mask
        self.base_widen_ms = base_widen_ms  # How much to widen the mask regions (milliseconds)
        self.base_percentile = base_percentile  # Threshold percentile for initial mask creation
        
        # Speech-specific parameters
        self.speech_band_hz = speech_band_hz    # Frequency range for speech detection (Hz)
        self.speech_sigma = speech_sigma        # Gaussian smoothing for speech mask
        self.speech_widen_ms = speech_widen_ms  # How much to widen speech regions (milliseconds)
        self.speech_percentile = speech_percentile  # Threshold percentile for speech detection
        self.speech_filter = speech_filter      # Whether to apply speech-specific filtering
        
        # Noise removal parameters
        self.min_duration_ms = min_duration_ms  # Minimum duration of valid segments (milliseconds)
        
        # Pre-calculate commonly used values (in samples)
        self.base_widen_samples = int((base_widen_ms / 1000) * rate)
        self.speech_widen_samples = int((speech_widen_ms / 1000) * rate)
        self.min_duration_samples = int((min_duration_ms / 1000) * rate)
        self.downsample_factor = int(rate * 0.1)  # 100ms worth of samples for mask smoothing

def mask(wiener_output, raw_audio, config, return_mask=False):
    """
    Modified masking function with two-pass filtering:
    1. Overall RMS-based mask (generated from the Wiener-filtered audio and applied to the raw audio)
    2. Speech-band specific mask (generated from the raw audio and applied to the raw audio)
    """
    # Get overall RMS
    rms = window_rms(wiener_output, rate=config.rate, window_ms=config.window_ms)
    rms_percentile = np.percentile(rms, config.base_percentile)
    
    # Generate first mask
    base_mask = np.ones_like(rms)
    base_mask[rms < rms_percentile] = 0
    
    # Remove short instances of 1's
    eroded = binary_dilation(1 - base_mask, structure=np.ones(config.min_duration_samples))
    base_mask = base_mask * (1 - eroded)
    
    # Apply dilation
    base_mask = binary_dilation(base_mask, structure=np.ones(config.base_widen_samples))

    # Second pass: Speech band specific mask
    if config.speech_filter:
        speech_band_rms = get_band_rms(wiener_output, config.speech_band_hz, 
                                     config.rate, config.window_ms)
        speech_threshold = np.percentile(speech_band_rms, config.speech_percentile)
        
        speech_mask = np.ones_like(speech_band_rms)
        speech_mask[speech_band_rms < speech_threshold] = 0
        speech_mask = binary_dilation(speech_mask, 
                                    structure=np.ones(config.speech_widen_samples))
        combined_mask = base_mask * speech_mask
    else:
        combined_mask = base_mask
    
    # Apply smoothing
    smoothed_mask = smooth_mask(combined_mask, config.base_sigma, 
                              config.downsample_factor, config.rate)
    
    # Apply mask to audio
    output = raw_audio * smoothed_mask
    
    if return_mask:
        return output, smoothed_mask
    else:
        return output

def mask_audio(wiener_outputs, raw_audio, config):
    """
    Takes Wiener-filtered audio and returns masked audio.
    """
    masked_audio = []
    for wout, raw in zip(wiener_outputs, raw_audio):
        masked_audio.append(mask(wout, raw, config))
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
                normalize_output=True, normalization_params=None):
    """
    Uses RMS values from Wiener-filtered audio to remove interference.
    
    Parameters:
        file_list: List of audio files to process
        config: MaskConfig object for masking parameters
        save_files: Whether to save the final isolated audio files
        output_path: Directory to save final isolated audio files
        cache_dir: Directory to store/load cached Wiener-filtered files
        overwrite: If True, recompute and overwrite existing cached Wiener files
        normalize_output: Whether to normalize the final output (default: True)
        normalization_params: Dictionary with normalization parameters. Defaults to 
                            {'method': 'lufs', 'target_level': -23}
    """
    if config is None:
        config = MaskConfig()
    
    if normalize_output and normalization_params is None:
        normalization_params = {'method': 'lufs', 'target_level': -23}
        
    print('Applying Wiener Filter or loading from cache...\n')
    wiener_outputs = apply_wiener(file_list, cache_dir=cache_dir, overwrite=overwrite, iterations=config.iterations)
    raw_audio = [nussl.AudioSignal(f).audio_data[0] for f in file_list]
    print('Masking...\n')
    masked_audio = mask_audio(wiener_outputs, raw_audio, config)
    
    if normalize_output:
        print('Normalizing output...\n')
        masked_audio = normalize_audio_tracks(masked_audio, 
                                           method=normalization_params['method'],
                                           target_level=normalization_params['target_level'])
    
    if save_files:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        saved_files = save_isolated_audio(masked_audio, config.rate, output_path)
        return saved_files
    else:
        return masked_audio