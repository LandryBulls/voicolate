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
                 batch_size=441000):
    """
    Takes list of .wav files and returns filtered audio.
    Assumes all audio files are mono and of the *exact* same length
    """
    naud = len(file_list)
    estimates = [nussl.AudioSignal(i) for i in file_list]
    rate = estimates[0].sample_rate
    shape = estimates[0].audio_data.shape[0]

    if not all([estimates[i].audio_data.shape for i in range(naud)]):
        raise Exception("Audio files are of different lengths!")
    
    # Extract audio data and normalize
    audio_data = [est.audio_data[0] for est in estimates]
    normalized_audio = normalize_audio_tracks(audio_data, method='lufs', target_level=-23)
    
    # Update the estimates with normalized audio
    for i, est in enumerate(estimates):
        est.audio_data = normalized_audio[i][np.newaxis, :]

    # this could be one problem
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

    if save_to_file:
        if output_path == None:
            output_path = os.path.getcwd()
        for f, file in enumerate(file_list):
            out = os.path.join(output_path, os.path.basename(file)[:-4] + '_wiener.wav')
            wavfile.write(out, estimates[0].sample_rate, outs[f])
    if return_outputs:
        return outs

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

def mask(wiener_output, raw_audio, rate=44100, window_ms=10, base_sigma=100,  base_widen_ms=200, base_percentile=75, 
         speech_band_hz=(550, 2205), speech_sigma=100, speech_widen_ms=100, speech_percentile=90, speech_filter=True, return_mask=False):
    """
    Modified masking function with two-pass filtering:
    1. Overall RMS-based mask (generated from the Wiener-filtered audio and applied to the raw audio)
    2. Speech-band specific mask (generated from the raw audio and applied to the raw audio)
    """
    # Print parameters  
    print(f'base_sigma: {base_sigma}, base_widen_ms: {base_widen_ms}, base_percentile: {base_percentile}, '
          f'speech_band_hz: {speech_band_hz}, speech_sigma: {speech_sigma}, speech_widen_ms: {speech_widen_ms}, '
          f'speech_percentile: {speech_percentile}')

    # First pass: Overall RMS-based mask
    base_widen_samples = int((base_widen_ms / 1000) * rate)
    base_dilate_struct = np.ones(base_widen_samples)

    # Get overall RMS
    rms = window_rms(wiener_output, rate=rate, window_ms=window_ms)
    rms_percentile = np.percentile(rms, base_percentile)
    
    # Generate first mask
    base_mask = np.ones_like(rms)
    base_mask[rms < rms_percentile] = 0
    
    # Apply dilation
    base_mask = binary_dilation(base_mask, structure=base_dilate_struct)

    # Second pass: Speech band specific mask
    if speech_filter:
        speech_band_rms = get_band_rms(wiener_output, speech_band_hz, rate, window_ms)
        speech_threshold = np.percentile(speech_band_rms, speech_percentile)
        
        speech_mask = np.ones_like(speech_band_rms)
        speech_mask[speech_band_rms < speech_threshold] = 0
        speech_widen_samples = int((speech_widen_ms / 1000) * rate)
        speech_dilate_struct = np.ones(speech_widen_samples)
        speech_mask = binary_dilation(speech_mask, structure=speech_dilate_struct)
        # Combine masks
        combined_mask = base_mask * speech_mask
    else:
        combined_mask = base_mask

    # Apply gaussian filter with proper scaling
    smoothed_mask = gaussian_filter(combined_mask.astype(float), sigma=base_sigma, mode='reflect')
    
    # Rescale the smoothed mask to [0,1] range
    smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min() + 1e-10)
    
    # Interpolate mask if needed to match audio length
    if len(smoothed_mask) != len(raw_audio):
        time_points = np.linspace(0, len(raw_audio), len(smoothed_mask))
        full_time = np.arange(len(raw_audio))
        smoothed_mask = np.interp(full_time, time_points, smoothed_mask)
    
    # Apply mask to audio
    output = raw_audio * smoothed_mask
    
    if return_mask:
        return output, smoothed_mask
    else:
        return output
    
def mask_audio(wiener_outputs, raw_audio, rate=44100, window_ms=10, base_sigma=100, base_widen_ms=200, base_percentile=75, 
               speech_band_hz=(550, 2205), speech_sigma=100, speech_widen_ms=100, speech_percentile=90, speech_filter=True, return_mask=False):
    """
    Takes Wiener-filtered audio and returns masked audio.
    """
    masked_audio = []
    for wout, raw in zip(wiener_outputs, raw_audio):
        masked_audio.append(mask(wout, raw, rate=rate, window_ms=window_ms, base_sigma=base_sigma, base_widen_ms=base_widen_ms, base_percentile=base_percentile, 
                                 speech_band_hz=speech_band_hz, speech_sigma=speech_sigma, speech_widen_ms=speech_widen_ms, speech_percentile=speech_percentile, 
                                 speech_filter=speech_filter, return_mask=return_mask))
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

def isolate_audio(file_list, rate=44100, window_ms=10, base_sigma=100, base_widen_ms=200, base_percentile=75, speech_filter=True,
                  speech_band_hz=(550, 2205), speech_sigma=100, speech_widen_ms=100, speech_percentile=90, save_files=False, output_path=None):
    """
    Uses RMS values from Wiener-filtered audio to remove interference. Input is a list of audio files
    Returns numpy vectors representing the cleaned sound.
    """
    print('Applying Wiener Filter, may take a while...\n')
    wiener_outputs = apply_wiener(file_list)
    raw_audio = [nussl.AudioSignal(f).audio_data[0] for f in file_list]
    print('Masking...\n')
    masked_audio = mask_audio(wiener_outputs, raw_audio, rate=rate, window_ms=window_ms, base_sigma=base_sigma, base_widen_ms=base_widen_ms, base_percentile=base_percentile, 
                             speech_band_hz=speech_band_hz, speech_sigma=speech_sigma, speech_widen_ms=speech_widen_ms, speech_percentile=speech_percentile, speech_filter=speech_filter)
    if save_files:
        # check if output path exists. If not, make it.
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        saved_files = save_isolated_audio(masked_audio, rate, output_path)
        return saved_files
    else:
        return masked_audio

