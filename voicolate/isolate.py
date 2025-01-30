import numpy as np
import nussl
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from tqdm import tqdm

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

def mask(wiener_output, raw_audio, sigma=100, rate=44100, window_ms=10, widen_ms=100, percentile=75, return_mask=False):
    """
    Modified masking function to preserve mask values after smoothing
    """
    # print all the parameters
    print(f'sigma: {sigma}, rate: {rate}, window_ms: {window_ms}, widen_ms: {widen_ms}, percentile: {percentile}')


    # get widen_ms in samples
    widen_samples = int((widen_ms / 1000) * rate)
    # create dilation structure
    dilate_struct = np.ones(widen_samples)

    # get rms
    rms = window_rms(wiener_output, rate=rate, window_ms=window_ms)
    
    # get percentile rms
    rms_percentile = np.percentile(rms, percentile)
    
    # generate mask
    mask = np.ones_like(rms)
    mask[rms < rms_percentile] = 0
    
    # Apply dilation
    mask = binary_dilation(mask, structure=dilate_struct)
    
    # Apply gaussian filter with proper scaling
    # Use mode='reflect' to handle edges better
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=sigma, mode='reflect')
    
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
    
def mask_audio(wiener_outputs, raw_audio, sigma=100, rate=44100, window_ms=10, widen_ms=100, percentile=75, return_mask=False):
    """
    Takes Wiener-filtered audio and returns masked audio.
    """
    masked_audio = []
    for wout, raw in zip(wiener_outputs, raw_audio):
        masked_audio.append(mask(wout, raw, sigma=sigma, rate=rate, window_ms=window_ms, widen_ms=widen_ms, percentile=percentile, return_mask=return_mask))
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

def isolate_audio(file_list, rate=44100, percentile=75, sigma=90, widen_ms=200, save_files=False, output_path=None):
    """
    Uses RMS values from Wiener-filtered audio to remove interference. Input is a list of audio files
    Returns numpy vectors representing the cleaned sound.
    """
    print('Applying Wiener Filter, may take a while...\n')
    wiener_outputs = apply_wiener(file_list)
    raw_audio = [nussl.AudioSignal(f).audio_data[0] for f in file_list]
    print('Masking...\n')
    masked_audio = mask_audio(wiener_outputs, raw_audio, sigma=sigma, percentile=percentile, rate=rate, widen_ms=widen_ms)
    if save_files:
        # check if output path exists. If not, make it.
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        saved_files = save_isolated_audio(masked_audio, rate, output_path)
        return saved_files
    else:
        return masked_audio

