import numpy as np
import nussl
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter

def apply_wiener(file_list, iterations=10, save_to_file=False, output_path = None, return_outputs=True):
    """
    Takes list of .wav files and returns filtered audio.
    """
    naud = len(file_list)
    estimates = [nussl.AudioSignal(i) for i in file_list]
    mix = np.zeros(estimates[0].audio_data.shape)
    for i in range(naud):
        mix += estimates[i].audio_data
    mix = nussl.core.AudioSignal(audio_data_array=mix,sample_rate=estimates[0].sample_rate)
    wiener  = nussl.separation.benchmark.WienerFilter(mix, estimates, iterations=iterations)
    wout = wiener()
    outputs = [i.audio_data[0] for i in wout]
    if save_to_file:
        if output_path==None:
            output_path = os.path.getcwd()
        for f, file in enumerate(file_list):
            out = os.path.join(output_path, os.path.basename(file)[:-4]+'_wiener.wav')
            wavfile.write(out, estimates[0].sample_rate, outputs[f])
    if return_outputs:
        return outputs

def window_rms(a, rate=44100, window_ms=10):
    """
    Takes a numpy array representing audio and returns rolling-window root-mean-squared value.
    """
    window_size = int(round((rate/1000)*window_ms))
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))

def mask_audio(wiener_outputs, raw_audio, rate=44100, window_ms=10, stride_ms=2, threshold=0.001, sigma=20):
    """
    Gets RMS of Wiener-filtered audio and uses it to mask the original audio to retain quality.
    A gaussian filter is used to smooth in and out phases of speech to reduce choppiness.
    """
    cleaned_outputs = []
    for w, wout in enumerate(wiener_outputs):
        loud = window_rms(wout)
        #smooth in and outs to reduce choppiness
        loud[np.where(loud!=1)] = gaussian_filter(loud, sigma)[np.where(loud!=1)]
        loud[np.where(loud<threshold)] = 0
        loud[np.where(loud>threshold)] = 1
        clean = loud*raw_audio[w]
        cleaned_outputs.append(clean)
    return cleaned_outputs

def save_audio(array_list, rate=44100, output_path = None, output_name=None):
    if not output_name:
        outnames = [str(i)+'_isolated.wav' for i in range(len(array_list))]
    else:
        outnames = [output_name+'_'+str(i)+'.wav' for i in range(len(array_list))]
    if not output_path:
        output_path = os.getcwd()
    for f, array in enumerate(array_list):
        wavfile.write(os.path.join(output_path, outnames[f]), rate, array)

def isolate_audio(file_list, rate=44100, save_files=False, output_path=None):
    """
    Uses RMS values from Wiener-filtered audio to remove interference. Input is a list of audio files
    Returns numpy vectors representing the cleaned sound.
    """
    print('Applying Wiener Filter, may take a while...')
    wiener_outputs = apply_wiener(file_list)
    raw_audio = [nussl.AudioSignal(f).audio_data[0] for f in file_list]
    print('Masking...')
    masked_audio = mask_audio(wiener_outputs, raw_audio, rate=rate)
    if save_files:
        save_audio(masked_audio, rate, output_path)
    return masked_audio

