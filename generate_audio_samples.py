from pathlib import Path
import os
from scipy.io import wavfile
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import time
from librosa import resample 

def extract_audio(posix_video_path):
    print("Extracting audio from video files...\n")
    out_audio_path = str(os.path.basename(str(posix_video_path)[:-4]+'.wav'))
    ffmpeg_extract_audio(str(posix_video_path), out_audio_path)
    # librosa returns the numpy array as well as the sample rate
    sr, loaded = wavfile.read(out_audio_path)
    if sr != 44100:
        loaded = resample(loaded, sr, 44100)
    # delete the file
    os.remove(out_audio_path)
    # returns both the audio as a numpy array and the sample rate (arr, sr)
    return loaded


main_dir = Path('../../data/andromeda_storage/conversations_unconstrained')


dirs = [d for d in main_dir.iterdir() if d.is_dir() and (d / 'processed' / '0_isolated.wav').exists()]

afile_names = [f'{i}_isolated.wav' for i in range(4)]

for session in dirs:
    try:
        session_name = session.name
        sessproc = session / 'processed'
        print(f"Processing {session_name}...")
        micaudiofiles = [f for f in sessproc.iterdir() if f.is_file() and f.name in afile_names]
        micaudiofiles.sort()
        micaudiodata = []
        for file in micaudiofiles:
            sr, aud = wavfile.read(str(file))
            if sr != 44100:
                aud = resample(aud, sr, 44100)
            micaudiodata.append(aud)
        micaudiodata = np.array(micaudiodata)

        # mix all mics
        mic_mix = np.sum(micaudiodata, axis=0)
        print(f'Mixed audio shape: {mic_mix.shape}')

        # get cam1 file
        cam1file = session / 'derivatives' / 'cam1_concatenated_trimmed.mp4'
        cam1_audio = extract_audio(cam1file)
        # convert to mono
        cam1_audio = cam1_audio.sum(axis=1)
        print(f'Cam1 audio shape: {cam1_audio.shape}')

        # get cam2 file
        cam2file = session / 'derivatives' / 'cam2_concatenated_trimmed.mp4'
        cam2_audio = extract_audio(cam2file)
        # make mono
        cam2_audio = cam2_audio.sum(axis=1)
        print(f'Cam2 audio shape: {cam2_audio.shape}')

        # mix all three (pad the shorter ones first)
        if mic_mix.shape[0] < cam1_audio.shape[0]:
            mic_mix = np.pad(mic_mix, (0, cam1_audio.shape[0] - mic_mix.shape[0]))
        elif cam1_audio.shape[0] < mic_mix.shape[0]:
            cam1_audio = np.pad(cam1_audio, (0, mic_mix.shape[0] - cam1_audio.shape[0]))
        if len(mic_mix) < len(cam2_audio):
            mic_mix = np.pad(mic_mix, (0, len(cam2_audio) - len(mic_mix)))
        elif len(cam2_audio) < len(mic_mix):
            cam2_audio = np.pad(cam2_audio, (0, len(mic_mix) - len(cam2_audio)))

        mixed_audio = np.sum([mic_mix, cam1_audio, cam2_audio], axis=0)

        # choose a 10-second clip from the middle (Hz if 44100)
        start = int(round(mixed_audio.shape[0] // 2 - 5 * 44100))
        end = int(round(mixed_audio.shape[0] // 2 + 5 * 44100))

        # convert start and end to hhmmss
        start_hhmmss = time.strftime('%H:%M:%S', time.gmtime(start / 44100))
        end_hhmmss = time.strftime('%H:%M:%S', time.gmtime(end / 44100))

        mixed_audio = mixed_audio[start:end]

        # normalize the audio
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

        # save the audio
        out_audio_path = str(session / f'audio_mix_{str(start_hhmmss)}-{str(end_hhmmss)}.wav')
        wavfile.write(out_audio_path, 44100, mixed_audio)

    except Exception as e:
        print(f"Error in {session_name}: {e}")
        continue
