import whisperx
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = whisperx.load_model("large", device)

def transcribe(audio_file):
    # WhisperX handles audio parameter so that it can be a path (str), np.ndarray, or torch.tensor
    # But here we're loading an audio file.
    audio, sr = librosa.load(audio_file, sr=44100)
    # resample to 16kHz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    result = model.transcribe(audio, language='en')
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    # align whisper output
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    return result_aligned