import whisperx
import torch
import librosa

# Check if GPU is available
# also check if m1 mac

if torch.cuda.is_available():
    device = 'cuda'
    compute_type = 'float16'
elif torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Apple M1':
    device = 'cuda'
    compute_type = 'float16'
else:
    device = 'cpu'
    compute_type = 'float32'

model = whisperx.load_model("large", device, compute_type=compute_type)

def transcribe(audio_file):
    # WhisperX handles audio parameter so that it can be a path (str), np.ndarray, or torch.tensor
    # But here we're loading an audio file.
    audio, sr = librosa.load(audio_file, sr=44100)
    # resample to 16kHz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    result = model.transcribe(audio, language='en')
    #model_a, metadata = whisperx.load_align_model(language_code='en', device=device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    # align whisper output
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    return result_aligned