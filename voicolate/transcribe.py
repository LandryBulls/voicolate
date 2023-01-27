import os
import glob
import whisperx
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = whisperx.load_model("large", device)

def transcribe(audio):
    # WhisperX handles audio parameter so that it can be a path (str), np.ndarray, or torch.tensor
    result = model.transcribe(audio, language='en')
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    # align whisper output
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    return result_aligned['segments']