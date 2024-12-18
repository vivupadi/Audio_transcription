import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import os
import soundfile as sf

#import requests
#from django.conf import settings

torchaudio.set_audio_backend("soundfile")

#mention FFMpeg path
AudioSegment.converter = r"C:\\Users\\Vivupadi\Downloads\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\\Users\\Vivupadi\Downloads\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\bin\\ffprobe.exe"

# Load Hugging Face model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

#audio transcription
def transcribe_audio(file_path):

    # load
    #aud = AudioSegment.converter
    audio = AudioSegment.from_file(file_path)
    #audio = aud.from_file(file_path)
    wav_path = file_path.replace(".ogg", ".wav")
    audio.export(wav_path, format="wav")
    #waveform, rate = torchaudio.load(file_path)

    # resample
    waveform, rate = sf.read(wav_path)

    # Convert waveform to a PyTorch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)
    
    resampler = torchaudio.transforms.Resample(rate, 16000)
    waveform = resampler(waveform).squeeze()

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    #os.remove(wav_path)  # Clean up
    return transcription

"""
def transcribe_audio(file_path):
    
    api_url = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
    api_key = settings.huggingface_api_key  # Get API key from settings
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as file:
        response = requests.post(api_url, headers=headers, data=file)

    if response.status_code == 200:
        transcription = response.json()
        return transcription.get("text", "No transcription available.")
    else:
        return f"Error: {response.status_code}, {response.text}"
"""