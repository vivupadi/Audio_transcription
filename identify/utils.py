import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import os

#import requests
#from django.conf import settings

torchaudio.set_audio_backend("ffmpeg")

#mention FFMpeg path
AudioSegment.converter = r"C:\\Users\\Vivupadi\Downloads\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\\Users\\Vivupadi\Downloads\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\ffmpeg-2024-11-18-git-970d57988d-essentials_build\\bin\\ffprobe.exe"

# Load Hugging Face model and processor
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def transcribe_audio(file_path):
    """
    Transcribe audio using Hugging Face Wav2Vec2.
    """
    # Convert audio to WAV format
    #aud = AudioSegment.converter
    #audio = AudioSegment.from_file(file_path)
    #audio = aud.from_file(file_path)
    #wav_path = file_path.replace(".mp3", ".wav")
    #audio.export(wav_path, format="wav")
    waveform, rate = torchaudio.load(file_path)

    # Load audio and resample
    #waveform, rate = torchaudio.load(wav_path)
    resampler = torchaudio.transforms.Resample(rate, 16000)
    waveform = resampler(waveform).squeeze()

    # Process and transcribe
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    #os.remove(wav_path)  # Clean up
    return transcription

"""
def transcribe_audio(file_path):
    
    #Transcribe audio using Hugging Face Wav2Vec2.
    
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