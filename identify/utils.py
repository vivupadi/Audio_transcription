import torch
torch.set_num_threads(1)
#from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import numpy as np

from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import platform
import soundfile as sf

#import requests
#from django.conf import settings

try:
    torchaudio.set_audio_backend("sox_io")
except Exception:
    torchaudio.set_audio_backend("soundfile")   ##Soundfile WOrks for windows systems

processor = AutoProcessor.from_pretrained("openai/whisper-tiny", language ='en')
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny", 
    torch_dtype=torch.float16,  # Uses less memory
    low_cpu_mem_usage=True,      # Optimizes memory allocation
).to("cpu")  # Move model to CPU

model = model.to(torch.float32)


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float32,
    device=-1
)

#audio transcription
def transcribe_audio(file_path):
    if not file_path.lower().endswith(".wav"):
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"  #In order to accept any audio format
        #wav_path = file_path.replace(".*", ".wav")
        
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format="wav")
    else:
        wav_path = file_path
    #waveform, rate = torchaudio.load(file_path)
    waveform, sample_rate = torchaudio.load(wav_path)
    print(f"Sample Rate: {sample_rate} Hz")

    # Convert to mono if necessary
    if waveform.shape[0] > 1:  # Check if audio has multiple channels
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
        print("Converted audio to mono.")

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
        print(f"Resampled audio to {sample_rate} Hz.")

    # Ensure waveform shape is (num_samples,)
    waveform = waveform.squeeze()  # Remove extra dimensions
    print(f"Waveform shape after squeezing: {waveform.shape}")

    # Convert waveform to numpy array
    waveform_np = waveform.numpy()
    print(f"Waveform shape for pipeline: {waveform_np.shape}")

    
    def chunk_audio(audio_array, sample_rate, chunk_size=30, overlap=1):
        step = chunk_size - overlap  # Shift next chunk by `step`
        chunks = []
        for i in range(0, len(audio_array), step * sample_rate):
            chunk = audio_array[i : i + chunk_size * sample_rate]
            chunks.append(chunk)
        return chunks

    # Initialize the pipeline
    #pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", tokenizer="openai/whisper-tiny")

    transcriptions = []

    chunks = chunk_audio(waveform_np, sample_rate)
    
    for chunk in chunks:
        # Convert audio chunk to numpy array
        waveform_chunk = np.array(chunk)
        
        # Process each chunk with the pipeline
        transcription = pipe(waveform_chunk, return_timestamps = True)
        transcriptions.append(transcription['text'])

    # Combine transcriptions
    return " ".join(transcriptions)

    """
    #audio = aud.from_file(file_path)
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"  #In order to accept any audio format
    #wav_path = file_path.replace(".*", ".wav")
    
    audio.export(wav_path, format="wav")
    #waveform, rate = torchaudio.load(file_path)

    # resample
    waveform, rate = sf.read(wav_path)

    # Convert waveform to a PyTorch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)

    print('waveform before squeezing:', waveform.shape)
    
    resampler = torchaudio.transforms.Resample(rate, 16000)
    waveform = resampler(waveform).squeeze()

    print('waveform after squeezing:', waveform.shape)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    decoder_input_ids = torch.zeros((inputs['input_features'].shape[0], 1), dtype=torch.long)  # Assuming start token
    inputs['decoder_input_ids'] = decoder_input_ids

    print(f"Processed inputs: {inputs.keys()}")
    
    with torch.no_grad():
        logits = model(inputs.input_features, decoder_input_ids=inputs['decoder_input_ids']).logits

    print("Logits shape:", logits.shape)

    predicted_ids = torch.argmax(logits, dim=-1)
    # Decode the predicted ids and remove special tokens
    decoded_output = processor.decode(predicted_ids[0])

    # Clean unwanted special tokens like <|en|>
    #cleaned_output = decoded_output.replace('<|en|>', '').strip()
    """