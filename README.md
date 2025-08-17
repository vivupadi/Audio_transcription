# Django-based Audio transcription Tool
The purpose is to create a Audio transcription tool using Django framework
<img width="1903" height="867" alt="image" src="https://github.com/user-attachments/assets/5e201cc1-a4a7-4ab2-924e-ae9014ac801e" />

## Tech Stack
**Django**
**HuggingFace API**

## Deployment
**Render** Free version

## Steps
The model used is Whisper 'tiny'.
The audio file was resampled to 'Mono' channel & 16KHz. (Since most of the audio transcription models are trained on mono channel and 16KHz channels)
CI-CD implemented using Github actions.
The APP is hosted on Render free tier.
