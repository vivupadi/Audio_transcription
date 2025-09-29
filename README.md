# Django-based Audio transcription Tool
The purpose is to create a Audio transcription tool using Django framework
<img width="1903" height="867" alt="image" src="https://github.com/user-attachments/assets/5e201cc1-a4a7-4ab2-924e-ae9014ac801e" />

## 📥Installation
### Prerequisites
**Python 3.8+** - Download

**Pytorch** - Pipeline, preprocessing

**pip** - Python package manager

**FFmpeg** - Audio processing (required)

**Git** - Version control

**Django 4.0+** - Web framework

**HuggingFace API Key** - For AI transcription models (Get API Key)

## Deployment
**Architecture Overview**

| Component | Purpose |
|-----------|---------|
| **Django** | Managing Frontend & Backend services |
| **HuggingFace API** | Model API for speech to text conversion |
| **Audio Processing** | Preprocessing audio |
| **Render** | Hosting the Django app on the cloud |


## Steps
The model used is Whisper 'tiny'.
The audio file was resampled to 'Mono' channel & 16KHz. (Since most of the audio transcription models are trained on mono channel and 16KHz channels)
CI-CD implemented using Github actions.
The APP is hosted on Render free tier.


⭐ Star this repo if you find it helpful!
Made with ❤️ by Vivek Padayattil
