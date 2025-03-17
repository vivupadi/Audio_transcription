# identify/tasks.py
from celery import shared_task
from .utils import transcribe_audio

@shared_task
def transcribe_audio_task(file_path):
    return transcribe_audio(file_path)