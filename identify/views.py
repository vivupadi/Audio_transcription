from django.shortcuts import render
from django.core.files.storage import default_storage
from .models import AudioFile
from .utils import transcribe_audio

def upload_audio(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        saved_path = default_storage.save(audio_file.name, audio_file)
        full_path = default_storage.path(saved_path)

        # Save the file to the database
        AudioFile.objects.create(file=audio_file)

        # Start the Celery task
        task = transcribe_audio.delay(full_path)

        # Return a response with the task ID
        return render(request, 'identify/upload.html', {'task_id': task.id})

    return render(request, 'identify/upload.html')

from celery.result import AsyncResult
from django.http import JsonResponse

def check_status(request, task_id):
    task = AsyncResult(task_id)
    if task.ready():
        transcription = task.result
        return render(request, 'identify/result.html', {'transcription': transcription})
    else:
        return render(request, 'identify/result.html', {'status': 'Task is still processing...'})