from django.shortcuts import render
from django.core.files.storage import default_storage
from .models import AudioFile
from .utils import transcribe_audio

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio']:
        audio_file = request.FILES['audio']
        saved_path = default_storage.save(audio_file.name, audio_file)
        full_path = default_storage.path(saved_path)

        # Transcribe audio
        transcription = transcribe_audio(full_path)

        # Save file and display results
        AudioFile.objects.create(file=audio_file)
        return render(request, 'identify/upload.html', {'transcription': transcription})

    return render(request, 'identify/upload.html')

