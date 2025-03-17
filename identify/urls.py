from django.urls import path
from .views import upload_audio, check_status

urlpatterns = [
    path('', upload_audio, name='upload_audio'),
    path('status/<str:task_id>/', check_status, name='check_status'),
]
