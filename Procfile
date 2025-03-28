
web: gunicorn --workers=1 --threads=2 --timeout=240 --bind 0.0.0.0:$PORT music_identifier.wsgi:application
worker: celery -A music_identifier worker --loglevel=info
