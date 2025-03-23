
web: gunicorn --workers=1 --threads=2 --timeout=240 --bind 0.0.0.0:$PORT your_project.wsgi:application
worker: celery -A your_project worker --loglevel=info
