import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    UPLOAD_FOLDER = 'app/uploads'
    MAX_CONTENT_LENGTH = 128 * 1024 * 1024  # Limit upload to 16 MB
