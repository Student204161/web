from flask import Flask
from app.routes.main_routes_11 import main
from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(main)

    return app
