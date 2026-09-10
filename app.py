"""
LawWise — Flask Application Entry Point
"""
import os
import logging
from flask import Flask
from config import Config
from database.db import init_db
from routes import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure data directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["VECTOR_STORE_PATH"], exist_ok=True)

    # Initialize database
    init_db(app)

    # Register blueprints
    app.register_blueprint(main)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
