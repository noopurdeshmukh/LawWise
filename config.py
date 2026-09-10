import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    VECTOR_STORE_PATH = os.path.join(os.getcwd(), "data", "vector_store")
    SQLALCHEMY_DATABASE_URI = "sqlite:///lawwise.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False