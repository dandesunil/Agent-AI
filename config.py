import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
HF_API_KEY = os.getenv("HF_API_KEY",None)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Model Configurations
HF_MODEL = os.getenv("HF_MODEL", None)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", None)

# Derived URLs
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


