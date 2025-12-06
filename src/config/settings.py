import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-2.0-flash"
    LLM_TEMPERATURE = 0.2
    CHROMA_PERSIST_DIRECTORY = "./chroma_isro"
    DATA_FILE_PATH = "src/data/scraped_data.json"
    
    # Weather API
    MOSDAC_WEATHER_API_URL = "https://mosdac.gov.in/apiweather1/weather"
    USER_AGENT = "my_weather_bot"

settings = Settings()
