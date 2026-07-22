import os

from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL = "models/gemini-3.1-flash-lite"
#model="models/gemini-3.5-flash",
#model="models/gemini-3.1-flash-lite",

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not configured. "
        "Please add it to the .env file."
    )