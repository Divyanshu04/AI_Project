from google import genai

from app.config import GOOGLE_API_KEY, MODEL


client = genai.Client(
    api_key=GOOGLE_API_KEY
)


def ask_gemini(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the response.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return response.text