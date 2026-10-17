import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(timeout=60.0, max_retries=1)


response = client.responses.create(
    model=os.getenv("BOTANICAL_AI_MODEL", "gpt-4o-mini"),
    input="Explain what a plant gene is in one simple sentence.",
)

print(response.output_text)