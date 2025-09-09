import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OR_API_KEY')
)

with open('transcript.txt', 'r', encoding='utf-8') as f:
    transcribed_text = f.read()

with open('prompt.txt', 'r', encoding='utf-8') as f:
    reference_text = f.read()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[
        {"role": "user", "content": f"""Reference: {reference_text}

Transcribed: {transcribed_text}

Use reference to correct errors, translate to English only:"""}
    ]
)

print(response.choices[0].message.content)
