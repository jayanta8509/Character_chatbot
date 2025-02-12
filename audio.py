from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
# speech_file_path = Path(__file__).parent / "speech.mp3"
# response = client.audio.speech.create(
#     model="tts-1",
#     voice="alloy",
#     input="Today is a wonderful day to build something people love!",
# )
# response.stream_to_file(speech_file_path)


# from flask import Flask, send_file


# def get_audio(path):
#     with open(path, "rb") as f:
#         audio_data = f.read()
#         return audio_data


# audio2 = 'speech.mp3'

# hits = get_audio(audio2)
# print(hits)

from openai import AsyncOpenAI
client = AsyncOpenAI()

async def generate(text,voicec):
        response = await client.audio.speech.create(
            model="tts-1",
            # voice="alloy",
            voice=voicec,
            input=text
        )
        
        # Use a regular for loop instead of async for
        for chunk in response.iter_bytes(chunk_size=4096):
            print("hiy : ",chunk)
            yield chunk


# voice = "alloy"
# gh = "HI i am bb"
# p= generate(gh,voice)
# print(p)