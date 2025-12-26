from pathlib import Path
from openai import AsyncOpenAI
import os
import asyncio
from dotenv import load_dotenv
import time
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI()

async def generate_voice(unic_user_id, text, voice_choice):
    # Create audio directory if it doesn't exist
    audio_dir = Path(__file__).parent / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Create consistent filename for each user ID
    # This will overwrite the previous file for the same user
    speech_file_path = audio_dir / f"{unic_user_id}_speech.mp3"
    
    # Use with_streaming_response as recommended
    async with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice_choice,
        input=text,
    ) as response:
        # Save the streaming response to a file
        with open(speech_file_path, "wb") as file:
            async for chunk in response.iter_bytes():
                file.write(chunk)
    
    return str(speech_file_path)


async def transcribe_audio(file_path):
    """
    Transcribe an audio file to text using OpenAI's Whisper model
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None
