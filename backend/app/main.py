"""
FastAPI application for a voice‑enabled AI agent.

This service exposes a WebSocket endpoint ``/ws`` that accepts
microphone audio from a web client. When the client sends raw
audio bytes (recorded via the MediaRecorder API), the server
accumulates those bytes until it receives a special ``"END"``
message. Once audio recording is complete, the server:

1. Saves the received audio to a temporary file.
2. Uses ``ffmpeg`` to convert the audio to 16‑kHz mono WAV.
3. Runs speech‑to‑text using the Faster Whisper model to
   transcribe the user’s utterance.
4. Sends the transcription to a vLLM server exposed via
   ``VLLM_URL``, along with a configurable system prompt.
5. Receives a textual response from the language model.
6. Uses Coqui TTS to synthesise the response to a WAV file.
7. Streams the audio bytes back to the client over the same
   WebSocket for playback.

The service also serves static files from ``/public`` so that
the browser can load ``index.html`` without needing a separate
HTTP server.

Environment variables:
    VLLM_URL (str): Base URL for the vLLM OpenAI compatible
        chat completion endpoint (e.g. http://vllm:8000/v1/chat/completions).
    SYSTEM_PROMPT (str, optional): A prompt inserted before the
        user’s message to steer the model’s behaviour. Defaults to
        a friendly assistant.

Before deploying, ensure that the ``vllm`` service is reachable
and that the required models are downloaded on first start.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List

import httpx
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from TTS.api import TTS


app = FastAPI()

# Serve static assets (HTML/JS/CSS) under the / path. The
# ``public`` directory is bundled into the Docker image and
# mounted at /app/public.
static_dir = Path(__file__).resolve().parents[1] / "public"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

# Load configuration from environment variables. Provide sensible
# defaults so the service runs out of the box in development.
VLLM_URL: str = os.getenv("VLLM_URL", "http://vllm:8000/v1/chat/completions")
SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are a helpful, friendly assistant. Answer clearly "
        "and concisely in a conversational tone."
    ),
)

# Instantiate the whisper and TTS models once at startup to avoid
# repeated initialisation. The model sizes are chosen to balance
# quality with runtime performance. Faster Whisper will
# automatically select the best available device (GPU/CPU). If
# running on CPU, smaller models like "base" or "small" work
# acceptably for short queries.

@app.on_event("startup")
async def load_models() -> None:
    """Load ASR and TTS models at startup."""
    # Whisper ASR: choose 'base' for CPU or 'small' for GPU. The
    # compute type 'float16' allows GPU acceleration when CUDA is
    # available; otherwise it falls back to 'int8' on CPU.
    asr_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    # Store on application state for reuse across requests.
    app.state.asr_model = WhisperModel(asr_size, device="auto", compute_type=compute_type)

    # Coqui TTS: choose a lightweight English model. The model
    # ``tts_models/en/ljspeech/tacotron2-DDC" provides natural
    # prosody and is relatively small. It will download on first
    # use and cache to the container file system.
    tts_model_name = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
    app.state.tts_model = TTS(tts_model_name, progress_bar=False, gpu=False)


async def transcribe_audio(wav_path: Path) -> str:
    """
    Perform speech‑to‑text on a WAV file using Faster Whisper.

    Args:
        wav_path: Path to a mono WAV audio file sampled at 16 kHz.

    Returns:
        The transcribed text as a single string.
    """
    segments, _ = app.state.asr_model.transcribe(
        str(wav_path), beam_size=5, best_of=5
    )
    transcript_parts: List[str] = [seg.text.strip() for seg in segments]
    return " ".join(transcript_parts).strip()


async def query_vllm(message: str) -> str:
    """
    Query the vLLM service with a chat message.

    The request uses the OpenAI Chat API schema. The system prompt
    precedes the user's message to set the tone. Temperature and
    top_p can be adjusted via environment variables if needed.

    Args:
        message: The user’s transcribed input.

    Returns:
        The assistant’s textual reply.
    """
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        # Use moderate randomness for conversational tone.
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(VLLM_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    # Extract the first choice's content.
    return data["choices"][0]["message"]["content"].strip()


async def synthesize_speech(text: str) -> bytes:
    """
    Convert text into speech using Coqui TTS.

    Args:
        text: The model’s response.

    Returns:
        Bytes of the generated audio in WAV format (22.05 kHz).
    """
    # Use TTS to generate a waveform array and save to memory.
    audio_array = app.state.tts_model.tts(text)
    # Write to an in‑memory buffer using SoundFile.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio_array, app.state.tts_model.synthesizer.output_sample_rate)
        return Path(f.name).read_bytes()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    WebSocket endpoint for full duplex audio communication.

    The client sends binary messages containing raw audio frames.
    When the client sends the text message ``"END"``, the server
    stops recording, processes the accumulated audio and sends back
    the synthesised response audio.
    """
    await ws.accept()
    # Accumulate audio bytes from the client.
    audio_bytes: bytearray = bytearray()
    try:
        while True:
            msg = await ws.receive()
            # ``msg`` contains either 'text' or 'bytes'. The client
            # should send a plain string "END" when finished.
            if "text" in msg:
                text_data = msg["text"]
                if text_data == "END":
                    break
                # Ignore other textual messages.
                continue
            elif "bytes" in msg:
                audio_bytes.extend(msg["bytes"])
    except WebSocketDisconnect:
        return

    # If no audio was received, do nothing.
    if not audio_bytes:
        await ws.close()
        return

    # Write the received audio to a temporary file. The browser
    # records as webm/opus by default. Save to an arbitrary file
    # extension; ffmpeg will detect the container format.
    with tempfile.TemporaryDirectory() as tmpdir:
        webm_path = Path(tmpdir) / f"input_{uuid.uuid4()}.webm"
        wav_path = Path(tmpdir) / f"input_{uuid.uuid4()}.wav"
        with open(webm_path, "wb") as f:
            f.write(audio_bytes)

        # Use ffmpeg to convert to mono 16 kHz WAV. Overwrite
        # without prompting. If conversion fails, close the
        # connection gracefully.
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(webm_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
                "-y",
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            await ws.close()
            return

        # Perform speech‑to‑text.
        transcript = await transcribe_audio(wav_path)
        # Immediately inform the client of the recognised text.
        await ws.send_json({"type": "transcript", "text": transcript})

        # Query the language model for a reply.
        try:
            answer = await query_vllm(transcript)
        except httpx.HTTPError:
            await ws.close()
            return

        # Synthesis
        audio_data = await synthesize_speech(answer)
        # Send the result back as binary bytes.
        await ws.send_bytes(audio_data)

    await ws.close()
