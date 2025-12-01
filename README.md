# Hello Agent

This project provides a full-stack, containerised voice assistant that runs on a GPU-enabled host and uses Whisper for speech-to-text, vLLM serving Qwen2.5 for conversational AI, and Coqui TTS for text-to-speech. It's built using FastAPI for the backend, and a simple browser-based frontend.

## Requirements

- A machine with an Nvidia GPU (tested on A10 and RTX A6000).
- Docker and Docker Compose installed.

## Running

To start the services in production mode, run:

`docker compose up --build`

Then visit http://localhost to access the voice agent.

The first startup may take some time while models are downloaded and initialised. After that, the services should be ready quickly.

## Components

- **backend/**: A FastAPI application exposing a WebSocket endpoint (`/ws`) for streaming audio from the frontend, running speech-to-text via faster-whisper, sending prompts to Qwen2.5 via vLLM, converting the response to speech, and returning it back to the client. It also includes a minimal HTTP endpoint (`/ping`) for health checks.
- **public/**: A simple HTML file that implements a browser-based interface to record audio, send it to the backend over WebSocket, and play back the returned audio.
- **docker-compose.yml**: Orchestrates multiple services, including the vLLM server for Qwen2.5, and the backend service.
- **Dockerfile**: The backend service container file that installs dependencies and sets up the application.

## Usage

1. Run `docker compose up`.
2. Open your browser to `http://localhost`. Click the button to start recording, speak into your microphone, and release the button to send. The agent will process your speech and reply with an audio response.
3. To stop services, press `Ctrl+C` in the terminal or run `docker compose down`.

## Customisation

You can modify `backend/app/main.py` to adjust parameters such as:
- Whisper model size (e.g., "large-v3") if you have more GPU memory.
- vLLM temperature and top_p sampling values.
- The TTS model loaded in the backend.

## License

This project is provided for educational purposes.
