# Phone Assist Voice

This repository is a small streaming STT + LLM demo that accepts audio from a client, performs Voice Activity Detection (VAD), transcribes speech with STT (Deepgram or Whisper), and forwards completed utterances to an LLM (Gemini by default) for a response.

This README explains the runtime flow (class/function level), how components are wired, how to run and test locally, and how to add new STT or LLM providers.

## Quick start (PowerShell)

1. Create and activate the virtual environment (already created as `.venv` if you followed earlier steps):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Create `server/.env` from `server/.env.template` and fill in keys:

```properties
DEEPGRAM_API_KEY="..."
GEMINI_API_KEY="..."
GEMINI_API_URL="https://..."   # optional if using genai SDK
LLM=gemini
STT_PROVIDER=deepgram
STT_LANGUAGE=en
VAD_WAIT_FOR_FINALS=0.6
```

3. Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --reload
```

4. In another terminal, run the client to stream audio from microphone:

```powershell
.\.venv\Scripts\python.exe client\send_audio.py
```

The client will print partial/final transcripts and receive LLM responses as JSON messages with key `llm_response`.

## High-level runtime flow (what happens after client sends audio)

Sequence (short):

1. Client (`client/send_audio.py`) opens a websocket to `ws://<server>/ws/transcribe` and streams raw 16 kHz 16-bit PCM audio bytes.
2. Server (`server/main.py`) accepts websocket and forwards audio to the streaming STT (Deepgram listen.v1) while also feeding the same PCM frames to local VAD (`server/vad_processor.py`).
3. STT provider (Deepgram) emits partial and final transcript events; server buffers final transcripts into `final_sentences`.
4. VAD detects end-of-utterance (silence after speech). When VAD triggers, the server waits briefly for any remaining STT final events, then assembles `final_sentences` into one utterance and sends that text to the configured LLM provider via `LLMFactory`.
5. LLM responds; server prints the response and sends it back to the client as `{"llm_response": "..."}` for now.

Sequence (detailed, class/function wise):

- client/send_audio.py
  - Reads microphone frames (16kHz, mono, 16-bit). Repeatedly sends raw bytes to the websocket.
  - Receives JSON messages from server and prints `partial`, `final`, `final_full`, `llm_response`, `error` messages.

- server/main.py
  - Endpoint: `@app.websocket("/ws/transcribe")`
  - Creates:
    - `AsyncDeepgramClient` (if using Deepgram streaming) to forward audio and receive transcription events.
    - `VADProcessor()` from `server/vad_processor.py` to detect end-of-speech on raw PCM.
    - `llm = LLMFactory.create(default_provider)` where default is configured via `LLM` env var (fallback to `gemini`).
  - `handle_message()` callback: runs when STT emits results. If `is_final`, append text to `final_sentences`.
  - Audio receive loop: for each audio chunk from the websocket:
    - `await dg_socket.send_media(data)` forwards to Deepgram.
    - `audio_np = np.frombuffer(data, dtype=np.int16)` then `is_end, chunk = vad.process_audio(audio_np)`.
    - If VAD signals end:
      - Wait up to `VAD_WAIT_FOR_FINALS` seconds for `final_sentences` to be populated.
      - Assemble `full_text = " ".join(final_sentences).strip()`.
      - Call `await loop.run_in_executor(None, llm.respond, full_text)` so the LLM call does not block the event loop.
      - Send back `{"llm_response": <response>}`.

- server/vad_processor.py
  - Class `VADProcessor` uses `webrtcvad`.
  - `process_audio(pcm_np: np.ndarray)` splits audio into fixed-duration frames and returns `(True, chunk)` when it detects silence long enough to mark the end of an utterance. It buffers frames while in-speech.

- server/stt/factory.py and `server/stt/*`
  - `STTFactory` conditionally imports and registers available STT providers (Deepgram and Whisper).
  - `DeepgramSTT` is streaming-focused and the server uses Deepgram's SDK directly in the websocket handler (server/main).
  - `WhisperSTT` uses local Whisper/faster-whisper for offline transcription (not used by the websocket default in current setup but available in factory).

- server/llm/factory.py and `server/llm/gemini_provider.py`
  - `LLMFactory.register_provider(name, class)` registers providers.
  - `GeminiLLM` uses the `google.genai` SDK (if installed) or returns an informative error if not available. It expects `GEMINI_API_KEY` and will use `genai.Client()` to call `models.generate_content`.

## Messages & JSON shapes

- From server to client (examples):
  - `{"partial": "partial text"}`
  - `{"final": "final sentence"}`
  - `{"final_full": "full assembled utterance"}`
  - `{"llm_response": "<LLM reply>"}`
  - `{"error": "<message>"}`

## How to add a new STT provider

1. Create file `server/stt/<your>_provider.py`.
2. Implement a class that inherits from `server.stt.base.STTProvider` and implements the required API (e.g., `transcribe`, `transcribe_streaming`, `get_supported_languages`, etc.). See `whisper_provider.py` for an example implementation.
3. Register the provider in `server/stt/factory.py` by importing and adding to `_providers` or call `STTFactory.register_provider("name", YourClass)`.
4. Switch provider at runtime by setting `STT_PROVIDER=<name>` in `server/.env` (the server currently uses Deepgram streaming by default; adding a new streaming flow may require updating the websocket handler to use that provider's streaming primitives).

Notes: The websocket handler currently uses Deepgram's SDK paths directly for streaming. If you want to support streaming for other providers, add an alternate websocket route or refactor `server/main.py` to create an abstraction that accepts provider streaming sockets.

## How to add a new LLM provider

1. Create `server/llm/<your>_provider.py`.
2. Implement a class that inherits from `server/llm/factory.BaseLLM` and implements `respond(prompt: str, history: Optional[list] = None) -> str`.
3. Register at module import time with `LLMFactory.register_provider("your", YourClass)` or register centrally.
4. Use the provider by setting `LLM=<your>` in `server/.env` or by passing `provider` in `/llm/respond` POST body.

## HTTP endpoints (for testing)

- GET /health
  - Returns {"status":"ok"}.

- GET /stt/providers
  - Lists available STT providers and their info.

- POST /llm/respond
  - Body: `{ "text": "...", "provider": "optional-provider-name" }`
  - Returns `{ "provider": "...", "response": "..." }` or an error.

## Troubleshooting

- Empty LLM calls after VAD fired
  - Increase `VAD_WAIT_FOR_FINALS` in `server/.env` (default `0.6` seconds) to allow STT to emit its final transcripts before the LLM is called.

- 403 from Google GenAI / Gemini
  - Ensure `GEMINI_API_KEY` (or ADC/service account credentials) belongs to a Google Cloud project with the Generative Language API enabled. The error payload usually includes the numeric project id. Enable the API in Cloud Console and/or use credentials from a project with the API enabled.

- Debugging startup/providers
  - At import time, LLM providers register themselves. If a provider does not appear in `LLMFactory.get_available_providers()`, confirm the provider module imports correctly and that registration code runs at module level.

## File map (important files)

- server/main.py — websocket & http endpoints, orchestration between VAD/STT/LLM.
- server/vad_processor.py — VAD implementation using webrtcvad.
- server/whisper_wrapper.py — (local whisper wrapper) used by whisper STT provider.
- server/stt/factory.py — STT factory and provider registration.
- server/stt/whisper_provider.py — Whisper provider.
- server/stt/deepgram_provider.py — Deepgram provider.
- server/llm/factory.py — LLM factory and BaseLLM.
- server/llm/gemini_provider.py — Gemini LLM provider (uses genai SDK if installed).
- client/send_audio.py — example client that streams audio to server websocket.

## Development tips

- Keep API keys out of source control. Use `server/.env` (not committed) and add `.env` to `.gitignore`.
- For local testing without Deepgram, you can add a websocket route that uses `WhisperSTT` directly; I can help add that if you want offline testing.
- When adding providers, add unit tests to `tests/` to validate registration and basic request/response shapes.

---

To do:
- Add a small `server/debug` endpoint that returns masked credential info and registered providers.

