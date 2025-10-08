import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from dotenv import load_dotenv, find_dotenv
import asyncio
from .stt import STTFactory
from .vad_processor import VADProcessor
from .llm import LLMFactory
import numpy as np
from .schemas.schema import LLMRequest

# Load env vars from repo-level .env (auto-discover), overriding existing vars if provided
load_dotenv(find_dotenv(), override=True)

# Also load server-local .env explicitly
_server_env = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_server_env):
    load_dotenv(_server_env, override=True)
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


    


# -----------------------------
# STT Management Endpoints
# -----------------------------
@app.get("/stt/providers")
def get_available_providers():
    """Get list of available STT providers."""
    try:
        providers = STTFactory.get_available_providers()
        provider_info = {}
        for provider_type in providers:
            provider_info[provider_type] = STTFactory.get_provider_info(provider_type)
        return {"providers": provider_info}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/llm/respond")
def llm_respond(req: LLMRequest):
    """Send text to an LLM and return its response. Uses provider from request or defaults to 'gemini' if available."""
    try:
        provider = req.provider or os.getenv("LLM") or ("gemini" if "gemini" in LLMFactory.get_available_providers() else None)
        if not provider:
            return JSONResponse(content={"error": "No LLM providers available"}, status_code=400)

        llm = LLMFactory.create(provider)
        resp = llm.respond(req.text)
        # print server-side for now and return the response
        print(f"[LLM] provider={provider} response={resp}")
        return {"provider": provider, "response": resp}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -----------------------------
# WebSocket: Deepgram Streaming
# -----------------------------
@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()

    # Lazy import to avoid hard dependency when not used
    try:
        from deepgram import AsyncDeepgramClient
        from deepgram.core.events import EventType
        from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    except Exception as e:
        await websocket.send_json({"error": f"Deepgram SDK not available: {e}"})
        await websocket.close()
        return

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        await websocket.send_json({"error": "DEEPGRAM_API_KEY not set"})
        await websocket.close()
        return

    try:
        dg = AsyncDeepgramClient(api_key=api_key)
        model = os.getenv("DEEPGRAM_MODEL", "nova-2")
        language = os.getenv("STT_LANGUAGE", "en")

        # Track final sentence assembly
        final_sentences = []

        # Initialize local VAD and LLM
        vad = VADProcessor()
        # Create a default LLM instance (from LLM env var or gemini fallback)
        llm = None
        try:
            default_provider = os.getenv("LLM") or ("gemini" if "gemini" in LLMFactory.get_available_providers() else None)
            print(f"[LLM-init] default_provider={default_provider} available={LLMFactory.get_available_providers()}")
            if default_provider:
                try:
                    llm = LLMFactory.create(default_provider)
                except Exception as e:
                    print(f"[LLM-init] failed to create provider {default_provider}: {e}")
                    llm = None
            print(f"[LLM-init] llm={'set' if llm else 'None'}")
        except Exception:
            llm = None
        
        # Open Deepgram listen v1 websocket
        async with dg.listen.v1.connect(
            model=model,
            language=language,
            encoding="linear16",
            sample_rate="16000",
            interim_results="true",
            punctuate="true",
            smart_format="true",
        ) as dg_socket:

            # Handle incoming messages
            async def handle_message(message):
                try:
                    if isinstance(message, ListenV1ResultsEvent):
                        alt = message.channel.alternatives[0] if message.channel.alternatives else None
                        text = alt.transcript if alt else ""
                        is_final = bool(message.is_final)
                        if not text:
                            return
                        if is_final:
                            print(f"[Final] {text}")
                            final_sentences.append(text)
                            print(f"[Final->buffer] final_sentences now: {final_sentences}")
                            await websocket.send_json({"final": text})
                        else:
                            print(f"[Partial] {text}")
                            await websocket.send_json({"partial": text})
                        sys.stdout.flush()
                except Exception:
                    pass

            dg_socket.on(EventType.MESSAGE, handle_message)

            # Start listening in background
            listen_task = asyncio.create_task(dg_socket.start_listening())

            # Receive audio frames from client and forward to Deepgram
            try:
                while True:
                    data = await websocket.receive_bytes()
                    if not data:
                        continue
                    # Forward raw bytes to Deepgram
                    await dg_socket.send_media(data)

                    # Also feed into local VAD. Expecting 16kHz 16-bit PCM little-endian
                    try:
                        audio_np = np.frombuffer(data, dtype=np.int16)
                        is_end, chunk = vad.process_audio(audio_np)
                        if is_end:
                            # When VAD detects end of utterance, wait briefly for Deepgram final events
                            print(f"[VAD] triggered; before wait final_sentences={final_sentences} llm={'set' if llm else 'None'}")
                            # to arrive (they may be delivered slightly after the audio frame).
                            WAIT_FOR_FINALS = float(os.getenv("VAD_WAIT_FOR_FINALS", "0.6"))
                            POLL_INTERVAL = 0.05
                            waited = 0.0
                            while waited < WAIT_FOR_FINALS and not final_sentences:
                                await asyncio.sleep(POLL_INTERVAL)
                                waited += POLL_INTERVAL
                            print(f"[VAD] after wait final_sentences={final_sentences}")
                            # Assemble final transcript and call LLM in executor
                            full_text = " ".join(final_sentences).strip()
                            print(f"[VAD] end-of-utterance detected. transcript='{full_text}'")
                            sys.stdout.flush()
                            if full_text and llm:
                                try:
                                    loop = asyncio.get_running_loop()
                                    print(f"[VAD] calling LLM.respond with: {full_text}")
                                    # Run potentially blocking LLM call in executor
                                    response = await loop.run_in_executor(None, llm.respond, full_text)
                                    print(f"[LLM] {response}")
                                    # Also send to websocket client
                                    try:
                                        await websocket.send_json({"llm_response": response})
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"[LLM] error: {e}")
                                    sys.stdout.flush()
                            # clear assembled sentences after handling
                            final_sentences = []
                    except Exception:
                        # ignore VAD errors and continue
                        pass
            except WebSocketDisconnect:
                pass
            except Exception as e:
                try:
                    await websocket.send_json({"error": str(e)})
                except Exception:
                    pass
            finally:
                # Finalize and close
                try:
                    listen_task.cancel()
                except Exception:
                    pass
                try:
                    if final_sentences:
                        await websocket.send_json({"final_full": " ".join(final_sentences).strip()})
                except Exception:
                    pass
                try:
                    await websocket.close()
                except Exception:
                    pass
    except Exception as e:
        try:
            await websocket.send_json({"error": f"Failed to start Deepgram live: {e}"})
            await websocket.close()
        except Exception:
            pass


    
