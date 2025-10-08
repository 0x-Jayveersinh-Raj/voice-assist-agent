import pyaudio
import time
import asyncio
import websockets

CHUNK_DURATION_MS = 5000  # 5 seconds
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes (16-bit)
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

SERVER_WS_URL = "ws://localhost:8000/ws/transcribe"

async def stream_ws_from_microphone():
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("[🎤] Recording from microphone... Press Ctrl+C to stop")

    try:
        async with websockets.connect(SERVER_WS_URL, max_size=None) as ws:
            async def receiver():
                try:
                    async for message in ws:
                        # Server sends JSON; websockets delivers str
                        try:
                            import json as _json
                            payload = _json.loads(message)
                            if 'partial' in payload:
                                print(f"[Partial] {payload['partial']}")
                            if 'final' in payload:
                                print(f"[Final] {payload['final']}")
                            if 'final_full' in payload:
                                print(f"[Sentence] {payload['final_full']}")
                            if 'error' in payload:
                                print(f"[Error] {payload['error']}")
                        except Exception:
                            print(message)
                except Exception:
                    pass

            recv_task = asyncio.create_task(receiver())

            while True:
                start = time.time()
                audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await ws.send(audio_chunk)
                end = time.time()
                # Optional pacing/logging
                # print(f"Sent {len(audio_chunk)} bytes in {end-start:.3f}s")

            await recv_task

    except KeyboardInterrupt:
        print("\n[🛑] Stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(stream_ws_from_microphone())
    except RuntimeError:
        # For environments where an event loop is already running
        loop = asyncio.get_event_loop()
        loop.run_until_complete(stream_ws_from_microphone())





# import wave
# import requests
# import time

# CHUNK_DURATION_MS = 5000  # 5 seconds  # milliseconds
# SAMPLE_RATE = 16000
# SAMPLE_WIDTH = 2  # bytes
# CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * SAMPLE_WIDTH
# SERVER_URL = "http://localhost:8000/stream/chunk"

# def send_audio_chunks(wav_path):
#     with wave.open(wav_path, 'rb') as wf:
#         assert wf.getframerate() == SAMPLE_RATE, "Audio must be 16kHz"
#         assert wf.getsampwidth() == SAMPLE_WIDTH, "Audio must be 16-bit"
#         assert wf.getnchannels() == 1, "Audio must be mono"

#         while True:
#             chunk = wf.readframes(CHUNK_SIZE // SAMPLE_WIDTH)
#             if not chunk:
#                 break

#             files = {"chunk": ("chunk.pcm", chunk, "application/octet-stream")}
#             start = time.time()
#             response = requests.post(SERVER_URL, files=files)

#             if response.ok:
#                 json = response.json()
#                 if "transcript" in json:
#                     print(f"[Transcript] {json['transcript']}")
#                     end = time.time()
#                     print(f"Time taken: {end-start}")
#             else:
#                 print(f"[⚠️] Error from server: {response.status_code} - {response.text}")

#             # Simulate real-time streaming
#             time.sleep(CHUNK_DURATION_MS / 1000.0)

# if __name__ == "__main__":
#     send_audio_chunks("test_audio/sample_english.wav")
