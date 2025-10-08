# import whisper
# import numpy as np
# import tempfile
# import os
# import soundfile as sf
# # Load model once globally
# model = whisper.load_model("tiny")  # or "small", "medium", etc.

# def transcribe_chunk(audio_np: np.ndarray) -> str:
#     # Convert to float32 in [-1, 1] range
#     audio_float32 = audio_np.astype(np.float32) / 32768.0

#     # Save to temporary WAV file (Whisper doesn't accept raw numpy)
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#         temp_path = f.name
#         sf.write(f, audio_float32, 16000)

#     # Transcribe (auto language detection, native script output)
#     result = model.transcribe(temp_path)

#     # Clean up
#     os.remove(temp_path)

#     return result["text"].strip()

from faster_whisper import WhisperModel
import numpy as np

# Load once globally — tiny model is fast even on CPU
model = WhisperModel("tiny", compute_type="int8")  # You can try "base" too

def transcribe_chunk(audio_np: np.ndarray) -> str:
    # Convert int16 PCM to float32 in [-1, 1] range as expected
    audio_float32 = audio_np.astype(np.float32) / 32768.0

    # Transcribe (we pass raw audio array directly)
    segments, _ = model.transcribe(audio_float32, language="en", beam_size=1)

    # Combine segments
    transcript = "".join(segment.text for segment in segments)
    return transcript.strip()
