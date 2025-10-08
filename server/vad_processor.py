import webrtcvad
import collections
import numpy as np

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 320 samples
MAX_SILENCE_FRAMES = 15  # 15 x 20ms = 300ms silence

class VADProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0–3
        self.buffer = []
        self.silence_counter = 0
        self.in_speech = False

    def process_audio(self, pcm_data: np.ndarray):
        frames = np.array_split(pcm_data, len(pcm_data) // FRAME_SIZE)
        output_chunk = []

        for frame in frames:
            if len(frame) < FRAME_SIZE:
                continue
            is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

            if is_speech:
                self.buffer.append(frame)
                self.in_speech = True
                self.silence_counter = 0
            elif self.in_speech:
                self.silence_counter += 1
                self.buffer.append(frame)

                if self.silence_counter > MAX_SILENCE_FRAMES:
                    chunk = np.concatenate(self.buffer)
                    self.reset()
                    return True, chunk

        return False, None

    def reset(self):
        self.buffer = []
        self.silence_counter = 0
        self.in_speech = False
