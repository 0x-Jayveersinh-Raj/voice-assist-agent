How to add a new STT provider

1. Create `server/stt/<your_provider>_provider.py`.
2. Implement a class that inherits from `server.stt.base.STTProvider` and implements required methods.
3. Make sure the provider file imports register itself or update `server/stt/factory.py` to include it.

Example (minimal):

```py
from .base import STTProvider

class DemoSTT(STTProvider):
    def __init__(self, config=None):
        self.config = config or {}

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        return "simulated transcription"

# In factory.py, import and register: _providers['demo'] = DemoSTT
```
