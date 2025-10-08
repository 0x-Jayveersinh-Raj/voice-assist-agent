# STT (Speech-to-Text) module
from .base import STTProvider
from .factory import STTFactory

# Import providers conditionally
try:
    from .deepgram_provider import DeepgramSTT
    __all__ = ['STTProvider', 'STTFactory', 'DeepgramSTT']
except ImportError:
    __all__ = ['STTProvider', 'STTFactory']

try:
    from .whisper_provider import WhisperSTT
    if 'DeepgramSTT' not in __all__:
        __all__ = ['STTProvider', 'STTFactory', 'WhisperSTT']
    else:
        __all__.append('WhisperSTT')
except ImportError:
    pass
