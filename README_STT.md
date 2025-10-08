# Speech-to-Text (STT) System

A modular and extensible Speech-to-Text system supporting multiple providers including Deepgram and Whisper.

## Features

- **Multiple STT Providers**: Support for Deepgram API and local Whisper models
- **Modular Architecture**: Easy to add new STT providers
- **Real-time Streaming**: Support for streaming audio transcription
- **Language Support**: Multiple language support for both providers
- **RESTful API**: Easy integration with web applications
- **Comprehensive Testing**: Full test coverage for all components

## Architecture

The STT system is built with a modular architecture:

```
server/stt/
├── __init__.py          # Module exports
├── base.py              # Abstract base class for STT providers
├── factory.py           # Factory for creating STT providers
├── deepgram_provider.py # Deepgram API implementation
└── whisper_provider.py  # Whisper local implementation
```

### Base Class (`STTProvider`)

All STT providers inherit from the `STTProvider` abstract base class, ensuring consistent interface:

- `transcribe(audio_data, sample_rate)` - Transcribe audio data
- `transcribe_streaming(audio_chunk, sample_rate)` - Transcribe streaming chunks
- `get_supported_languages()` - Get supported languages
- `set_language(language)` - Set transcription language
- `get_config()` / `update_config()` - Configuration management

### Factory Pattern

The `STTFactory` provides a clean way to create and manage STT providers:

```python
from server.stt.factory import STTFactory

# Create a provider
provider = STTFactory.create_provider('whisper', {'model': 'tiny'})

# Get available providers
providers = STTFactory.get_available_providers()

# Get provider information
info = STTFactory.get_provider_info('deepgram')
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
# For Deepgram
export DEEPGRAM_API_KEY="your_deepgram_api_key"

# For Whisper (optional)
export WHISPER_MODEL="tiny"  # tiny, base, small, medium, large
export WHISPER_COMPUTE_TYPE="int8"  # int8, float16, float32

# General STT settings
export STT_PROVIDER="whisper"  # whisper or deepgram
export STT_LANGUAGE="en"  # Language code
```

## Usage

### Server

Start the FastAPI server:

```bash
cd server
python main.py
```

The server provides the following endpoints:

- `POST /stream/chunk` - Stream audio chunks for transcription
- `GET /stt/providers` - Get available STT providers
- `GET /stt/current` - Get current provider information
- `POST /stt/switch` - Switch STT provider
- `POST /stt/language` - Set transcription language

### Client

Run the audio streaming client:

```bash
cd client
python send_audio.py
```

The client supports interactive commands:
- `providers` - Show available STT providers
- `current` - Show current provider info
- `switch <provider>` - Switch STT provider
- `language <lang>` - Set language
- `quit` - Exit

### Python API

```python
from server.stt.factory import STTFactory
import numpy as np

# Create a Whisper provider
whisper_provider = STTFactory.create_provider('whisper', {
    'model': 'tiny',
    'language': 'en'
})

# Create a Deepgram provider
deepgram_provider = STTFactory.create_provider('deepgram', {
    'api_key': 'your_api_key',
    'language': 'en',
    'model': 'nova-2'
})

# Transcribe audio
audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
transcript = whisper_provider.transcribe(audio_data)
print(transcript)
```

## Providers

### Whisper Provider

Local Whisper models using either `faster-whisper` or `openai-whisper`.

**Configuration:**
- `model`: Model size (tiny, base, small, medium, large)
- `language`: Language code or 'auto' for automatic detection
- `compute_type`: Computation type for faster-whisper (int8, float16, float32)
- `beam_size`: Beam search size
- `use_faster_whisper`: Use faster-whisper if available (default: True)

**Supported Languages:** 100+ languages including English, Hindi, Gujarati, Bengali, Tamil, Telugu, etc.

### Deepgram Provider

Cloud-based STT using Deepgram API.

**Configuration:**
- `api_key`: Deepgram API key (required)
- `language`: Language code
- `model`: Deepgram model (nova-2, base, enhanced)
- `smart_format`: Enable smart formatting
- `punctuate`: Enable punctuation

**Supported Languages:** 30+ languages including English, Hindi, Gujarati, Spanish, French, German, etc.

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/ -m unit
python -m pytest tests/ -m integration

# Run with coverage
python -m pytest tests/ --cov=server.stt
```

Test structure:
- `test_stt_base.py` - Base class tests
- `test_stt_factory.py` - Factory tests
- `test_whisper_provider.py` - Whisper provider tests
- `test_deepgram_provider.py` - Deepgram provider tests
- `test_integration.py` - Integration tests

## Adding New Providers

To add a new STT provider:

1. Create a new class inheriting from `STTProvider`:

```python
from server.stt.base import STTProvider

class CustomSTT(STTProvider):
    def transcribe(self, audio_data, sample_rate=16000):
        # Implementation
        pass
    
    def transcribe_streaming(self, audio_chunk, sample_rate=16000):
        # Implementation
        pass
    
    def get_supported_languages(self):
        return ['en', 'custom']
    
    def set_language(self, language):
        self.config['language'] = language
```

2. Register with the factory:

```python
from server.stt.factory import STTFactory
STTFactory.register_provider('custom', CustomSTT)
```

3. Add tests in `tests/test_custom_provider.py`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STT_PROVIDER` | STT provider to use | `whisper` |
| `STT_LANGUAGE` | Transcription language | `en` |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required for Deepgram |
| `DEEPGRAM_MODEL` | Deepgram model | `nova-2` |
| `WHISPER_MODEL` | Whisper model size | `tiny` |
| `WHISPER_COMPUTE_TYPE` | Whisper compute type | `int8` |

## Performance Considerations

### Whisper
- **tiny**: Fastest, lowest accuracy (~39 MB)
- **base**: Good balance (~74 MB)
- **small**: Better accuracy (~244 MB)
- **medium**: High accuracy (~769 MB)
- **large**: Best accuracy (~1550 MB)

### Deepgram
- **nova-2**: Latest model, best accuracy
- **base**: Standard model
- **enhanced**: Enhanced model for better accuracy

## Troubleshooting

### Common Issues

1. **ImportError for faster-whisper**: Install with `pip install faster-whisper`
2. **Deepgram API errors**: Check API key and quota
3. **Audio format issues**: Ensure 16kHz, 16-bit, mono audio
4. **Memory issues**: Use smaller Whisper models or Deepgram

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.
