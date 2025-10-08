"""
Pytest configuration and fixtures.
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    # Generate 1 second of 16kHz audio data
    return np.random.randint(-32768, 32767, 16000, dtype=np.int16)


@pytest.fixture
def sample_audio_chunk():
    """Generate sample audio chunk for streaming tests."""
    # Generate 0.1 second of 16kHz audio data
    return np.random.randint(-32768, 32767, 1600, dtype=np.int16)


@pytest.fixture
def temp_audio_file(sample_audio_data):
    """Create a temporary audio file for testing."""
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Convert to float32 and normalize
        audio_float = sample_audio_data.astype(np.float32) / 32768.0
        sf.write(temp_file.name, audio_float, 16000)
        yield temp_file.name
    
    # Clean up
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def mock_deepgram_config():
    """Mock configuration for Deepgram provider."""
    return {
        'api_key': 'test_deepgram_key',
        'language': 'en',
        'model': 'nova-2',
        'smart_format': True,
        'punctuate': True
    }


@pytest.fixture
def mock_whisper_config():
    """Mock configuration for Whisper provider."""
    return {
        'model': 'tiny',
        'compute_type': 'int8',
        'language': 'en',
        'beam_size': 1,
        'use_faster_whisper': True
    }


@pytest.fixture
def mock_deepgram_response():
    """Mock Deepgram API response."""
    mock_response = Mock()
    mock_channel = Mock()
    mock_alternative = Mock()
    mock_alternative.transcript = "Hello world"
    mock_channel.alternatives = [mock_alternative]
    mock_response.results.channels = [mock_channel]
    return mock_response


@pytest.fixture
def mock_whisper_segments():
    """Mock Whisper transcription segments."""
    mock_segment = Mock()
    mock_segment.text = "Hello world"
    return [mock_segment], None


@pytest.fixture(autouse=True)
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'DEEPGRAM_API_KEY': 'test_deepgram_key',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        yield


@pytest.fixture
def mock_audio_file():
    """Mock audio file for testing."""
    mock_file = Mock()
    mock_file.read.return_value = b'mock_audio_data'
    mock_file.filename = 'test.wav'
    return mock_file
