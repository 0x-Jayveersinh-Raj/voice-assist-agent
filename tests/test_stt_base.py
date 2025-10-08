"""
Tests for STT base class.
"""
import pytest
import numpy as np
from unittest.mock import Mock
from server.stt.base import STTProvider


class MockSTTProvider(STTProvider):
    """Mock STT provider for testing."""
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        return "mock transcription"
    
    def transcribe_streaming(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        return "mock streaming transcription"
    
    def get_supported_languages(self) -> list:
        return ['en', 'hi', 'gu']
    
    def set_language(self, language: str) -> None:
        self.config['language'] = language


class TestSTTProvider:
    """Test cases for STTProvider base class."""
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {'language': 'en', 'model': 'test'}
        provider = MockSTTProvider(config)
        
        assert provider.config == config
        assert provider.get_config() == config
    
    def test_init_without_config(self):
        """Test initialization without configuration."""
        provider = MockSTTProvider()
        
        assert provider.config == {}
        assert provider.get_config() == {}
    
    def test_update_config(self):
        """Test configuration updates."""
        provider = MockSTTProvider()
        
        provider.update_config({'language': 'hi'})
        assert provider.config['language'] == 'hi'
        
        provider.update_config({'model': 'test'})
        assert provider.config['language'] == 'hi'
        assert provider.config['model'] == 'test'
    
    def test_transcribe(self):
        """Test transcription method."""
        provider = MockSTTProvider()
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        result = provider.transcribe(audio_data)
        assert result == "mock transcription"
    
    def test_transcribe_streaming(self):
        """Test streaming transcription method."""
        provider = MockSTTProvider()
        audio_chunk = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        
        result = provider.transcribe_streaming(audio_chunk)
        assert result == "mock streaming transcription"
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        provider = MockSTTProvider()
        
        languages = provider.get_supported_languages()
        assert languages == ['en', 'hi', 'gu']
        assert isinstance(languages, list)
    
    def test_set_language(self):
        """Test setting language."""
        provider = MockSTTProvider()
        
        provider.set_language('hi')
        assert provider.config['language'] == 'hi'
