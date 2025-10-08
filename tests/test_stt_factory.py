"""
Tests for STT factory.
"""
import pytest
from unittest.mock import Mock, patch
from server.stt.factory import STTFactory
from server.stt.base import STTProvider


class TestSTTFactory:
    """Test cases for STTFactory."""
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = STTFactory.get_available_providers()
        
        assert 'deepgram' in providers
        assert 'whisper' in providers
        assert isinstance(providers, list)
    
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider type"):
            STTFactory.create_provider('unsupported')
    
    @patch('server.stt.factory.DeepgramSTT')
    def test_create_deepgram_provider(self, mock_deepgram):
        """Test creating Deepgram provider."""
        mock_instance = Mock()
        mock_deepgram.return_value = mock_instance
        
        config = {'api_key': 'test_key'}
        provider = STTFactory.create_provider('deepgram', config)
        
        mock_deepgram.assert_called_once_with(config)
        assert provider == mock_instance
    
    @patch('server.stt.factory.WhisperSTT')
    def test_create_whisper_provider(self, mock_whisper):
        """Test creating Whisper provider."""
        mock_instance = Mock()
        mock_whisper.return_value = mock_instance
        
        config = {'model': 'tiny'}
        provider = STTFactory.create_provider('whisper', config)
        
        mock_whisper.assert_called_once_with(config)
        assert provider == mock_instance
    
    def test_register_provider(self):
        """Test registering a new provider."""
        class CustomSTT(STTProvider):
            def transcribe(self, audio_data, sample_rate=16000):
                return "custom"
            
            def transcribe_streaming(self, audio_chunk, sample_rate=16000):
                return "custom streaming"
            
            def get_supported_languages(self):
                return ['en']
            
            def set_language(self, language):
                pass
        
        STTFactory.register_provider('custom', CustomSTT)
        
        assert 'custom' in STTFactory.get_available_providers()
        
        provider = STTFactory.create_provider('custom')
        assert isinstance(provider, CustomSTT)
    
    def test_register_invalid_provider(self):
        """Test registering invalid provider class."""
        class InvalidProvider:
            pass
        
        with pytest.raises(ValueError, match="Provider class must inherit from STTProvider"):
            STTFactory.register_provider('invalid', InvalidProvider)
    
    @patch('server.stt.factory.DeepgramSTT')
    def test_get_provider_info(self, mock_deepgram):
        """Test getting provider information."""
        mock_instance = Mock()
        mock_instance.get_supported_languages.return_value = ['en', 'hi']
        mock_instance.get_config.return_value = {'language': 'en'}
        mock_deepgram.return_value = mock_instance
        
        info = STTFactory.get_provider_info('deepgram')
        
        assert info['name'] == 'deepgram'
        assert info['class'] == 'DeepgramSTT'
        assert info['supported_languages'] == ['en', 'hi']
        assert info['default_config'] == {'language': 'en'}
    
    def test_get_provider_info_invalid(self):
        """Test getting info for invalid provider."""
        with pytest.raises(ValueError, match="Provider type 'invalid' not found"):
            STTFactory.get_provider_info('invalid')
