"""
Integration tests for STT system.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from server.stt.factory import STTFactory
from server.stt.base import STTProvider


class TestSTTIntegration:
    """Integration tests for the STT system."""
    
    def test_factory_creates_providers(self):
        """Test that factory can create all available providers."""
        providers = STTFactory.get_available_providers()
        
        for provider_type in providers:
            # Mock the providers to avoid actual initialization
            if provider_type == 'deepgram':
                with patch('server.stt.deepgram_provider.DeepgramClient'):
                    provider = STTFactory.create_provider(provider_type, {'api_key': 'test'})
            elif provider_type == 'whisper':
                with patch('server.stt.whisper_provider.WhisperModel'):
                    provider = STTFactory.create_provider(provider_type)
            
            assert isinstance(provider, STTProvider)
            assert hasattr(provider, 'transcribe')
            assert hasattr(provider, 'transcribe_streaming')
            assert hasattr(provider, 'get_supported_languages')
            assert hasattr(provider, 'set_language')
    
    def test_provider_interface_consistency(self):
        """Test that all providers implement the same interface."""
        providers = STTFactory.get_available_providers()
        
        for provider_type in providers:
            # Mock the providers to avoid actual initialization
            if provider_type == 'deepgram':
                with patch('server.stt.deepgram_provider.DeepgramClient'):
                    provider = STTFactory.create_provider(provider_type, {'api_key': 'test'})
            elif provider_type == 'whisper':
                with patch('server.stt.whisper_provider.WhisperModel'):
                    provider = STTFactory.create_provider(provider_type)
            
            # Test all required methods exist and are callable
            assert callable(provider.transcribe)
            assert callable(provider.transcribe_streaming)
            assert callable(provider.get_supported_languages)
            assert callable(provider.set_language)
            assert callable(provider.get_config)
            assert callable(provider.update_config)
            
            # Test method signatures with sample data
            audio_data = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
            
            # These should not raise exceptions (even if they return empty strings)
            result1 = provider.transcribe(audio_data)
            result2 = provider.transcribe_streaming(audio_data)
            languages = provider.get_supported_languages()
            config = provider.get_config()
            
            assert isinstance(result1, str)
            assert isinstance(result2, str)
            assert isinstance(languages, list)
            assert isinstance(config, dict)
    
    def test_provider_configuration_persistence(self):
        """Test that provider configurations persist correctly."""
        providers = STTFactory.get_available_providers()
        
        for provider_type in providers:
            # Mock the providers to avoid actual initialization
            if provider_type == 'deepgram':
                with patch('server.stt.deepgram_provider.DeepgramClient'):
                    provider = STTFactory.create_provider(provider_type, {'api_key': 'test'})
            elif provider_type == 'whisper':
                with patch('server.stt.whisper_provider.WhisperModel'):
                    provider = STTFactory.create_provider(provider_type)
            
            # Test configuration updates
            initial_config = provider.get_config()
            provider.update_config({'test_key': 'test_value'})
            updated_config = provider.get_config()
            
            assert 'test_key' in updated_config
            assert updated_config['test_key'] == 'test_value'
            assert updated_config != initial_config
    
    def test_language_support_consistency(self):
        """Test that all providers support common languages."""
        common_languages = ['en', 'hi', 'gu']
        providers = STTFactory.get_available_providers()
        
        for provider_type in providers:
            # Mock the providers to avoid actual initialization
            if provider_type == 'deepgram':
                with patch('server.stt.deepgram_provider.DeepgramClient'):
                    provider = STTFactory.create_provider(provider_type, {'api_key': 'test'})
            elif provider_type == 'whisper':
                with patch('server.stt.whisper_provider.WhisperModel'):
                    provider = STTFactory.create_provider(provider_type)
            
            supported_languages = provider.get_supported_languages()
            
            # All providers should support English
            assert 'en' in supported_languages
            
            # Test setting supported languages
            for lang in common_languages:
                if lang in supported_languages:
                    provider.set_language(lang)
                    assert provider.config.get('language') == lang
    
    def test_error_handling_consistency(self):
        """Test that all providers handle errors gracefully."""
        providers = STTFactory.get_available_providers()
        
        for provider_type in providers:
            # Mock the providers to avoid actual initialization
            if provider_type == 'deepgram':
                with patch('server.stt.deepgram_provider.DeepgramClient'):
                    provider = STTFactory.create_provider(provider_type, {'api_key': 'test'})
            elif provider_type == 'whisper':
                with patch('server.stt.whisper_provider.WhisperModel'):
                    provider = STTFactory.create_provider(provider_type)
            
            # Test with invalid audio data
            invalid_audio = np.array([])  # Empty array
            
            # Should not raise exceptions, should return empty string or handle gracefully
            result = provider.transcribe(invalid_audio)
            assert isinstance(result, str)
            
            # Test with None audio (should handle gracefully)
            try:
                result = provider.transcribe(None)
                assert isinstance(result, str)
            except (TypeError, AttributeError):
                # Some providers might raise exceptions for None input, which is acceptable
                pass
    
    def test_provider_info_retrieval(self):
        """Test that provider info can be retrieved for all providers."""
        providers = STTFactory.get_available_providers()
        
        for provider_type in providers:
            info = STTFactory.get_provider_info(provider_type)
            
            assert 'name' in info
            assert 'class' in info
            assert info['name'] == provider_type
            
            # Should have either supported_languages or error
            assert 'supported_languages' in info or 'error' in info
    
    def test_custom_provider_registration(self):
        """Test registering and using a custom provider."""
        class CustomSTT(STTProvider):
            def __init__(self, config=None):
                super().__init__(config)
                self.transcription_count = 0
            
            def transcribe(self, audio_data, sample_rate=16000):
                self.transcription_count += 1
                return f"Custom transcription {self.transcription_count}"
            
            def transcribe_streaming(self, audio_chunk, sample_rate=16000):
                return "Custom streaming"
            
            def get_supported_languages(self):
                return ['en', 'custom']
            
            def set_language(self, language):
                self.config['language'] = language
        
        # Register custom provider
        STTFactory.register_provider('custom', CustomSTT)
        
        # Create and test custom provider
        provider = STTFactory.create_provider('custom', {'test': 'value'})
        
        assert isinstance(provider, CustomSTT)
        assert provider.config['test'] == 'value'
        
        audio_data = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        result = provider.transcribe(audio_data)
        assert result == "Custom transcription 1"
        
        # Test that it's now available in the factory
        assert 'custom' in STTFactory.get_available_providers()
        
        # Test provider info
        info = STTFactory.get_provider_info('custom')
        assert info['name'] == 'custom'
        assert info['class'] == 'CustomSTT'
