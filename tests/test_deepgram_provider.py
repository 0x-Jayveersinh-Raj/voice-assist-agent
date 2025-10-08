"""
Tests for Deepgram STT provider.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestDeepgramSTT:
    """Test cases for DeepgramSTT."""
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_init_with_api_key(self, mock_deepgram_client):
        """Test initialization with API key."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        config = {'api_key': 'test_key', 'language': 'hi'}
        provider = DeepgramSTT(config)
        
        assert provider.language == 'hi'
        assert provider.model == 'nova-2'
        assert provider.smart_format is True
        assert provider.punctuate is True
        mock_deepgram_client.assert_called_once_with('test_key')
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    @patch.dict(os.environ, {'DEEPGRAM_API_KEY': 'env_key'})
    def test_init_with_env_key(self, mock_deepgram_client):
        """Test initialization with environment API key."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT()
        
        assert provider.language == 'en'
        mock_deepgram_client.assert_called_once_with('env_key')
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_init_no_api_key(self, mock_deepgram_client):
        """Test initialization without API key."""
        from server.stt.deepgram_provider import DeepgramSTT
        
        with pytest.raises(ValueError, match="Deepgram API key is required"):
            DeepgramSTT()
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    @patch('server.stt.deepgram_provider.sf.write')
    @patch('server.stt.deepgram_provider.os.unlink')
    def test_transcribe_success(self, mock_unlink, mock_sf_write, mock_deepgram_client):
        """Test successful transcription."""
        # Mock the Deepgram client and response
        mock_client = Mock()
        mock_listen = Mock()
        mock_prerecorded = Mock()
        mock_v = Mock()
        mock_transcribe = Mock()
        
        mock_response = Mock()
        mock_channel = Mock()
        mock_alternative = Mock()
        mock_alternative.transcript = "Hello world"
        mock_channel.alternatives = [mock_alternative]
        mock_response.results.channels = [mock_channel]
        mock_transcribe.return_value = mock_response
        
        mock_v.transcribe_file = mock_transcribe
        mock_prerecorded.v.return_value = mock_v
        mock_listen.prerecorded = mock_prerecorded
        mock_client.listen = mock_listen
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'
            result = provider.transcribe(audio_data)
        
        assert result == "Hello world"
        mock_transcribe.assert_called_once()
        mock_unlink.assert_called_once()
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    @patch('server.stt.deepgram_provider.sf.write')
    @patch('server.stt.deepgram_provider.os.unlink')
    def test_transcribe_no_results(self, mock_unlink, mock_sf_write, mock_deepgram_client):
        """Test transcription with no results."""
        # Mock the Deepgram client and response
        mock_client = Mock()
        mock_listen = Mock()
        mock_prerecorded = Mock()
        mock_v = Mock()
        mock_transcribe = Mock()
        
        mock_response = Mock()
        mock_response.results = None
        mock_transcribe.return_value = mock_response
        
        mock_v.transcribe_file = mock_transcribe
        mock_prerecorded.v.return_value = mock_v
        mock_listen.prerecorded = mock_prerecorded
        mock_client.listen = mock_listen
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'
            result = provider.transcribe(audio_data)
        
        assert result == ""
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    @patch('server.stt.deepgram_provider.sf.write')
    def test_transcribe_error_handling(self, mock_sf_write, mock_deepgram_client):
        """Test error handling in transcription."""
        mock_client = Mock()
        mock_listen = Mock()
        mock_prerecorded = Mock()
        mock_v = Mock()
        mock_transcribe = Mock()
        mock_transcribe.side_effect = Exception("API Error")
        
        mock_v.transcribe_file = mock_transcribe
        mock_prerecorded.v.return_value = mock_v
        mock_listen.prerecorded = mock_prerecorded
        mock_client.listen = mock_listen
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'
            result = provider.transcribe(audio_data)
        
        assert result == ""
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_transcribe_streaming(self, mock_deepgram_client):
        """Test streaming transcription."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        audio_chunk = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        
        with patch.object(provider, 'transcribe', return_value='streaming result') as mock_transcribe:
            result = provider.transcribe_streaming(audio_chunk)
        
        assert result == 'streaming result'
        mock_transcribe.assert_called_once_with(audio_chunk, 16000)
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_get_supported_languages(self, mock_deepgram_client):
        """Test getting supported languages."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        languages = provider.get_supported_languages()
        
        assert 'en' in languages
        assert 'hi' in languages
        assert 'gu' in languages
        assert isinstance(languages, list)
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_set_language_valid(self, mock_deepgram_client):
        """Test setting valid language."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        provider.set_language('hi')
        
        assert provider.language == 'hi'
        assert provider.config['language'] == 'hi'
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_set_language_invalid(self, mock_deepgram_client):
        """Test setting invalid language."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        
        with pytest.raises(ValueError, match="Language 'invalid' not supported"):
            provider.set_language('invalid')
    
    @patch('server.stt.deepgram_provider.DeepgramClient')
    def test_update_model(self, mock_deepgram_client):
        """Test updating model."""
        mock_client = Mock()
        mock_deepgram_client.return_value = mock_client
        
        from server.stt.deepgram_provider import DeepgramSTT
        
        provider = DeepgramSTT({'api_key': 'test_key'})
        provider.update_model('base')
        
        assert provider.model == 'base'
        assert provider.config['model'] == 'base'
