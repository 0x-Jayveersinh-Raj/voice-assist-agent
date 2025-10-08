"""
Tests for Whisper STT provider.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestWhisperSTT:
    """Test cases for WhisperSTT."""
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_init_with_faster_whisper(self, mock_whisper_model):
        """Test initialization with faster-whisper."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        config = {'model': 'tiny', 'compute_type': 'int8'}
        provider = WhisperSTT(config)
        
        assert provider.model_name == 'tiny'
        assert provider.compute_type == 'int8'
        assert provider.language == 'en'
        assert provider.model_type == 'faster_whisper'
        mock_whisper_model.assert_called_once_with('tiny', compute_type='int8', device='cpu')
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', False)
    @patch('server.stt.whisper_provider.WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.whisper')
    def test_init_with_openai_whisper(self, mock_whisper):
        """Test initialization with OpenAI Whisper."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        config = {'model': 'base', 'use_faster_whisper': False}
        provider = WhisperSTT(config)
        
        assert provider.model_name == 'base'
        assert provider.model_type == 'openai_whisper"
        mock_whisper.load_model.assert_called_once_with('base')
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', False)
    @patch('server.stt.whisper_provider.WHISPER_AVAILABLE', False)
    def test_init_no_whisper_available(self):
        """Test initialization when no Whisper implementation is available."""
        from server.stt.whisper_provider import WhisperSTT
        
        with pytest.raises(ImportError, match="Neither faster-whisper nor openai-whisper is available"):
            WhisperSTT()
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_transcribe_faster_whisper(self, mock_whisper_model):
        """Test transcription with faster-whisper."""
        # Mock the model and its transcribe method
        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT({'model': 'tiny'})
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        result = provider.transcribe(audio_data)
        
        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', False)
    @patch('server.stt.whisper_provider.WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.whisper')
    @patch('server.stt.whisper_provider.sf.write')
    @patch('server.stt.whisper_provider.os.unlink')
    def test_transcribe_openai_whisper(self, mock_unlink, mock_sf_write, mock_whisper):
        """Test transcription with OpenAI Whisper."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Hello world"}
        mock_whisper.load_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT({'model': 'base', 'use_faster_whisper': False})
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'
            result = provider.transcribe(audio_data)
        
        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()
        mock_unlink.assert_called_once()
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_transcribe_error_handling(self, mock_whisper_model):
        """Test error handling in transcription."""
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT({'model': 'tiny'})
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        result = provider.transcribe(audio_data)
        
        assert result == ""
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_get_supported_languages(self, mock_whisper_model):
        """Test getting supported languages."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT()
        languages = provider.get_supported_languages()
        
        assert 'en' in languages
        assert 'hi' in languages
        assert 'gu' in languages
        assert isinstance(languages, list)
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_set_language_valid(self, mock_whisper_model):
        """Test setting valid language."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT()
        provider.set_language('hi')
        
        assert provider.language == 'hi'
        assert provider.config['language'] == 'hi'
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_set_language_auto(self, mock_whisper_model):
        """Test setting language to auto."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT()
        provider.set_language('auto')
        
        assert provider.language == 'auto'
        assert provider.config['language'] == 'auto'
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_set_language_invalid(self, mock_whisper_model):
        """Test setting invalid language."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT()
        
        with pytest.raises(ValueError, match="Language 'invalid' not supported"):
            provider.set_language('invalid')
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_update_model(self, mock_whisper_model):
        """Test updating model."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT({'model': 'tiny'})
        provider.update_model('base')
        
        assert provider.model_name == 'base'
        assert provider.config['model'] == 'base'
        # Should call _initialize_model again
        assert mock_whisper_model.call_count == 2
    
    @patch('server.stt.whisper_provider.FASTER_WHISPER_AVAILABLE', True)
    @patch('server.stt.whisper_provider.WhisperModel')
    def test_get_model_info(self, mock_whisper_model):
        """Test getting model information."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        from server.stt.whisper_provider import WhisperSTT
        
        provider = WhisperSTT({'model': 'tiny', 'compute_type': 'int8'})
        info = provider.get_model_info()
        
        assert info['model_name'] == 'tiny'
        assert info['model_type'] == 'faster_whisper'
        assert info['compute_type'] == 'int8'
        assert info['language'] == 'en'
        assert info['beam_size'] == 1
