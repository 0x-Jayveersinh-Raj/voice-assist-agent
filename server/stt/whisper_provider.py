"""
Whisper STT provider implementation.
"""
import os
import tempfile
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf
from .base import STTProvider

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WhisperSTT(STTProvider):
    """Whisper Speech-to-Text provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Whisper STT provider.
        
        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__(config)
        
        if not FASTER_WHISPER_AVAILABLE and not WHISPER_AVAILABLE:
            raise ImportError("Neither faster-whisper nor openai-whisper is available. Please install one of them.")
        
        # Configuration
        self.model_name = self.config.get('model', 'tiny')
        self.compute_type = self.config.get('compute_type', 'int8')
        self.language = self.config.get('language', 'en')
        self.beam_size = self.config.get('beam_size', 1)
        self.use_faster_whisper = self.config.get('use_faster_whisper', FASTER_WHISPER_AVAILABLE)
        
        # Initialize model
        self._initialize_model()
        
        # Supported languages (Whisper supports many languages)
        self.supported_languages = [
            'en', 'hi', 'gu', 'bn', 'ta', 'te', 'ml', 'kn', 'mr', 'pa', 'or', 'as',
            'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'tr', 'pl',
            'nl', 'sv', 'da', 'no', 'fi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl',
            'et', 'lv', 'lt', 'mt', 'cy', 'ga', 'is', 'mk', 'sq', 'sr', 'uk', 'be',
            'ka', 'hy', 'az', 'kk', 'ky', 'uz', 'mn', 'ne', 'si', 'my', 'km', 'lo',
            'th', 'vi', 'id', 'ms', 'tl', 'sw', 'am', 'ha', 'ig', 'yo', 'zu', 'xh',
            'af', 'eu', 'ca', 'gl', 'he', 'fa', 'ur', 'ps', 'sd', 'dv', 'bo', 'dz',
            'ti', 'om', 'so', 'rw', 'rn', 'ny', 'sn', 'st', 'tn', 'ts', 've', 'xh',
            'zu', 'ss', 'nr', 'nso', 'zu', 'xh', 'af', 'sq', 'eu', 'ca', 'gl', 'is',
            'ga', 'cy', 'mt', 'mk', 'sq', 'sr', 'uk', 'be', 'ka', 'hy', 'az', 'kk',
            'ky', 'uz', 'mn', 'ne', 'si', 'my', 'km', 'lo', 'th', 'vi', 'id', 'ms',
            'tl', 'sw', 'am', 'ha', 'ig', 'yo', 'zu', 'xh', 'af'
        ]
    
    def _initialize_model(self):
        """Initialize the Whisper model."""
        try:
            if self.use_faster_whisper and FASTER_WHISPER_AVAILABLE:
                self.model = WhisperModel(
                    self.model_name, 
                    compute_type=self.compute_type,
                    device="cpu"  # Use CPU for compatibility
                )
                self.model_type = "faster_whisper"
            elif WHISPER_AVAILABLE:
                self.model = whisper.load_model(self.model_name)
                self.model_type = "openai_whisper"
            else:
                raise ImportError("No Whisper implementation available")
                
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            if self.model_type == "faster_whisper":
                # Use faster-whisper
                segments, _ = self.model.transcribe(
                    audio_float, 
                    language=self.language if self.language != 'auto' else None,
                    beam_size=self.beam_size
                )
                transcript = "".join(segment.text for segment in segments)
                return transcript.strip()
                
            else:
                # Use OpenAI Whisper
                # Save to temporary file for OpenAI Whisper
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_float, sample_rate)
                    
                    # Transcribe
                    result = self.model.transcribe(
                        temp_file.name,
                        language=self.language if self.language != 'auto' else None
                    )
                    
                    # Clean up
                    os.unlink(temp_file.name)
                    
                    return result["text"].strip()
                    
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return ""
    
    def transcribe_streaming(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a streaming audio chunk using Whisper.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text for this chunk
        """
        return self.transcribe(audio_chunk, sample_rate)
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def set_language(self, language: str) -> None:
        """
        Set the language for transcription.
        
        Args:
            language: Language code or 'auto' for automatic detection
        """
        if language == 'auto' or language in self.supported_languages:
            self.language = language
            self.config['language'] = language
        else:
            raise ValueError(f"Language '{language}' not supported. Use 'auto' or one of: {self.supported_languages}")
    
    def update_model(self, model_name: str) -> None:
        """
        Update the Whisper model.
        
        Args:
            model_name: Model name (e.g., 'tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_name = model_name
        self.config['model'] = model_name
        self._initialize_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'compute_type': self.compute_type if self.model_type == "faster_whisper" else None,
            'language': self.language,
            'beam_size': self.beam_size
        }
